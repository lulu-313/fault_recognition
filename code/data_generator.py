# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 16:18:46 2022

@author: lulu
## Fault Train

1. Create 1D reflection model
2. Apply Gaussian deformation
3. Apply planar deformation
4. Add fault throws
5. Convolve reflection model with a wavelet
6. Add some random noise
7. Extract the central part of the volume in the size of 128x128x128
8. Standardize amplitudes within the image
"""

import os
import numpy as np
import cupy as cp
import bruges
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.interpolate import RegularGridInterpolator


# Parameters
class DefineParams():
    def __init__(self, num_data, patch_size):
        ' Feature patch '
        nx, ny, nz = ([patch_size] * 3)
        nxy = nx * ny
        nxyz = nxy * nz
        self.nx = nx  # Number of elements in x
        self.ny = ny  # Number of elements in y
        self.nz = nz  # Number of elements in z
        self.nxy = nxy  # Number of elements in xy
        self.nxyz = nxyz  # Number of elements in xyz
        self.num_data = num_data  # Number of synthetic data
        self.x0 = int(nx / 2)  # Center of x location
        self.y0 = int(ny / 2)  # Center of y location
        self.z0 = int(nz / 2)  # Center of z location

        ' Synthetic traces '
        size_tr = 200
        nx_tr, ny_tr, nz_tr = ([size_tr] * 3)
        nxy_tr = nx_tr * ny_tr
        nxyz_tr = nxy_tr * nz_tr
        x = np.linspace(0, nx_tr - 1, nx_tr)
        y = np.linspace(0, nx_tr - 1, ny_tr)
        z = np.linspace(0, nz_tr - 1, nz_tr)
        xy = np.reshape(np.array([np.meshgrid(x, y, indexing='ij')]), [2, nxy_tr]).T
        xyz = np.reshape(np.array([np.meshgrid(x, y, z, indexing='ij')]), [3, nxyz_tr]).T
        self.x = x  # x
        self.y = y  # y
        self.z = z  # z
        self.xy = xy  # xy grid (x: xy[:,0], y: xy[:,1])
        self.xyz = xyz  # xyz grid (x: xyz[:,0], y: xyz[:,1], z: xyz[:,2])
        self.nx_tr = nx_tr  # Number of elements in x
        self.ny_tr = ny_tr  # Number of elements in y
        self.nz_tr = nz_tr  # Number of elements in z
        self.nxy_tr = nx_tr * ny_tr  # Number of elements in xy
        self.nxyz_tr = nxy_tr * nz_tr  # Number of elements in xyz
        self.x0_tr = int(nx_tr / 2)  # Trace center of traces in x
        self.y0_tr = int(ny_tr / 2)  # Synthetic Traces: y center
        self.z0_tr = int(nz_tr / 2)  # Synthetic Traces: z center


prm = DefineParams(num_data=1, patch_size=128)


# 1.Start from creating 1D synthetic reflectivity model
def create_1d_model(prm):
    num_rand = int(prm.nz_tr * 0.5)
    idx_refl = np.random.randint(0, prm.nz_tr, num_rand)
    refl = np.zeros(prm.nz_tr)
    refl[idx_refl] = 2 * np.random.rand(num_rand) - 1
    refl = np.tile(refl, [prm.nxy_tr, 1])
    return refl


def show_img(img, size=200, idx_img=150):
    plt.imshow(np.reshape(img, [size] * 3)[:, idx_img, :].T, cmap=plt.cm.gray_r)
    plt.colorbar()
    plt.show()


refl = create_1d_model(prm)
show_img(refl)


# 2.Apply 2D Gaussian deformation
def func_gauss2d(prm, refl, a, b, c, d, sigma):
    ''' Apply 2D Gaussian deformation '''
    xy_cp = cp.asarray(prm.xy)
    refl_cp = cp.asarray(refl)
    a_cp = cp.asarray(a.astype('float64'))
    b_cp = cp.asarray(b.astype('float64'))
    c_cp = cp.asarray(c.astype('float64'))
    d_cp = cp.asarray(d.astype('float64'))
    sigma_cp = cp.asarray(sigma.astype('float64'))
    z_cp = cp.asarray(prm.z)

    # Parallelize computation on GPU using cupy
    func_gauss2d = cp.ElementwiseKernel(
        in_params='T x, T y, T b, T c, T d, T sigma',
        out_params='T z',
        operation=''' z = b*expf(-(powf(x-c,2) + powf(y-d,2))/(2*powf(sigma,2))); ''',
        name='func_gauss2d'
    )

    gauss_2d_cp = cp.zeros_like(xy_cp[:, 0])
    for i in range(len(b)):
        gauss_2d_cp += func_gauss2d(xy_cp[:, 0], xy_cp[:, 1], b_cp[i], c_cp[i], d_cp[i], sigma_cp[i])
    s1_cp = a_cp + (1.5 / z_cp) * cp.outer(cp.transpose(gauss_2d_cp), z_cp)

    for i in range(prm.nxy_tr):
        s = s1_cp[i, :] + z_cp
        mat = cp.tile(z_cp, (len(s), 1)) - cp.tile(cp.expand_dims(s, 1), (1, len(z_cp)))
        refl_cp[i, :] = cp.dot(refl_cp[i, :], cp.sinc(mat))

    refl = np.reshape(cp.asnumpy(refl_cp), [prm.nxy_tr, prm.nz_tr])
    return refl


a = np.array([5])  # Offset
b = np.array([15, 5])  # Magnitude of deformation
c = np.array([46, 96])  # x0 location
d = np.array([146, 46])  # y0 location
sigma = np.array([20, 10])  # Size of deformation radius
refl = func_gauss2d(prm, refl, a, b, c, d, sigma)
show_img(refl)


# 3.Apply planar deformation
def func_planar(prm, refl, e, f, g):
    ''' Apply planar deformation '''
    xy_cp = cp.asarray(prm.xy)
    refl_cp = cp.asarray(refl)
    e_cp = cp.asarray(e.astype('float64'))
    f_cp = cp.asarray(f.astype('float64'))
    g_cp = cp.asarray(g.astype('float64'))
    z_cp = cp.asarray(prm.z)

    # Parallelize computation on GPU using cupy
    func_planar = cp.ElementwiseKernel(
        in_params='T x, T y, T e, T f, T g',
        out_params='T z',
        operation=''' z = e + f*x + g*y; ''',
        name='func_planar'
    )

    s2_cp = func_planar(xy_cp[:, 0], xy_cp[:, 1], e_cp, f_cp, g_cp)

    for i in range(prm.nxy_tr):
        s = s2_cp[i] + z_cp
        mat = cp.tile(z_cp, (len(s), 1)) - cp.tile(cp.expand_dims(s, 1), (1, len(z_cp)))
        refl_cp[i, :] = cp.dot(refl_cp[i, :], cp.sinc(mat))

    refl = np.reshape(cp.asnumpy(refl_cp), [prm.nxy_tr, prm.nz_tr])
    return refl


e = np.array([0.1])  # Intercept of the plane
f = np.array([0.2])  # Slope of the plane in x direction
g = np.array([0.05])  # Slope of the plane in y direction
refl = func_planar(prm, refl, e, f, g)
show_img(refl)


# 4.Add fault throw with linear offset increase
def displace_trace(refl, labels, dip, strike, throw, x0_f, y0_f, z0_f, type_flt, i):
    # z values on a fault plane
    theta = dip / 180 * np.pi
    phi = strike / 180 * np.pi
    x, y, z = prm.xyz[:, 0], prm.xyz[:, 1], prm.xyz[:, 2]
    z_flt_plane = z_proj(x, y, z, x0_f, y0_f, z0_f, theta, phi)
    idx_repl = prm.xyz[:, 2] <= z_flt_plane
    z_shift, flag_offset = fault_throw(theta, phi, throw, z0_f, type_flt, prm)
    x1 = prm.xyz[:, 0] - np.tile(z_shift, prm.nxy_tr) * np.cos(theta) * np.cos(phi)
    y1 = prm.xyz[:, 1] - np.tile(z_shift, prm.nxy_tr) * np.cos(theta) * np.sin(phi)
    z1 = prm.xyz[:, 2] - np.tile(z_shift, prm.nxy_tr) * np.sin(theta)

    # Fault throw
    refl = refl.copy()
    refl = replace(refl, idx_repl, x1, y1, z1, prm)
    refl = np.reshape(refl, [prm.nxy_tr, prm.nz_tr])

    # Fault Label
    if i > 0:
        labels = replace(labels, idx_repl, x1, y1, z1, prm)
        labels[labels > 0.4] = 1
        labels[labels <= 0.4] = 0
    flt_flag = (0.5 * np.tan(dip / 180 * np.pi) > abs(z - z_flt_plane)) & flag_offset
    labels[flt_flag] = 1
    return refl, labels


def z_proj(x, y, z, x0_f, y0_f, z0_f, theta, phi):
    x1 = x0_f + (prm.nx_tr - prm.nx) / 2
    y1 = y0_f + (prm.ny_tr - prm.ny) / 2
    z1 = z0_f + (prm.nz_tr - prm.nz) / 2
    z_flt_plane = z1 + (np.cos(phi) * (x - x1) + np.sin(phi) * (y - y1)) * np.tan(theta)
    return z_flt_plane


def replace(xyz0, idx_repl, x1, y1, z1, prm):
    """ Replace """
    xyz1 = np.reshape(xyz0.copy(), [prm.nx_tr, prm.ny_tr, prm.nz_tr])
    func_3d_interp = RegularGridInterpolator((prm.x, prm.y, prm.z), xyz1, method='linear',
                                             bounds_error=False, fill_value=0)
    idx_interp = np.reshape(idx_repl, prm.nxyz_tr)
    xyz1 = np.reshape(xyz1, prm.nxyz_tr)
    xyz1[idx_interp] = func_3d_interp((x1[idx_interp], y1[idx_interp], z1[idx_interp]))
    return xyz1


def fault_throw(theta, phi, throw, z0_f, type_flt, prm):
    """ Define z shifts"""
    z1 = (prm.nz_tr - prm.nz) / 2 + z0_f
    z2 = (prm.nz_tr - prm.nz) / 2 + prm.nz
    z3 = (prm.nz_tr - prm.nz) / 2
    if type_flt == 0:  # Linear offset
        if throw > 0:  # Normal fault
            z_shift = throw * np.cos(theta) * (prm.z - z1) / (z2 - z1)
            z_shift[z_shift < 0] = 0
        else:  # Reverse fault
            z_shift = throw * np.cos(theta) * (prm.z - z1) / (z3 - z1)
            z_shift[z_shift > 0] = 0
    else:  # Gaussian offset
        gaussian1d = lambda z, sigma: throw * np.sin(theta) * np.exp(-(z - z1) ** 2 / (2 * sigma ** 2))
        z_shift = gaussian1d(prm.z, sigma=20)

    """ Flag offset """
    flag_offset = np.zeros([prm.nxy_tr, prm.nz_tr], dtype=bool)
    for i in range(prm.nxy_tr):
        flag_offset[i, :] = np.abs(z_shift) > 1
    flag_offset = np.reshape(flag_offset, prm.nxyz_tr)
    return z_shift, flag_offset


type_flt = np.array([0])  # Fault type (0: Linear throw, 1: Gaussian throw)
x0_f = np.array([50])  # x-location of fault (Gaussian: center, Linear: start point)
y0_f = np.array([0])  # y-location of fault
z0_f = np.array([0])  # z-location of fault
throw = np.array([30])  # Fault throw (Normal fault: > 0, Reverse fault: < 0)
dip = np.array([70])  # Fault dip (0 deg < dip < 90 deg)
strike = np.array([0])  # Fault strike (0 deg < strike <= 360 deg)

labels = np.zeros(prm.nxyz_tr)
for i in range(len(throw)):
    refl, labels = displace_trace(refl, labels, dip[i], strike[i], throw[i],
                                  x0_f[i], y0_f[i], z0_f[i], type_flt[i], i)
show_img(refl)


# 5.Convolve reflectivity model with a Ricker wavelet
def convolve_wavelet(prm, refl):
    traces = np.zeros([prm.nxy_tr, prm.nz_tr])
    wl = bruges.filters.wavelets.ricker(t_lng, dt, 30)
    for i in range(prm.nxy_tr):
        traces[i, :] = np.convolve(refl[i, :], wl, mode='same')
    return traces


dt = 0.004  # Sampling interval (ms)
t_lng = 0.082  # Length of Ricker wavelet in ms
traces = convolve_wavelet(prm, refl)
show_img(traces)


# 6.Add some noise to traces to imitate real seismic data
def add_noise(traces, snr, f0):
    nyq = 1 / dt / 2
    low = lcut / nyq
    high = hcut / nyq
    b, a = butter(order, [low, high], btype='band')
    for i in range(prm.nxy_tr):
        noise = bruges.noise.noise_db(traces[i, :], 3)
        traces[i, :] = filtfilt(b, a, traces[i, :] + noise)
    return traces


snr = np.array([3])  # Signal Noise Ratio
f0 = np.array([30])  # Central frequency
dt = 0.004  # Sampling interval (ms)
lcut = 5  # Bandpass filter: Lower cutoff
hcut = 80  # Bandpass filter: Upper cutoff
order = 5  # Order of Butterworth Filter
traces = add_noise(traces, snr, f0)
show_img(traces)


# Extract the central part in the input size
def crop_center_patch(prm, traces, labels):
    def func_crop(xyz):
        xyz = np.reshape(xyz, [prm.nx_tr, prm.ny_tr, prm.nz_tr])
        xyz_crop = xyz[prm.x0_tr - prm.x0:prm.x0_tr + prm.x0,
                   prm.y0_tr - prm.y0:prm.y0_tr + prm.y0,
                   prm.z0_tr - prm.z0:prm.z0_tr + prm.z0]
        return np.reshape(xyz_crop, [prm.nxy, prm.nz])

    traces = func_crop(traces)
    labels = np.reshape(labels, [prm.nxy_tr, prm.nz_tr])
    labels = func_crop(labels)
    return traces, labels


traces, labels = crop_center_patch(prm, traces, labels)
show_img(traces, 128, 150 - 36)


# Standardize amplitudes within the image
def standardizer(traces):
    std_func = lambda x: (x - np.mean(x)) / np.std(x)
    tr_std = std_func(traces)
    tr_std[tr_std > 1] = 1
    tr_std[tr_std < -1] = -1
    traces = tr_std
    return traces


traces = standardizer(traces)
show_img(traces, 128, 150 - 36)


# Display x-, y-, and z- slices with and without fault label
def CreateImgAlpha(img_input):
    img_alpha = np.zeros([np.shape(img_input)[0], np.shape(img_input)[1], 4])
    # Yellow: (1,1,0), Red: (1,0,0)
    img_alpha[:, :, 0] = 1
    img_alpha[:, :, 1] = 0
    img_alpha[:, :, 2] = 0
    img_alpha[..., -1] = img_input
    return img_alpha


def show_img_slice(seis_slice, fault_slice, title, cmap_bg=plt.cm.gray_r, i=63):
    plt.figure(figsize=(8, 12))
    for j in range(2):
        plt.subplot(int(221 + j))
        plt.imshow(seis_slice.T, cmap_bg)
        if j == 1:
            img_alpha = CreateImgAlpha(fault_slice.T)
            plt.imshow(img_alpha, alpha=1)
            plt.title(title + ' with fault')
        else:
            plt.title(title)
        plt.tight_layout()
    plt.show()


seis_vlm = np.reshape(traces, [128] * 3)
fault_vlm = np.reshape(labels, [128] * 3)
idx = 100

show_img_slice(seis_vlm[idx, :, :], fault_vlm[idx, :, :], 'x-slice')
show_img_slice(seis_vlm[:, idx, :], fault_vlm[:, idx, :], 'y-slice')
show_img_slice(seis_vlm[:, :, idx], fault_vlm[:, :, idx], 'z-slice')


# Show one example of synthetic volume with multiple faults
def load_data_synth(path_dataset, name_file, size_vlm):
    ''' Load already generated data '''
    path_seis = os.path.join(path_dataset, 'seis', name_file)
    path_fault = os.path.join(path_dataset, 'fault', name_file)
    path_pred = os.path.join(path_dataset, 'pred', name_file)
    seis_vlm = np.fromfile(path_seis, dtype=np.single)
    fault_vlm = np.fromfile(path_fault, dtype=np.single)
    pred_vlm = np.fromfile(path_pred, dtype=np.single)
    seis_vlm = np.reshape(seis_vlm, size_vlm)
    fault_vlm = np.reshape(fault_vlm, size_vlm)
    pred_vlm = np.reshape(pred_vlm, size_vlm)
    return seis_vlm, fault_vlm, pred_vlm


path_dataset = './dataset/12.07.2019/train'
seis_vlm, fault_vlm, pred_vlm = load_data_synth(path_dataset, '100.dat', (128, 128, 128))

show_img_slice(seis_vlm[idx, :, :], fault_vlm[idx, :, :], 'x-slice')
show_img_slice(seis_vlm[:, idx, :], fault_vlm[:, idx, :], 'y-slice')
show_img_slice(seis_vlm[:, :, idx], fault_vlm[:, :, idx], 'z-slice')