import os
import numpy
import segyio
from PIL import Image

# .dat file generate iamge
def datToimage(src,saveDir):

    image_name_array=os.listdir(src)
    image_src_array=[os.path.join(src,i) for i in image_name_array]

    for i in range(len(src)):
        image_numpy=numpy.fromfile(src[i],dtype=numpy.single)
        print(image_numpy.shape)
        image_numpy=numpy.reshape(image_numpy,(128,128,128))*255
        print(image_numpy.shape)
        for j in range(128):
            image=Image.fromarray(image_numpy[:,:,j]).convert("L")
            image.show()
            image.save(os.path.join(saveDir,image_name_array[i].split(".")[0]+"_"+str(j)+".png"))
            print(i,j)

# .segy file generate iamge
def segyToimage(src,saveDir):
    image_name_array = os.listdir(src)
    image_src_array = [os.path.join(src, i) for i in image_name_array]

    for i in range(len(image_name_array)):
        filename=image_src_array[i]
        with segyio.open(filename) as f:
            inline_number = f.inlines[320]
            inline_slice = f.inline[inline_number]
            # inline_slice = f.depth_slice[352]
            image = Image.fromarray(inline_slice.T).convert("L")
            image.show()
            image.save(os.path.join(saveDir,image_name_array[i].split(".")[0]+"_"+str(j)+".png"))
