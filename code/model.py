## Imports

import random
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.optimizers import *
from keras.backend import *
from keras.layers import *

## Seeding
seed = 2022
random.seed = seed
np.random.seed = seed
tf.seed = seed

## Hyperparameters
image_size = 128
batch_size = 16
val_data_size = 200

def binary_focal_loss(gamma=2, alpha=0.25):

    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true *alpha + (ones_like(y_true ) -y_true ) *( 1 -alpha)

        p_t = y_true *y_pred + (ones_like(y_true ) -y_true ) *(ones_like(y_true ) -y_pred) + epsilon()
        focal_loss = - alpha_t * pow((ones_like(y_true ) -p_t) ,gamma) * log(p_t)
        return mean(focal_loss)
    return binary_focal_loss_fixed

#ASPP
def ASPP(x, rate1, rate2, rate3, rate4, channel):
    ##第一层
    layer1_1 = keras.layers.Conv2D(channel, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(rate1, rate1))(x)
    layer1_2 = keras.layers.Conv2D(channel, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(rate2, rate2))(x)
    layer1_3 = keras.layers.Conv2D(channel, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(rate3, rate3))(x)
    layer1_4 = keras.layers.Conv2D(channel, kernel_size=(3, 3), strides=(1, 1), padding='same', dilation_rate=(rate4, rate4))(x)
    ##第二层
    layer1_1 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_1)
    layer1_2 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_2)
    layer1_3 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_3)
    layer1_4 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_4)
    # 第三层
    layer1_1 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_1)
    layer1_2 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_2)
    layer1_3 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_3)
    layer1_4 = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=(1, 1), padding='same')(layer1_4)

    output = concatenate([layer1_1,layer1_2,layer1_3,layer1_4,x],axis=3)
    output = keras.layers.Conv2D(channel,kernel_size=(1,1),strides=(1,1),padding='same')(output)
    return output

## Attention
# 判断输入数据格式，是channels_first还是channels_last
channel_axis = 1 if image_data_format() == "channels_first" else 3

# CAM
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = GlobalMaxPooling2D()(input_xs)
    maxpool_channel = Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = GlobalAvgPool2D()(input_xs)
    avgpool_channel = Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = Activation('sigmoid')(channel_attention_feature)
    return Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = Lambda(lambda x: max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = Lambda(lambda x: mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)

def csam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = Multiply()([channel_refined_feature, spatial_attention_feature])
    return Add()([refined_feature, input_xs])


## Blocks
def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x


def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv


def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([conv, shortcut])
    return output


def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = keras.layers.Add()([shortcut, res])
    return output


def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    pu= spatial_attention(u)
    c = keras.layers.Concatenate()([pu, xskip])
    return c


def GFFResUNet(pretrainedWeights=None,optimizerfunction="Adam(lr=1e-4)"):
    f = [64, 128, 256 ,512 ,1024]
    inputs = keras.layers.Input((image_size, image_size, 3))

    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    c1= csam_module(e1)
    e2 = residual_block(e1, f[1], strides=2)
    c2= csam_module(e2)
    e3 = residual_block(e2, f[2], strides=2)
    c3= csam_module(e3)
    e4 = residual_block(e3, f[3], strides=2)
    c4= csam_module(e4)
    e5 = residual_block(e4, f[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    a = ASPP(b1, 2, 4, 6, 8, 512)

    ## Decoder
    u1 = upsample_concat_block( a, c4)
    d1 = conv_block(u1, f[4])

    u2 = upsample_concat_block( d1, c3)
    d2 = conv_block(u2, f[3])

    u3 = upsample_concat_block(d2, c2)
    d3 = conv_block(u3, f[2])

    u4 = upsample_concat_block( d3, c1)
    d4 = conv_block(u4, f[1])

    outputs = keras.layers.Conv2D(2, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)

    if optimizerfunction == "Adam(lr=1e-4)":
        optimizerfunction = Adam(lr=1e-4)
    else:
        optimizerfunction = Adam(lr=1e-3)
    model.compile(optimizer=optimizerfunction, loss=[binary_focal_loss(2,0.25)], metrics=['accuracy'])
    if (pretrainedWeights):
        model.load_weights(pretrainedWeights)
    return model










