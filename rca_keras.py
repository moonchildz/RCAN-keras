import numpy as np
import keras
import keras.layers as L
import keras.models as M
import keras.optimizers as O
import keras.backend as B
import keras.backend.tensorflow_backend as B_
import tensorflow as tf
from keras.utils import multi_gpu_model
import os
import psutil
import datetime
import cv2
import random as rnd
from keras.utils import Sequence as Seq

_rgb_to_YCbCr_kernel = [[65.738 / 256, -37.945 / 256, 112.439 / 256],
                        [129.057 / 256, -74.494 / 256, -94.154 / 256],
                        [25.064 / 256, 112.439 / 256, -18.214 / 256]]
def rgb2ycbcr(img):
    ycc = np.zeros(img.shape)
    ycc[:,:,0] = 16 + img[:,:,2]*65.738 / 256 + img[:,:,1]*129.057 / 256 + img[:,:,0]* 25.064 / 256
    ycc[:,:,1] = 128+ img[:,:,2]*-37.94 / 256 + img[:,:,1]*-74.494 / 256 + img[:,:,0]* 112.439 / 256
    ycc[:, :, 1] = 128 + img[:, :, 2] * 112.439/ 256 + img[:, :, 1] * -94.154 / 256 + img[:, :, 0] * -18.214 / 256
    return ycc

route_lrhr = 'D:/lrhr_dataset/'
route_save = route_lrhr+'rcan.hdf5'
w_low =48
w_high = 96
def data_loader(route_lrhr):
    route_low = route_lrhr+'DIV2K_train_LR_bicubic/X2/'
    route_high = route_lrhr+'DIV2K_train_HR/'
    list_low = os.listdir(route_low)
    list_high = os.listdir(route_high)
    list_limg=[]
    list_himg=[]
    for l,h in zip(list_low,list_high):
        img_l = rgb2ycbcr(cv2.imread(route_low+l,-1))/255.0
        img_h = rgb2ycbcr(cv2.imread(route_high+h,-1))/255.0
        for v in range(24,img_l.shape[0]-24,rnd.randint(40,55)):
            for u in range(24, img_l.shape[1] - 24, rnd.randint(40, 55)):
                v2 = v*2
                u2 = u*2
                patch_l = img_l[v-24:v+24,u-24:u+24].astype(np.float32)
                patch_h = img_h[v2-48:v2+48,u2-48:u2+48].astype(np.float32)
                list_limg.append(patch_l)
                list_himg.append(patch_h)
    return np.array(list_limg),np.array(list_himg)
def channel_aten_keras(x_i):
    def mean_deriv(x):
        x_m = B.mean(x_i,axis=[1,2],keepdims=True)
        return x_m

    x = L.Lambda(mean_deriv)(x_i)
    x = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x)
    x = L.ReLU()(x)
    x = L.Activation(activation='sigmoid')(x)
    x = L.Multiply()([x,x_i])
    return x
def rca_block_keras(x_i):

    x = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x_i)
    x = L.ReLU()(x)
    x = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x)
    x = channel_aten_keras(x)
    x = L.Add()([x,x_i])
    return x
def rca_res_keras(x_i):
    x = rca_block_keras(x_i)
    x = rca_block_keras(x)
    x = rca_block_keras(x)
    x = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x)
    return L.Add()([x,x_i])
def rca_keras(x_i):
    x_i = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x_i)
    x_r = rca_res_keras(x_i)
    x_r = rca_res_keras(x_r)
    x_r = rca_res_keras(x_r)
    x_f = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x_r)
    x_f = L.Add()([x_f,x_i])
    x_u = L.UpSampling2D(size=(2,2))(x_f)
    x_f = L.Conv2D(filters= 64,kernel_size=(3, 3), strides=1, padding='same')(x_u)
    x_f = L.Conv2D(filters= 3,kernel_size=(3, 3), strides=1, padding='same')(x_f)
    return x_f

input = M.Input((None,None,3))
output = rca_keras(input)
print(output)
x,y = data_loader(route_lrhr)
with B_.tf.device('/gpu:1'):

    rcan = M.Model(inputs=input,outputs = output)
    rcan.compile(optimizer=O.Adam(lr=0.0005,decay = 0.00008), loss='mean_absolute_error', metrics=['accuracy'])
    rcan.summary()
    rcan.fit(x=x,y=y,epochs=50,validation_split=0.05)
    rcan.save(route_save)
