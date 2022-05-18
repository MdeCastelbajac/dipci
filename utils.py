#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from keras.models import Model
from tensorflow.nn import depth_to_space
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import math
from scipy.signal import convolve2d
from keras import backend as K

def normalize(image, min, max):
    factor = (max - min) / (np.amax(image) - np.amin(image))
    result = (image - np.amin(image)) * factor + min
    return result

def downsample(image, factor):
    kernel = np.ones((factor, factor))
    convolved = convolve2d(image, kernel, mode='valid')
    return convolved[::factor, ::factor] / factor 

def mse(y_pred, y_true):
    return np.square(np.subtract(y_true,y_pred)).mean() 

def rmse(y_pred, y_true):
    return math.sqrt(mse(y_pred, y_true))

def psnr(y_pred, y_true):
    rMSE = rmse(y_pred, y_true)
    if(rMSE == 0):  
        return 100
    return 20 * math.log10(y_true.max() / rMSE)

def PSNR(y_pred, y_true):
     return tf.image.psnr(y_pred, y_true, max_val=1.0)

def bicubic_interpolation(array, img_size):
    img = Image.fromarray(array)
    img = img.resize([img_size, img_size])
    return np.asarray(img)

def test_srcnn(model, lr_data, data):
    bicubic_predictions = np.array( [bicubic_interpolation(img, data[0].shape[0]) for img in lr_data] )
    srcnn_predictions = np.array( [model.predict(np.expand_dims(img, axis=0)) for img in lr_data] )
    # denormalize data
    bicubic_predictions = np.array( [ normalize(bicubic_predictions[i], data[i].min() , data[i].max()) for i in range(bicubic_predictions.shape[0]) ] )
    srcnn_predictions = np.array( [ normalize(srcnn_predictions[i].reshape((data[0].shape[0], data[0].shape[1])), data[i].min() , data[i].max()) for i in range(srcnn_predictions.shape[0]) ] )

    # metrics 
    bicubic_psnr = np.array([ psnr(bicubic_predictions[i], data[i]) for i in range(bicubic_predictions.shape[0]) ])
    bicubic_rmse = np.array([ rmse(bicubic_predictions[i], data[i]) for i in range(bicubic_predictions.shape[0]) ])
    srcnn_psnr = np.array([ psnr(srcnn_predictions[i], data[i]) for i in range(srcnn_predictions.shape[0]) ])
    srcnn_rmse = np.array([ rmse(srcnn_predictions[i], data[i]) for i in range(srcnn_predictions.shape[0]) ])
    
    # plot results
    print('Average bicubic PSNR : ', bicubic_psnr.mean())
    print('Average bicubic RMSE : ', bicubic_rmse.mean())
    print('Average SRCNN PSNR : ', srcnn_psnr.mean())
    print('Average SRCNN RMSE : ',srcnn_rmse.mean())
    
    for i in range(0,100,20):
        plt.figure()
        f, (ax2, ax3, ax4) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
        ax2.imshow(bicubic_predictions[i])
        ax2.set_title('bicubic')
        ax3.imshow(srcnn_predictions[i])
        ax3.set_title('SRCNN')
        ax4.imshow(data[i])
        ax4.set_title('Ground Truth')
        plt.show()
        print("PSNR of Bicubic and Ground Truth image is ", bicubic_psnr[i])
        print("PSNR of SRCNN and Ground Truth is ", srcnn_psnr[i])


def test_dipci(model, ssh_lr, sst_norm, ssh):
    bicubic_pred = np.array( [bicubic_interpolation(img, ssh[0].shape[0]) for img in ssh_lr] )
    
    # model prediction
    input_tuple = []
    for i in range( ssh_lr.shape[0] ):
        input_tuple.append(( ssh_lr[i] , sst_norm[i] ))

    dipci_pred = np.array( [model.predict( [np.expand_dims(img[0], axis=0),np.expand_dims(img[1], axis=0)] ) for img in input_tuple] )

    # denormalize data
    bicubic_pred = np.array( [ normalize(bicubic_pred[i], ssh[i].min(), ssh[i].max()) for i in range(bicubic_pred.shape[0]) ] )
    dipci_pred = np.array( [ normalize(dipci_pred[i].reshape(ssh[0].shape[0], ssh[0].shape[1]), ssh[i].min(), ssh[i].max()) for i in range(dipci_pred.shape[0]) ] )
    
    # metrics
    bicubic_psnr = np.array([ psnr(bicubic_pred[i], ssh[i]) for i in range(bicubic_pred.shape[0]) ])
    bicubic_rmse = np.array([ rmse(bicubic_pred[i], ssh[i]) for i in range(bicubic_pred.shape[0]) ])
    dipci_psnr = np.array([ psnr(dipci_pred[i], ssh[i]) for i in range(dipci_pred.shape[0]) ])
    dipci_rmse = np.array([ rmse(dipci_pred[i], ssh[i]) for i in range(dipci_pred.shape[0]) ])
               
    # plot results
    print('Average bicubic PSNR : ', bicubic_psnr.mean())
    print('Average bicubic RMSE : ', bicubic_rmse.mean())
    print('Average DIPCI PSNR : ', dipci_psnr.mean())
    print('Average DIPCI RMSE : ', dipci_rmse.mean())
                            
    for i in range(0,100,20):
        plt.figure()
        f, (ax2, ax3, ax4) = plt.subplots(1, 3, sharey=True, figsize=(10,5))
        ax2.imshow(bicubic_pred[i])
        ax2.set_title('bicubic')
        ax3.imshow(dipci_pred[i])
        ax3.set_title('ours')
        ax4.imshow(ssh[i])
        ax4.set_title('Ground Truth')
        plt.show()
        print("PSNR of Bicubic and Ground Truth image is ", bicubic_psnr[i])
        print("PSNR of DIPCI and Ground Truth is ", dipci_psnr[i])
