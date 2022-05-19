#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
import tensorflow as tf
from keras.models import Model
from matplotlib import pyplot as plt
import math
from scipy.signal import convolve2d
from PIL import Image

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
    # for numpy arrays
    rMSE = rmse(y_pred, y_true)
    if(rMSE == 0):  
        return 100
    return 20 * math.log10(y_true.max() / rMSE)

def PSNR(y_pred, y_true):
    # tensor safe 
    # keras metric 
    return tf.image.psnr(y_pred, y_true, max_val=1.0)

def bicubicInterpolation(array, img_size):
    img = Image.fromarray(array)
    img = img.resize([img_size, img_size])
    return np.asarray(img)

def plot_line(Images,Titres,cmap,save_name,label="SSH(m)",shrink=0.3,center_colormap=True):
    # Images : a list of images
    # Titres : The titles of the n images in the same order
    # cmap : the colormap to use (advice : "terrain" for SSH, "seismic" for difference)
    # label : the label of the colorbar
    # shrink :a float that shrinks the cbar
    # center_colorbar : a boolean to center the colobar (for differences for exemple.)
    fig,axes=plt.subplots(nrows=1,ncols=len(Images),figsize=(35,15))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    clim=(100,-100)
    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap)
        clim_new=im.properties()["clim"]
        clim=(min(clim[0],clim_new[0]),max(clim[1],clim_new[1]))
    if center_colormap:
        clim=(-max(abs(clim[0]),abs(clim[1])),max(abs(clim[0]),abs(clim[1])))

    for n in range (len(Images)):
        im=axes[0].imshow(Images[n],cmap=cmap,clim=clim)
        
    for n in range (len(Images)):
 
        axes[n].imshow(Images[n],clim=clim,cmap=cmap)
        axes[n].set_title(Titres[n],fontsize=20)
    col=fig.colorbar(im,ax= axes[:], location ="right",shrink=shrink)
    col.ax.tick_params(labelsize=20)
    col.set_label(label=label,size=20)
    plt.savefig(save_name+".pdf",bbox_inches="tight")
    return


def bicubicPredict( lr_data, data ):
    length = lr_data.shape[0]
    bicubic_upsampled_images = np.array(
            [bicubicInterpolation(img, data[0].shape[0]) for img in lr_data]
            )
    bicubic_denormalized = np.array(
            [normalize(bicubic_upsampled_images[i], data[i].min(), data[i].max()) for i in range(length)]
            )
    return bicubic_denormalized


def bicubicMetrics( lr_data, data ):
    length = lr_data.shape[0]
    bicubic_predictions = bicubicPredict(lr_data, data)
    bicubic_psnr = np.array([ psnr(bicubic_predictions[i], data[i]) for i in range(length) ])
    bicubic_rmse = np.array([ rmse(bicubic_predictions[i], data[i]) for i in range(length) ])
    return bicubic_predictions, bicubic_rmse, bicubic_psnr


def srcnnPredict(model, lr_data, data):
    length = lr_data.shape[0]
    srcnn_predictions = np.array( [model.predict(np.expand_dims(img, axis=0), verbose=0) for img in lr_data] )
    srcnn_predictions = np.array( [srcnn_predictions[i].reshape((data[0].shape[0], data[0].shape[1])) for i in range(length)] )
    srcnn_predictions = np.array([ normalize(srcnn_predictions[i], data[i].min() , data[i].max()) for i in range(length) ])
    return srcnn_predictions


def srcnnMetrics( model, lr_data, data ):
    length = lr_data.shape[0]
    srcnn_predictions = srcnnPredict( model, lr_data, data )
    srcnn_psnr = np.array([ psnr(srcnn_predictions[i], data[i]) for i in range(length) ])
    srcnn_rmse = np.array([ rmse(srcnn_predictions[i], data[i]) for i in range(length) ])
    return srcnn_predictions, srcnn_rmse, srcnn_psnr
    

def dipciPredict( model, ssh_lr, sst, ssh ):
    length = ssh_lr.shape[0]
    input_tuple = []
    for i in range( ssh_lr.shape[0] ):
        input_tuple.append(( ssh_lr[i] , sst[i] ))
    dipci_predictions = np.array( [model.predict( [np.expand_dims(img[0], axis=0),np.expand_dims(img[1], axis=0)] ) for img in input_tuple] )
    dipci_predictions = np.array( [dipci_predictions[i].reshape(ssh[0].shape[0], ssh[0].shape[1]) for i in range(length)] )
    dipci_predictions = np.array( [ normalize(dipci_pred[i], ssh[i].min(), ssh[i].max()) for i in range(length) ] )
    return dipci_predictions
    

def dipciMetrics( model, ssh_lr, sst, ssh ):
    length = ssh_lr.shape[0]
    dipci_predictions = dipciPredict( model, ssh_lr, sst, ssh )
    dipci_psnr = np.array([ psnr(dipci_predictions[i], ssh[i]) for i in range(length) ])
    dipci_rmse = np.array([ rmse(dipci_predictions[i], ssh[i]) for i in range(length) ])
    return dipci_predictions, dipci_rmse, dipci_psnr

