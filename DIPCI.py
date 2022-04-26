#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space
from matplotlib import pyplot as plt


def simple_conv_model( shape, filters):
    ''' extracts SST features '''
    input_img = Input(shape=(None, None, shape))
    x = Conv2D(filters, (3,3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)        
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    return Model(inputs=input_img, outputs=x)


def shuffle_model( filters ):
    ''' upsamples in feature branch '''
    input_img = Input(shape=(None, None, filters))
    x = Conv2D(filters, (3,3), padding='same')(input_img)
    x = depth_to_space(x, upscale_factor)        
    x = Conv2D(filters, (2,2), padding='same')(x)
    x = LeakyReLU()(x)
    return Model(inputs=input_img, outputs=x)

def upsample_model( shape, filters):
    ''' upsamples SSH '''
    input_img = Input(shape=(None, None, shape))
    upsample = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same')(input_img)
    return Model(inputs=input_img, outputs=upsample)

def net(filters):
    simple_conv_1 = simple_conv_model(1, filters)
    simple_conv_2 = simple_conv_model(filters,filters)
    shuffle = shuffle_model(filters)
    upsample_1 = upsample_model(1,filters)
    upsample_2 = upsample_model(filters,filters)
    
    input_hr = Input(shape=(None, None, 1))
    input_lr = Input(shape=(None, None, 1))
    
    # firstPass
    x = simple_conv_1(input_hr)
    y = upsample_1(input_lr)
    hr = Add()([y, x])
    # secondPass
    x = shuffle(x)
    x = simple_conv_2(x)
    y = upsample_2(hr)
    hr = Add()([y, x])
    
    return Model(inputs=[input_lr, input_hr], outputs=hr)
