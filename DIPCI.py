#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space
from matplotlib import pyplot as plt

filters = 32
upscale_factor = 2

def simple_conv_model( filters=filters ):
    ''' extracts SST features '''
    input_img = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)        
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(1, (3,3), padding='same')(x)
    return Model(inputs=input_img, outputs=x)

def shuffle_model( filters=filters ):
    ''' upsamples in feature branch '''
    input_img = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same')(input_img)
    x = depth_to_space(x, upscale_factor)        
    x = Conv2D(filters, (2,2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(1, (3,3), padding='same')(x)
    return Model(inputs=input_img, outputs=x)

def upsample_model( filters=filters ):
    ''' upsamples SSH '''
    input_img = Input(shape=(None, None, 1))
    x = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same')(input_img)
    x = Conv2D(1, (3,3), padding='same')(x)
    return Model(inputs=input_img, outputs=x)

def net():
    simple_conv = simple_conv_model()
    shuffle = shuffle_model()
    upsample = upsample_model()
    
    lr_input = Input(shape=(None,None,1), name="lr_input")
    hr_input = Input(shape=(None,None,1), name="hr_input") 

    # firstPass
    x = simple_conv(hr_input)
    y = upsample(lr_input)
    hr = Add()([y, x])

    return Model(inputs=[lr_input, hr_input], outputs=hr)