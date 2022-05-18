#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space
from matplotlib import pyplot as plt
from utils import PSNR

filters = 32
upscale_factor = 2

def simple_conv_model( filters=filters*2 ):
    ''' extracts SST features '''
    input_img = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same')(input_img)
    x = LeakyReLU(alpha=0.2)(x)        
    for i in range( 21 ):
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

def upsample_model( filters=filters*2 ):
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

    # secondPass
    x = shuffle(x)
    x = simple_conv(x)
    y = upsample(hr)
    hr = Add()([y, x])

    return Model(inputs=[lr_input, hr_input], outputs=hr)

def compile():
    epochs = 100
    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    checkpoint_filepath = "./tmp/checkpoint_dipci"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [model_checkpoint_callback]

    dipci = net()
    dipci.compile(optimizer=optimizer, loss=loss, metrics=['mse', PSNR])
    dipci.summary()

    return dipci, callbacks, epochs, checkpoint_filepath

def train( dipci, ssh_lr, sst_lr, ssh_norm, callbacks, epochs ):
    history = dipci.fit(
        x = {
            "lr_input": ssh_lr[0:366], 
            "hr_input": sst_lr[0:366]
            },
        y=ssh_norm[0:366], 
        batch_size=16,
        epochs=epochs, 
        callbacks=callbacks, 
        validation_data= {
            "lr_input": ssh_lr[366:0], 
            "hr_input": sst_lr[366:0] 
            },
        verbose=1 
    )

    plt.plot(history.history['mse'][10:])
    plt.figure()
    plt.plot(history.history['PSNR'])

    dipci.save('tmp/model/dipci')
    
    return dipci


