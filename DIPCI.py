#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Add, Input
from tensorflow.keras.models import Model
from tensorflow.nn import depth_to_space
from matplotlib import pyplot as plt
from utils import PSNR


# DIPCI

# This is a direct adaptation of the LapSRN for cross input 
# replacing deconvolution by subpixel convolution to 
# preserve details at the cost of introducing artifacts


filters = 64
D = 3
R = 3

def embedding_model(filters=filters, layer_num=D, name="feature_extraction"):
    '''
    Extracting features inside the shared module
    '''
    img_input = Input(shape=(None, None, filters))
    x = LeakyReLU(alpha=0.2)(img_input)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    for _ in range(layer_num-1):
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters, (3,3), padding='same',
                    use_bias=False)(x)
    model = Model(inputs=img_input, outputs=x, name=name)
    return model


def _upsample_and_condense(x, filters=filters):
    '''
    upsample and condense to residual image 
    inside the shared module
    '''
    # Note that we need to shrink it to 4 filters (2^2 feature maps)
    # to successfully compute sub pixel convolution 
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(32, 3,  padding='same' )(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(16, 3, padding='same' )(x) 
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(4, 3, padding='same' )(x)
    x = LeakyReLU(alpha=0.2)(x)
    upsample = depth_to_space(x, 2)
    x = LeakyReLU(alpha=0.2)(upsample)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    condense = Conv2D(1, (3,3), padding='same',
                        use_bias=False)(x)
    return condense, x

def residual_model(recursive_num=R, filters=filters):
    '''
    shares extracted features with the SSH and upsamples SST (subpixel)
    '''
    embedding = embedding_model()
    img_input = Input(shape=(None, None, filters))
    ##
    x = Input(shape=(None, None, filters))
    x = embedding(img_input)
    x = Add()([x, img_input])
    for _ in range(recursive_num-1):
        x = embedding(x)
        x = Add()([x, img_input])
    res, upsample = _upsample_and_condense(x)
    model = Model(inputs=img_input, outputs=[res, upsample], name="share_and_upsample")
    return model

def upsample_model(filters=filters):
    '''
    upsample SSH
    '''
    img_input = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    upsample = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same',
                        use_bias=False)(x)
    model = Model(inputs=img_input, outputs=upsample, name="upsample")
    return model

def initConv_model(filters=filters):
    '''
    transform the original image into a set of feature maps
    '''
    img_input = Input(shape=(None, None, 1))
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(img_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    x = LeakyReLU(alpha=0.2)(x)
    output = Conv2D(filters, (3,3), padding='same',
                use_bias=False)(x)
    model = Model(inputs=img_input, outputs=output, name="init_feature_extraction")
    return model

def net():
    initConv = initConv_model()
    residual = residual_model()
    upsample = upsample_model()
    
    lr_input = Input(shape=(None,None,1), name="lr_input")
    hr_input = Input(shape=(None,None,1), name="hr_input") 
    
    # x2
    embedded_x = initConv(hr_input)
    upsample_1 = upsample(lr_input)
    residual_1, f_upsample_1 = residual(embedded_x)
    hr1 = Add()([upsample_1, residual_1])

    # x4
    upsample_2 = upsample(hr1)
    residual_2, f_upsample_2 = residual(f_upsample_1)
    hr2 = Add()([upsample_2, residual_2])
    model = Model(inputs=[lr_input, hr_input], outputs=[hr2])
    return model

def compile_model(epochs, lr, exp):
    epochs = epochs
    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="loss", patience=10
        )
    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    checkpoint_filepath = "./tmp/checkpoint_dipci/"+str(exp)
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

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
        verbose=2
    )

    plt.plot(history.history['mse'][10:])
    plt.figure()
    plt.plot(history.history['PSNR'])

    #dipci.save('tmp/model/dipci')
    
    return dipci

