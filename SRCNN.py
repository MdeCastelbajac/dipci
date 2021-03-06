#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import math
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from utils import PSNR, bicubicInterpolation

# SRCNN
# 3 convolutional layers 
# on top of bicubic pre-upsampling

def getModel(upscale_factor=4, channels=1):
    conv_args = {
        "activation": "relu",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(128, 9, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(1, 5, **conv_args)(x)
    return keras.Model(inputs, x)

def compileModel(epochs):
    early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="loss", patience=10
            )

    checkpoint_filepath = "./tmp/checkpoint"

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
    )

    callbacks = [model_checkpoint_callback]
        #early_stopping_callback, model_checkpoint_callback]
    srcnn = getModel()
    srcnn.summary()
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.002)
    srcnn.compile(
        optimizer=optimizer, loss=loss_fn, metrics=['mse', PSNR]
    )   
    return srcnn, callbacks, epochs, checkpoint_filepath

def trainModel( srcnn, batch_size, ssh_lr, ssh_norm, callbacks, epochs, exp):
    
    train_ds = tf.data.Dataset.from_tensor_slices(
            (ssh_lr[0:366], ssh_norm[0:366])
            )

    train_ds = train_ds.batch(batch_size)
    validation_ds = ( 
            ssh_lr[366:],
            ssh_norm[366:]
            )

    history = srcnn.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_ds,
        verbose=2
    )

    plt.plot(history.history['mse'][10:])
    plt.figure()
    plt.plot(history.history['PSNR'])

    srcnn.save('tmp/model/srcnn/'+str(exp))

    return srcnn
