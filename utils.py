#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import PIL
from matplotlib import pyplot as plt

def downsample(tensor, factor, size):
    return image.resize(tensor, [size//factor, size//factor], method='bicubic') 

def normalize(image, maximum):
    factor = maximum / (np.amax(image) - np.amin(image))
    result = (image - np.amin(image)) * factor 
    return result

def processInput(tensor):
    tensor = tensor / 255.0
    last_dimension_axis = len(tensor.shape) - 1
    y, u, v = tf.split(tensor, 3, axis=last_dimension_axis)
    return y


def predict(model, img):
    img = img[:,:,0]
    y = img_to_array(img)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)
    
    outRes = np.zeros([out[0].shape[0], out[0].shape[1], 3])
    for i in range(out[0].shape[0]):
        for j in range(out[0].shape[1]):
            for k in range(3):
                outRes [i,j,k] = out[0][i,j]
    return outRes


def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, 1, name=None)

    
def test_srcnn(model, test_img_paths):
    upscale_factor=2
    img_size=144
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0
    for index, test_img_path in enumerate(test_img_paths[100:200]):
        img = load_img(test_img_path)
        lowres_input = downsample(img, upscale_factor, img_size)
        lowres_img_arr = normalize(img_to_array(lowres_input), 1)
        
        # bicubic interpolation 
        highres_img = image.resize(lowres_input, [img_size, img_size], method='bicubic')

        # model prediction
        prediction = predict(model, lowres_input)

        highres_img_arr = normalize(img_to_array(highres_img),1)
        predict_img_arr = normalize(img_to_array(prediction),1)
        ground_truth = normalize(img_to_array(img),1)
        
        bicubic_psnr = tf.image.psnr(highres_img_arr, ground_truth, max_val=1)
        test_psnr = tf.image.psnr(predict_img_arr, ground_truth, max_val=1)
        
        total_bicubic_psnr += bicubic_psnr
        total_test_psnr += test_psnr
        if index%10 == 0:
            plt.figure()
            f, (ax2, ax3, ax4) = plt.subplots(1, 3, sharey=True, figsize=(15,10))
            ax2.imshow(highres_img_arr)
            ax2.set_title('bicubic')
            ax3.imshow(predict_img_arr)
            ax3.set_title('srcnn')
            ax4.imshow(ground_truth)
            ax4.set_title('ground truth')
            plt.show()
            print(
            "PSNR of Bicubic and Ground Truth image is %.4f" % bicubic_psnr   
             )
            print("PSNR of Prediction and Ground Truth is %.4f" % test_psnr)
           

    print("Avg. PSNR for BICUBIC is %.4f" % (total_bicubic_psnr / 100))
    print("Avg. PSNR for PREDICTION is %.4f" % (total_test_psnr / 100))