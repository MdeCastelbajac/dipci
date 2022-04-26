#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.signal import convolve2d

def downsample(image, factor):
    kernel = np.ones((factor, factor))
    convolved = convolve2d(image, kernel, mode='valid')
    return convolved[::factor, ::factor] / factor

def normalize(image):
    factor = 1 / (np.amax(image) - np.amin(image))
    result = (image - np.amin(image)) * factor
    return result


