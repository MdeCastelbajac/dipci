#!/usr/bin/env python
# coding: utf-8

# In[15]:


from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Input
import numpy as np
import math


# In[16]:


def srcnn():
    SRCNN = Sequential()
    SRCNN.add(Conv2D(128, 9, activation='relu', input_shape=(32, 32, 1)))
    SRCNN.add(Conv2D(64, 3, activation='relu'))
    SRCNN.add(Conv2D(1, 5, activation='linear',))
    return SRCNN

