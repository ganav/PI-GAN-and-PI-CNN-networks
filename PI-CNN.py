#!/usr/bin/env python
#title           :Network.py
#description     :Architecture file(Generator and Discriminator)
#author          :Deepak Birla
#date            :2018/10/30
#usage           :from Network import Generator, Discriminator
#python_version  :3.5.4 

# Modules
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D,Dense,concatenate,Input,add,Reshape,LeakyReLU, PReLU,UpSampling2D,Conv2D, Conv2DTranspose,MaxPooling2D, Activation,Flatten,Lambda
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer

def res_block(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model) 
    model = add([gen, model])
    return model

def model(shape,n_class):
    
    inp = Input(shape = shape)

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "valid")(inp)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(4):
        model = res_block(model, 3, 64, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(4):
        model = res_block(model, 3, 128, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(4):
        model = res_block(model, 3, 128, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)
    for index in range(4):
        model = res_block(model, 3, 128, 1)

    model = Flatten()(model)
    model = Dense(n_class,activation = "softmax")(model)
    generator_model = Model(inputs = inp, outputs = model)

    return model


