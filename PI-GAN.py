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

# Residual block
def res_block_gen(model, kernal_size, filters, strides):
    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
        
    model = add([gen, model])
    
    return model
    
    
def up_sampling_block(model, kernal_size, filters, strides):

    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def discriminator_block(model, filters, kernel_size, strides):
    
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model


def encod(model, filters, kernel_size, strides):

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = MaxPooling2D((2,2), strides=(2,2))(model)

    return model

def decod(model, filters, kernel_size, strides):

    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = UpSampling2D(size = 2)(model)

    return model

def encoder(model):
    model=encod(model, 128, 3, 1)
    model=encod(model, 128, 3, 1)
    model=encod(model, 128, 3, 1)
    model=encod(model, 128, 3, 1)
    return model

def decoder(model):
    model=decod(model, 128, 3, 1)
    model=decod(model, 128, 3, 1)
    model=decod(model, 128, 3, 1)
    model=decod(model, 128, 3, 1)
    
    return model

def generator(shape):
    inp = Input(shape = shape)
    inp2 = Input(shape = shape)

    model1 = encoder(inp)
    model2 = encoder(inp2)

    for index in range(2):
        model1 = res_block_gen(model1, 3, 128, 1)
        model2 = res_block_gen(model2, 3, 128, 1)

    model = concatenate([model1,model2],axis=-1)

    for index in range(3):
        model = res_block_gen(model, 3, 256, 1)

    model = decoder(model)

    model = Conv2D(filters = 3, kernel_size = 3, strides = 1, padding = "same")(model)
    model = Activation('tanh')(model)
    
    generator_model = Model(inputs = [inp,inp2], outputs = model)
    return generator_model

    
def discriminator(image_shape):
        
    inp = Input(shape = image_shape)
  
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(inp)
    model = LeakyReLU(alpha = 0.2)(model)

    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 128, 3, 1)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 256, 3, 2)
    model = discriminator_block(model, 256, 3, 2)
    
    model = Flatten()(model)
    model = LeakyReLU(alpha = 0.2)(model)
       
    model = Dense(1)(model)
    model = Activation('sigmoid')(model) 
        
    discriminator_model = Model(inputs = inp, outputs = model)
        
    return discriminator_model
