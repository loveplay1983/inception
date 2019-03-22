"""
demo code from https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
"""
###################### Import packages ############################################
import keras

# Layer class definition "keras/engine/base_layer.py"
# "from ..engine.base_layer import Layer" is defined within "keras/layers/core.py"
from keras.layers.core import Layer

import keras.backend as K
import tensorflow as tf

# "cifar10" is defined within "keras/datasets/cifar10.py"
from keras.datasets import cifar10

# Model class definition "keras/engine/training.py"
# "from .engine.training import Model" is defined within "keras folder - models.py file"
from keras.models import Model

# Most of the functions or classes were imported within "keras/layers/__init__.py"
# The definition details were defined within "keras/layers"
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, \
    GlobalAveragePooling2D, AveragePooling2D, Flatten

# opencv for python
import cv2
import numpy as np

# Numpy related utilities   "keras/utils/np_utils.py"
from keras.utils import np_utils

import math

# "keras.optimizers" is defined within "keras" root directory
from keras.optimizers import SGD

# "keras.callbacks" is defined within "keras" root directory
from keras.callbacks import LearningRateScheduler

#######################  Preprocessing before trainig ##############################
num_classes = 10


def load_cifar10_data(img_rows, img_cols):
    """
    Load the cifar10 data and do some preprocessing like resizing...

    img_rows, img_cols -  size of resized image
    """

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, \
                                         img_cols)) for img in X_train[:, :, :, :]])

    X_valid = np.array([cv2.resize(img, (img_rows, \
                                         img_cols)) for img in X_valid[:, :, :, :]])

    # Check the data type of X_train or X_valid
    for each in X_train:
        print(type(each))

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    # Data normalization
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train, Y_train, X_valid, Y_valid


X_train, y_train, X_test, y_test = load_cifar10_data(74, 74)

###################### Define deep learning architecture ###########################
# Auxilliary output
# def aux_output(input_x, output_name):
#
#    input_x = AveragePooling2D((5, 5), strides=3)(input_x)
#    input_x = Conv2D(128, (1, 1), padding='same', activation='relu')(input_x)
#    input_x = Flatten()(input_x)
#    input_x = Dense(1024, activation='relu')(input_x)
#    input_x = Dropout(0.7)(input_x)
#    input_x = Dense(10, activation='softmax', name=output_name)(input_x)
#
#    return input_x
#
# Inception module 
"""
Previous layer ------------------------1x1 convolutions ---|
               ----1x1 convolutions -- 3x3 convolutions ---|--- Filter concat
               ----1x1 convolutions ---5x5 convolutions ---|
               ----3x3 max pooling  ---1x1 convolutions ---|
filters_1x1          -      number of 1x1 filter
filters_3x3_reduce   -      number of 3x3_reduce filter, i.e. the 1x1 filter 
... 
filters_pool_proj    -      number of pooling projection filter, i.e another conv
"""

def inception_cell(x, \
                   filters_1x1, \
                   filters_3x3_reduce, \
                   filters_3x3, \
                   filters_5x5_reduce, \
                   filters_5x5, \
                   filters_pool_proj, \
                   name=None):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', \
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation= \
        'relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', \
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)

    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation= \
        'relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5, (1, 1), padding='same', activation='relu', \
                      kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5_reduce)

    # First make a max-pooling in (3,3) and stride 1 
    pool_proj_3x3 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    # Then do a final conv base on the above max-pooling
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', \
                       kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj_3x3)

    # Final concatenation of inception cell in which it combines all the different filter elements
    '''
    keras/layers/merge.py
    
    class Concatenate(_Merge):
    """Layer that concatenates a list of inputs.

    It takes as input a list of tensors,
    all of the same shape except for the concatenation axis,
    and returns a single tensor, the concatenation of all inputs.

    # Arguments
        axis: Axis along which to concatenate.
        **kwargs: standard layer keyword arguments.
    """

    def concatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `Concatenate` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    return Concatenate(axis=axis, **kwargs)(inputs)

    '''
    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

# Initialize the kernel and bias (kernel is a.k.a weight matrix in "CNN")
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=0.2)

"""
Inception network structure - You can check the whole network structure image "inception-model.png" in the current folder.

The basic structure in text - You can check the text network in the current folder too, named "inception-model-text.png" 

'''

Notice that when viewing the inception cell name, you can find mark like (num+letter, e.g. 3a, 3b, ...), those are the symbols 
of the inception cell location.

num - the location or the index of the current layer
a,b,c - the repetition number of the inception cell 

'''

####################################  Basic structure of the inception layeyr ################################################
conv 7x7/2 -> maxpool 3x3/2 -> conv 3x3/1 -> maxpool 3x3/2 -> inception-cell(3a) -> inception-cell(3b) -> maxpool 3x3/2 -> 

inception(4a) -> inception(4b) -> inception(4c) -> inception(4d) -> inception()4e -> maxpool 3x3/2 -> inception(5a) -> 

inception(5b) -> avgpool 7x7x1 -> dropout(40%) -> linear -> softmax
###############################################################################################################################

Sometimes we can also include the branch output such as pull one of the inception cell to an independent branch conv, flatten, 
dropout and then final dense, i.e. a softmax, then see whether our current network works fine. 

"""
# Before getting into the structure of this inception network, we first make one simple idea clear that is how to seperate layers
# i.e what exactly is a single layer consisted of.
# For "CNN" we often put conv and max pooling layer together as one layer

input_layer = Input(shape=(224, 224, 3))  # "from ..engine import Input"

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', \
           kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)

# Important note and `CNN REVIEW`, max pooling is different from conv layer where it doesn't count the volume for individual max pooling filter,
# rather it use only a 2D filter without volume dim and go through each of the previous corresponding 2D output of the volume,
# finally, max pooling puts all the piece of result to form a new 3D output, in other words, the volume of the new formed output
# is usually the number of the channels of the previous layer.

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)

# The following technic is often used in convolution neural networks in which we first use a 1x1 filter and a 3x3 or ixi(i stands for arbitrary number)
# right after which is also called "bottle neck". The main idea is to reduce the computational cost.
x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)

x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)

x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

# First inception cell of layer 3
x = inception_cell(x, \
                   filters_1x1=64, \
                   filters_3x3_reduce=96, \
                   filters_3x3=128, \
                   filters_5x5_reduce=16, \
                   filters_5x5=32, \
                   filters_pool_proj=32, \
                   name='inception_3a')

# Second inception cell of layer 3
x = inception_cell(x, \
                   filters_1x1=128, \
                   filters_3x3_reduce=128, \
                   filters_3x3=192, \
                   filters_5x5_reduce=32, \
                   filters_5x5=96, \
                   filters_pool_proj=64, \
                   name='inception_3b')

# Pooling for layer 3
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

# First inception cell for layer 4
x = inception_cell(x, \
                   filters_1x1=192, \
                   filters_3x3_reduce=96, \
                   filters_3x3=208, \
                   filters_5x5_reduce=16, \
                   filters_5x5=48, \
                   filters_pool_proj=64, \
                   name='inception_4a')

######################## Auxilliary output - x1  #####################################

# x1 = aux_output(x, "auxilliary_output_1")

x1 = AveragePooling2D((5, 5), strides=3)(x)
x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
x1 = Flatten()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.7)(x1)
x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

# Second inception cell for layer 4
x = inception_cell(x, \
                   filters_1x1=160, \
                   filters_3x3_reduce=112, \
                   filters_3x3=224, \
                   filters_5x5_reduce=24, \
                   filters_5x5=64, \
                   filters_pool_proj=64, \
                   name='inception_4b')

# Thrid inception cell for layer 4
x = inception_cell(x, \
                   filters_1x1=128, \
                   filters_3x3_reduce=128, \
                   filters_3x3=256, \
                   filters_5x5_reduce=24, \
                   filters_5x5=64, \
                   filters_pool_proj=64, \
                   name='inception_4c')

# Fourth inception cell for layer 4
x = inception_cell(x, \
                   filters_1x1=112, \
                   filters_3x3_reduce=144, \
                   filters_3x3=288, \
                   filters_5x5_reduce=32, \
                   filters_5x5=64, \
                   filters_pool_proj=64, \
                   name='inception_4d')

######################## Auxilliary output - x2 #####################################
# x2 = aux_output(x, "auxilliary_output_2")

x2 = AveragePooling2D((5, 5), strides=3)(x)
x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
x2 = Flatten()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.7)(x2)
x2 = Dense(10, activation='softmax', name="auxilliary_output_2")(x2)

# Fifth inception cell for layer 4
x = inception_cell(x, \
                   filters_1x1=256, \
                   filters_3x3_reduce=160, \
                   filters_3x3=320, \
                   filters_5x5_reduce=32, \
                   filters_5x5=128, \
                   filters_pool_proj=128, \
                   name='inception_4e')

# Pooling for layer 4
x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

# First inception cell for layer 5
x = inception_cell(x, \
                   filters_1x1=256, \
                   filters_3x3_reduce=160, \
                   filters_3x3=320, \
                   filters_5x5_reduce=32, \
                   filters_5x5=128, \
                   filters_pool_proj=128, \
                   name='inception_5a')

# Second inception cell for layer 5
x = inception_cell(x, \
                   filters_1x1=384, \
                   filters_3x3_reduce=192, \
                   filters_3x3=384, \
                   filters_5x5_reduce=48, \
                   filters_5x5=128, \
                   filters_pool_proj=128, \
                   name='inception_5b')

# Global pooling for layer 5
x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

# Final steps
# Dropout

x = Dropout(0.4)(x)

# Dense
x = Dense(10, activation='softmax', name='output')(x)

################################  init the model  ############################################
# Model(input, output, name, *args, **kwargs)
model = Model(input_layer, [x, x1, x2], name='inception_v1')

###############################  summary the model ##########################################
model.summary()

############################## run the model ################################################
epochs = 25

# learning rate initialization
initial_lrate = 0.01


def decay(epoch, steps=100):
    initial_lrate = 0.01

    # decay rate
    drop = 0.96

    # decay steps
    epochs_drop = 8

    # decayed_learning_rate = lrate * decay_rate ^ (global_step / decay_steps)
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

"""
class LearningRateScheduler(Callback):
    
    '''
    schedule: a function that takes an epoch index as input and current learning rate
    and returns the new learning rate as output

    verbose: 1 for updating messages and 0 quiet
    '''
"""
lr_sc = LearningRateScheduler(decay, verbose=1)

# categorical_crossentropy - For multi-classification
# loss_weights             - Optional list or dirtionary specifying scalar coefficients
#                            to weight the loss contributions of different model outputs
# metrics                  - List of metrics to be evaluated by the model during training 
#                            and testing, typically you will use metrics=['accuracy']
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], \
              loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

history = model.fit(X_train, [y_train, y_train, y_train], validation_data=(X_test, [y_test, y_test, y_test]), \
                    epochs=epochs, batch_size=256, callbacks=[lr_sc])













