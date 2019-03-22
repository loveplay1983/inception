"""
demo code from https://www.analyticsvidhya.com/blog/2018/10/understanding-inception-network-from-scratch/
"""
###################### Import packages ############################################
import keras

# Layer class definition "keras/engine/base_layer.py"
# "from ..engine.base_layer import Layer" is defined within "keras/layers/core.py"
from keras.layers.core import Layer

import keras.backend as k
import tensorflow as tf

# "cifar10" is defined within "keras/datasets/cifar10.py"
from keras.datasets import cifar10

# Model class definition "keras/engine/training.py"
# "from .engine.training import Model" is defined within "keras folder - models.py file"
from keras.models import Model

# Most of the functions or classes were imported within "keras/layers/__init__.py"
# The definition details were defined within "keras/layers"
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate,\
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
    
X_train, y_train, X_test, y_test = load_cifar10_data(112, 112)

###################### Define deep learning architecture ###########################
def inception_cell(x,
                     filters_1x1,
                     filters_3x3_reduce,   # 1x1 conv
                     filters_3x3, 
                     filters_5x5_reduce,   # 1x1 conv
                     filters_5x5,
                     filters_pool_proj,    # max pooling
                     name=None):
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

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',\
            kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation=\
            'relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', \
            kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)

    
    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation=\
            'relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

    conv_5x5 = Conv2D(filters_5x5, (1, 1), padding='same', activation='relu', \
            kernel_initializer=kernel_init, bias_initalizer=bias_init)(conv_5x5_reduce)

    # First make a max-pooling in (3,3) and stride 1 
    pool_proj_3x3 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    # Then do a final conv base on the above max-pooling
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relue', \
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
    

# Initialize the kernel and bias (kernel is a.k.a weight matrix in cnn)
kernel_init = keras.initializers.glorot_uniform()
bias_init = keras.initializers.Constant(value=.2)

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

input_layer = Input(shape=(224, 224, 3))            # "from ..engine import Input"

x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', \
        kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)

# Important note and `CNN REVIEW`, max pooling is differently from conv layer where it doesn't count the volume for individual max pooling filter,
# rather it use only a 2D filter without volume dim and go through each of the previous corresponding 2D output of the volume,
# finally, max pooling puts all the piece of result to form a new 3D output, in other words, the volume of the new formed output
# is usually the number of the channels of the previous layer.

x = MaxPool2D((3,3))












