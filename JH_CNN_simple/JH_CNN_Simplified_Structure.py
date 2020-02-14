"""
CNN SIMPLIFIED STRUCTURE. ALEXNET.

2020 02 14 JJH.

REFERED FROM
https://bskyvision.com/421
https://keras.io/layers/pooling/
https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/

CAUTION : ALEXNET STRUCTURE CURRENTLY RUN SINGLE EPOCH, IMPERFECT. DO NOT RUN IF POSSIBLE.

"""

import sys, os

sys.path.append(os.pardir)
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer

print('CURRENT WORKING DIRECTORY : ', os.getcwd())
print(tf.__version__)


class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=1e-4, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1), padding="same", pool_mode='avg')
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def compute_output_shape(self, input_shape):
        return input_shape

    # load data.


LDB_array = np.load(file='LDB_array.npy')
LDB_ohe = np.load(file='LDB_ohe.npy')
# I'm not sure if array and ohe have same file order. IF the result is poor, out suspect will be clear

# for x in enumerate(LDB_ohe): print(x)
print(np.max(LDB_array))  # 1.0 but file size is still 141MB.

print(LDB_array.shape)  # (3029, 64, 64, 3)
print(type(LDB_array))  # numpy.ndarray.
print(len(LDB_array))  # 3029

print(LDB_ohe.shape)  # (3029, 10)
print(type(LDB_ohe))  # numpy.ndarray.
print(len(LDB_ohe))  # 3029

np.random.seed(20200214)

X_tr, X_ts, Y_tr, Y_ts = train_test_split(LDB_array, LDB_ohe, test_size=0.2)

print(f'X_tr.shape = {X_tr.shape}')  # (2423, 64, 64, 3)
print(f'Y_tr.shape = {Y_tr.shape}')  # (2423, 10)
print(f'X_ts.shape = {X_ts.shape}')  # (606, 64, 64, 3)
print(f'Y_ts.shape = {Y_ts.shape}')  # (606, 10)

# CONSTRUCT CNN STRUCTURE

CNN_model = Sequential()

"""
Structure(Oversimplified)

(Conv2D -> MaxPool2D)*5 -> Flatten -> Dense(4096)*2 -> Softmax.
"""

#####################################################
# initial layer (including input layer)
CNN_model.add(
    Conv2D(filters=48, kernel_size=(4, 4), strides=1, activation='relu', input_shape=(64, 64, 3)))  # << input layer.

print(f'1st filter shape={CNN_model.output_shape}\n')  # None, 61, 61, 48
# Apply Local Response Normalization (LRN).
CNN_model.add(LocalResponseNormalization(input_shape=CNN_model.output_shape[1:]))
# Apply MaxPooling
CNN_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
print(f'2nd filter shape={CNN_model.output_shape}\n')  # None, 30, 30, 48
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=1, activation='relu'))
print(f'3rd filter shape = {CNN_model.output_shape}')  # None, 29, 29, 128
CNN_model.add(LocalResponseNormalization(input_shape=CNN_model.output_shape[1:]))
CNN_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
print(f'4th filter shape = {CNN_model.output_shape}\n')  # None, 14, 14, 128
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
print(f'5th filter shape = {CNN_model.output_shape}')  # None, 14, 14, 192
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
print(f'6th filter shape = {CNN_model.output_shape}')
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
print(f'7th filter shape = {CNN_model.output_shape}')
#####################################################


# Fully Connected Layers

CNN_model.add(Flatten())

#####################################################
CNN_model.add(Dense(2048, activation='relu'))
CNN_model.add(Dropout(0.375))
#####################################################


#####################################################
CNN_model.add(Dense(2048, activation='relu'))
CNN_model.add(Dropout(0.375))
#####################################################


#####################################################
CNN_model.add(Dense(10, activation='softmax'))
#####################################################

CNN_model.summary()  # show summary
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 61, 61, 48)        2352      
_________________________________________________________________
local_response_normalization (None, 61, 61, 48)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 30, 30, 48)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 128)       24704     
_________________________________________________________________
local_response_normalization (None, 29, 29, 128)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 14, 192)       221376    
_________________________________________________________________
batch_normalization (BatchNo (None, 14, 14, 192)       768       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 192)       331968    
_________________________________________________________________
batch_normalization_1 (Batch (None, 14, 14, 192)       768       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 128)       221312    
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 128)       512       
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0         
_________________________________________________________________
dense (Dense)                (None, 2048)              51382272  
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dropout_1 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                20490     
=================================================================
Total params: 56,402,874
Trainable params: 56,401,850
Non-trainable params: 1,024
_________________________________________________________________
'''
#####################################################

early_stop = EarlyStopping(monitor='accuracy', verbose=1, patience=10)

CNN_optimizer = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-7)  # we are going to use Adam for optimizing.

CNN_model.compile(loss='binary_crossentropy', optimizer=CNN_optimizer, metrics=['accuracy'])

#####################################################

# RUN CONVOLUTIONAL NEURAL NETWORK -> Won't it take too long time??

fitting = CNN_model.fit(x=X_tr, y=Y_tr, batch_size=50, epochs=20, verbose=1, callbacks=[early_stop],
                        validation_data=(X_ts, Y_ts))

evaluation = CNN_model.evaluate(X_ts, Y_ts)
print(f'Test loss: {evaluation[0]}, accuracy: {evaluation[1]}')
