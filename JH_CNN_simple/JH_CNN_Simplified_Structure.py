"""
CNN SIMPLIFIED STRUCTURE. ALEXNET.

2020 02 21 JJH.

REFERED FROM
https://bskyvision.com/421
https://keras.io/layers/pooling/
https://datascienceschool.net/view-notebook/d19e803640094f76b93f11b850b920a4/  -> INVALID.

CAUTION : ALEXNET STRUCTURE CURRENTLY RUN SINGLE EPOCH, IMPERFECT. DO NOT RUN IF POSSIBLE.
"""
import os
import sys

import simplejson

sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers

print('CURRENT WORKING DIRECTORY : ', os.getcwd())

"""
QUASI ALEXNET

INITIALIZING CONVOLUTIONAL NEURAL NETWORK

SAVE MODEL WITH SIMPLE JSON, WEIGHTS AS HDF5(STANDARD)
"""

# load data.

LDB_array = np.load(file='LDB_half_array_15.npy')
LDB_ohe = np.load(file='LDB_half_ohe_15.npy')  # THESE ARE LOCAL FILES
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
print(f'TYPE OF CNN_model = {type(CNN_model)}')
# CNN_model_loaded = load_model('variables/variables.data-00000-of-00001')
"""
Structure(Oversimplified)

(Conv2D -> MaxPool2D)*5 -> Flatten -> Dense(4096)*2 -> Softmax.
"""

#####################################################
# initial layer (including input layer)
CNN_model.add(
    Conv2D(filters=48, kernel_size=(4, 4), strides=1, activation='relu', input_shape=(64, 64, 3)))  # << input layer.

# print(f'1st filter shape={CNN_model.output_shape}\n')  # None, 61, 61, 48
# Apply MaxPooling
# CNN_model.add(local_response_normalization())   # as tensorflow.nn.lrn supports tensor type only, we can't put CNN_model, whose type is Sequential
CNN_model.add(BatchNormalization())  # hence, we're going to apply batch normalization at all steps.
CNN_model.add(
    MaxPooling2D(pool_size=(3, 3), strides=2))  # in case of input_shape=(128, 128, 3), pool_size=(3,3) -> (4, 4)
# print(f'2nd filter shape={CNN_model.output_shape}\n')  # None, 30, 30, 48
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=128, kernel_size=(2, 2), strides=1, activation='relu'))
# print(f'3rd filter shape = {CNN_model.output_shape}')  # None, 29, 29, 128
CNN_model.add(BatchNormalization())
CNN_model.add(MaxPooling2D(pool_size=(3, 3), strides=2))  # in case of 128, pool_size=(4, 4)
# print(f'4th filter shape = {CNN_model.output_shape}\n')  # None, 14, 14, 128
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
# print(f'5th filter shape = {CNN_model.output_shape}')  # None, 14, 14, 192
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
# print(f'6th filter shape = {CNN_model.output_shape}')
#####################################################


#####################################################
CNN_model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
CNN_model.add(BatchNormalization())
# print(f'7th filter shape = {CNN_model.output_shape}')
#####################################################


# Fully Connected Layers

CNN_model.add(Flatten())

#####################################################
CNN_model.add(Dense(2048, activation='relu'))
CNN_model.add(Dropout(0.5))
#####################################################


#####################################################
CNN_model.add(Dense(2048, activation='relu'))
CNN_model.add(Dropout(0.5))
#####################################################


#####################################################
CNN_model.add(Dense(5, activation='softmax'))
#####################################################

CNN_model.summary()  # show summary
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 61, 61, 48)        2352      
_________________________________________________________________
batch_normalization (BatchNo (None, 61, 61, 48)        192       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 30, 30, 48)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 128)       24704     
_________________________________________________________________
batch_normalization_1 (Batch (None, 29, 29, 128)       512       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 14, 192)       221376    
_________________________________________________________________
batch_normalization_2 (Batch (None, 14, 14, 192)       768       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 14, 14, 192)       331968    
_________________________________________________________________
batch_normalization_3 (Batch (None, 14, 14, 192)       768       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 128)       221312    
_________________________________________________________________
batch_normalization_4 (Batch (None, 14, 14, 128)       512       
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
Total params: 56,403,578
Trainable params: 56,402,202
Non-trainable params: 1,376
_________________________________________________________________
'''
#####################################################

early_stop = EarlyStopping(monitor='accuracy', verbose=1, patience=5)

model_check_point = ModelCheckpoint(filepath='variables/quasiAlexNet_2020_02_21_half15_a.h5', monitor='accuracy', verbose=1,
                                    save_best_only=True)

CNN_optimizer = optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999,
                                epsilon=1e-7)  # we are going to use Adam for optimizing.

CNN_model.compile(loss='binary_crossentropy', optimizer=CNN_optimizer, metrics=['accuracy'])

#####################################################

# RUN CONVOLUTIONAL NEURAL NETWORK -> Won't it take too long time??

fitting = CNN_model.fit(x=X_tr, y=Y_tr, batch_size=100, epochs=20, verbose=1, callbacks=[early_stop, model_check_point],
                        validation_data=(X_ts, Y_ts), workers=0, use_multiprocessing=True)

evaluation = CNN_model.evaluate(X_ts, Y_ts)
print(f'Test loss: {evaluation[0]}, accuracy: {evaluation[1]}')

# serialize model to JSON
model_json = CNN_model.to_json()
with open('quasiAlexNet_2020_02_21_half15m.json', "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

# serialize weights to HDF5
CNN_model.save_weights('quasiAlexNet_2020_02_21_half15w.h5')
print("Saved model to disk")


# CNN_model.save('variables/quasiAlexNet_2020_02_21_half15_a.h5')
#####################################################

# Draw graphs about train and test loss/accuracy. refer from mnist_cnn.py.
# LOSS GRAPH

grp_train_loss = fitting.history['loss']
grp_test_loss = fitting.history['val_loss']

x = range(len(grp_train_loss))
plt.plot(x, grp_train_loss, marker='.', color='red', label='Train loss')
plt.plot(x, grp_test_loss, marker='.', color='blue', label='Test loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#####################################################
# ACCURACY GRAPH

grp_train_acc = fitting.history['accuracy']
grp_test_acc = fitting.history['val_accuracy']

plt.plot(x, grp_train_acc, marker='.', c='red', label='Train Acc.')
plt.plot(x, grp_test_acc, marker='.', c='blue', label='Test Acc.')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

"""
As a result, losses are converged to 2.4 in any case. so is accuracy(about 84.6%).
Let us run our CNN with converted images.
"""
