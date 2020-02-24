# -*- coding: utf-8 -*-
"""final_practice12_adding_more_dropout.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B1lqSjLHJXUrrKQ52eQWUFqdTyZfnnCu
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf

print(tf.__version__)
print(tf.test.gpu_device_name())

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import plot_model

#1 64x64 이미지 5000장씩 저장한 numpy 데이터를 불러와서 shape 확인
X_train, X_test, y_train, y_test = np.load("./drive/My Drive/final_data/practice7_data.npy", allow_pickle=True)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

#2 정규화
X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

#3 모델 생성
model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

#4 모델 확인
model.summary()
plot_model(model)

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy']) # vgg16에선 'adam'보다 'sgd'가 적합

#5 모델 학습
model_path = "./drive/My Drive/final_data/practice12_sgd.model"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
# early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_sgd = model.fit(X_train, y_train, batch_size=200, epochs=150, callbacks=[checkpoint], validation_split=0.2)

from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# best 모델 불러와서 예측 후 평가
best_model = load_model("./drive/My Drive/final_data/practice12_sgd.model")

eval = best_model.evaluate(X_test, y_test)
print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

predictions = best_model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
print(f'Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print(f'Classification Report')
print(classification_report(y_test, y_pred, target_names=['df', 'mel', 'nv', 'tsu', 'vl']))

X_train, X_test, y_train, y_test = np.load("./drive/My Drive/final_data/practice7_data.npy", allow_pickle=True)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9), metrics=['accuracy'])

#6 모델 학습2
model_path = "./drive/My Drive/final_data/practice12_sgd_momentum.model"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
# early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_sgd_momentum = model.fit(X_train, y_train, batch_size=200, epochs=150, callbacks=[checkpoint], validation_split=0.2)

# best 모델 불러와서 예측 후 평가
best_model = load_model("./drive/My Drive/final_data/practice12_sgd_momentum.model")

eval = best_model.evaluate(X_test, y_test)
print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

predictions = best_model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
print(f'Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print(f'Classification Report')
print(classification_report(y_test, y_pred, target_names=['df', 'mel', 'nv', 'tsu', 'vl']))

X_train, X_test, y_train, y_test = np.load("./drive/My Drive/final_data/practice7_data.npy", allow_pickle=True)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, nesterov=True), metrics=['accuracy'])

#7 모델 학습3
model_path = "./drive/My Drive/final_data/practice12_sgd_nesterov.model"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
# early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_sgd_nesterov = model.fit(X_train, y_train, batch_size=200, epochs=150, callbacks=[checkpoint], validation_split=0.2)

# best 모델 불러와서 예측 후 평가
best_model = load_model("./drive/My Drive/final_data/practice12_sgd_nesterov.model")

eval = best_model.evaluate(X_test, y_test)
print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

predictions = best_model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
print(f'Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print(f'Classification Report')
print(classification_report(y_test, y_pred, target_names=['df', 'mel', 'nv', 'tsu', 'vl']))

X_train, X_test, y_train, y_test = np.load("./drive/My Drive/final_data/practice7_data.npy", allow_pickle=True)
print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')

X_train = X_train.astype(float) / 255
X_test = X_test.astype(float) / 255

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

#8 모델 학습4
model_path = "./drive/My Drive/final_data/practice12_sgd_momentum_nesterov.model"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
# early_stop = EarlyStopping(monitor='val_loss', patience=10)
history_sgd_momentum_nesterov = model.fit(X_train, y_train, batch_size=200, epochs=150, callbacks=[checkpoint], validation_split=0.2)

# best 모델 불러와서 예측 후 평가
best_model = load_model("./drive/My Drive/final_data/practice12_sgd_momentum_nesterov.model")

eval = best_model.evaluate(X_test, y_test)
print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

predictions = best_model.predict(X_test)
y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)
print(f'Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print(f'Classification Report')
print(classification_report(y_test, y_pred, target_names=['df', 'mel', 'nv', 'tsu', 'vl']))

import matplotlib.pyplot as plt

#9 epoch-loss 그래프
# train data의 loss 그래프
train_loss_sgd = history_sgd.history['loss']
train_loss_sgd_momentum = history_sgd_momentum.history['loss']
train_loss_sgd_nesterov = history_sgd_nesterov.history['loss']
train_loss_sgd_momentum_nesterov = history_sgd_momentum_nesterov.history['loss']
x_len = range(150)
plt.plot(x_len, train_loss_sgd, marker='.', color='red', label='sgd')
plt.plot(x_len, train_loss_sgd_momentum, marker='.', color='orange', label='sgd_momentum')
plt.plot(x_len, train_loss_sgd_nesterov, marker='.', color='pink', label='sgd_nesterov')
plt.plot(x_len, train_loss_sgd_momentum_nesterov, marker='.', color='purple', label='sgd_momentum_nesterov')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()
# validation data의 loss 그래프
val_loss_sgd = history_sgd.history['val_loss']
val_loss_sgd_momentum = history_sgd_momentum.history['val_loss']
val_loss_sgd_nesterov = history_sgd_nesterov.history['loss']
val_loss_sgd_momentum_nesterov = history_sgd_momentum_nesterov.history['val_loss']
plt.plot(x_len, val_loss_sgd, marker='.', color='red', label='sgd')
plt.plot(x_len, val_loss_sgd_momentum, marker='.', color='orange', label='sgd_momentum')
plt.plot(x_len, val_loss_sgd_nesterov, marker='.', color='pink', label='sgd_nesterov')
plt.plot(x_len, val_loss_sgd_momentum_nesterov, marker='.', color='purple', label='sgd_momentum_nesterov')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Val Loss')
plt.show()

#10 epoch-accuracy 그래프
# train data의 accuracy 그래프
train_acc_sgd = history_sgd.history['accuracy']
train_acc_sgd_momentum = history_sgd_momentum.history['accuracy']
train_acc_sgd_nesterov = history_sgd_nesterov.history['accuracy']
train_acc_sgd_momentum_nesterov = history_sgd_momentum_nesterov.history['accuracy']
x_len = range(150)
plt.plot(x_len, train_acc_sgd, marker='.', color='red', label='sgd')
plt.plot(x_len, train_acc_sgd_momentum, marker='.', color='orange', label='sgd_momentum')
plt.plot(x_len, train_acc_sgd_nesterov, marker='.', color='pink', label='sgd_nesterov')
plt.plot(x_len, train_acc_sgd_momentum_nesterov, marker='.', color='purple', label='sgd_momentum_nesterov')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Train Acc')
plt.show()
# validation data의 accuracy 그래프
val_acc_sgd = history_sgd.history['val_accuracy']
val_acc_sgd_momentum = history_sgd_momentum.history['val_accuracy']
val_acc_sgd_nesterov = history_sgd_nesterov.history['val_accuracy']
val_acc_sgd_momentum_nesterov = history_sgd_momentum_nesterov.history['val_accuracy']
plt.plot(x_len, val_acc_sgd, marker='.', color='red', label='sgd')
plt.plot(x_len, val_acc_sgd_momentum, marker='.', color='orange', label='sgd_momentum')
plt.plot(x_len, val_acc_sgd_nesterov, marker='.', color='pink', label='sgd_nesterov')
plt.plot(x_len, val_acc_sgd_momentum_nesterov, marker='.', color='purple', label='sgd_momentum_nesterov')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Val Acc')
plt.show()