import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense
from SPPNet import SpatialPyramidPooling
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    # 1. practice7으로 디렉토리 설정
    practice7_dir = './practice7'

    # 2. numpy 데이터를 불러와서 shape 확인
    X_train, X_test, y_train, y_test = np.load("./practice7_data.npy", allow_pickle=True)
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')  # X_train: (20802, 64, 64, 3), y_train: (20802, 5)
    print(f'X_test: {X_test.shape}, y_train: {y_test.shape}')  # X_test: (5201, 64, 64, 3), y_train: (5201, 5)

    # 3. SpatialPyramidPooling층 추가해서 모델 학습
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 'sgd'

    model_dir = "./model"
    model_path = model_dir + '/practice8_2.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=6)
    model.summary()

    history = model.fit(X_train, y_train, batch_size=200, epochs=100, callbacks=[checkpoint, early_stop],
                        validation_split=0.2)
    # Train on 16641 samples, validate on 4161 samples

    # test 데이터의 loss, accuracy
    eval = model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')
    # Test loss: 0.27035794288945686, accuracy: 0.8996346592903137 # Test loss: 0.2838927661540577, accuracy: 0.8886752724647522

    # train 데이터, validation 데이터의 손실 그래프
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    x_len = range(len(train_loss))
    plt.plot(x_len, train_loss, marker='.', color='red', label='Train loss')
    plt.plot(x_len, val_loss, marker='.', color='blue', label='Val loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    # train 데이터, validation 데이터의 정확도 그래프
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(x_len, train_acc, marker='.', c='red', label='Train Acc.')
    plt.plot(x_len, val_acc, marker='.', c='blue', label='Val Acc.')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # 4. 만든 모델 불러와서 예측 후 평가
    model = load_model('./model/practice8_2.model', custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})

    predictions = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    print('* Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('* Classification Report')
    print(classification_report(y_test, y_pred, target_names=os.listdir(practice7_dir)))