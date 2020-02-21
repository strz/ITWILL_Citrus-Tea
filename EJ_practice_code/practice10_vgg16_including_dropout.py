"""
practice10
- pre_trained vgg16에 dropout층 추가
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


if __name__ == '__main__':
    # 1. practice7으로 디렉토리 설정
    practice7_dir = './practice7'

    # 2. numpy 데이터를 불러와서 shape 확인
    X_train, X_test, y_train, y_test = np.load("./practice7_data.npy", allow_pickle=True)
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')  # X_train: (20802, 64, 64, 3), y_train: (20802, 5)
    print(f'X_test: {X_test.shape}, y_train: {y_test.shape}')  # X_test: (5201, 64, 64, 3), y_train: (5201, 5)

    # 3. 이미 훈련된 vgg16 모델에 globalaveragepooling2d, dense 연결해서 사용
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255

    pre_trained_vgg = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
    # include_top: 네트워크의 최상단에 3개의 완전 연결 레이어를 넣을지 여부
    # input_shape: include_top이 False일 경우만 특정. 그렇지 않다면 인풋의 형태가 (224, 224, 3)이어야 함.
    pre_trained_vgg.trainable = False # 이미지넷으로 학습된 값들을 그대로 사용할 것이기 때문에
    pre_trained_vgg.summary()

    additional_model = Sequential()
    additional_model.add(pre_trained_vgg)
    additional_model.add(GlobalAveragePooling2D()) # pre_trained_vgg의 디폴트된 input_shape과 다른 형태를 넣었기 때문에 Flatten()대신 Global AveragePooling2D() 이용. 이 부분은 더 공부 필요
    additional_model.add(Dense(4096, activation='relu'))
    additional_model.add(Dropout(0.5))
    additional_model.add(Dense(4096, activation='relu'))
    additional_model.add(Dropout(0.5))
    additional_model.add(Dense(1000, activation='relu'))
    additional_model.add(Dense(5, activation='softmax'))
    additional_model.summary()
    plot_model(additional_model, to_file='practice10.png')

    additional_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 'sgd'

    model_dir = "./model"
    model_path = model_dir + '/practice10.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)


    history = additional_model.fit(X_train, y_train, batch_size=200, epochs=100, callbacks=[checkpoint, early_stop],
                        validation_split=0.2)
    # Train on 16641 samples, validate on 4161 samples

    # test 데이터의 loss, accuracy
    eval = additional_model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')
    # Test loss: 0.34357965903611304, accuracy: 0.8848298192024231

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
    model = load_model('./model/practice10.model')

    predictions = model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    print('* Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('* Classification Report')
    print(classification_report(y_test, y_pred, target_names=os.listdir(practice7_dir))) # target_names=['df', 'mel', 'nv', 'tsu', 'vl']