""" 이미지 부풀리기
- 사진당 10장 부풀리기
"""

from keras_preprocessing.image import ImageDataGenerator , load_img , img_to_array
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D , MaxPool2D , Flatten , Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


def file_to_array(path, n=10000):
    """ 폴더 안의 파일을 불러와 리스트로 저장하고,
        이미지를 픽셀 배열로 변환
    :param n: 원하는 만큼의 이미지 수"""
    file_list = os.listdir(path)
    print('file_list{}'.format(file_list))
    X = []
    for name in file_list[:n]:
        im = Image.open(path + name)
        im = img_to_array(im)   # list
        X.append(im)
    X = np.array(X)   # list를 array로 변환
    return X


#  폴더 안에 있는 파일이름으로 one-hot-encoding
def creat_label(path):
    """ 폴더 안에 있는 파일이름으로 one-hot-encoding """
    file_list = os.listdir(path)
    print(file_list)
    y = []
    for name in file_list:
        if 'tsutsu' in name:
            tsutsu_encoding = np.array([1, 0, 0, 0, 0])
            y.append(tsutsu_encoding)
        elif 'df' in name:
            df_encoding = np.array([0, 1, 0, 0, 0])
            y.append(df_encoding)
        elif 'mel' in name:
            mel_encoding = np.array([0, 0, 1, 0, 0])
            y.append(mel_encoding)
        elif 'nv' in name:
            nv_encoding = np.array([0, 0, 0, 1, 0])
            y.append(nv_encoding)
        elif 'vl' in name:
            vl_encoding = np.array([0, 0, 0, 0, 1])
            y.append(vl_encoding)
    return np.array(y)


if __name__ == '__main__':
    # generator 만들기
    train_datagen = ImageDataGenerator(rescale=1. / 255 ,
                                       rotation_range=30 ,  # 각도 범위내 회전
                                       width_shift_range=0.1 ,  # 수평방향
                                       height_shift_range=0.1 ,  # 수직방향
                                       brightness_range=[0.2 , 0.7] ,  # 밝기
                                       shear_range=0.7 ,  # 시계반대방향
                                       zoom_range=[0.9 , 1.1] ,
                                       horizontal_flip=True ,  # 수평방향 뒤집기
                                       vertical_flip=True ,
                                       fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1 / 255)  # 픽셀 정규화만 해주고 나머지는 그대로 둠.

    path = "./code7"

    file_list = os.listdir(path)
    print('file_list: {}'.format(file_list))
    print(len(file_list))

    save_to_dir = "./code7"

    for name in file_list:
        img = load_img(path + '/' + name)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # x 전체를 하나의 []로 묶음
        i = 0
        for _ in train_datagen.flow(x, save_to_dir=save_to_dir, save_prefix=name.split('.')[0] + '_gen',
                                    save_format='jpg'):
            i += 1
            if i >= 10:
                break
    #
    # x = np.array([0,1,2,3,4])
    np.save("./code7.npy", x)   # x_save.npy
    # x_save_load = np.load("./code1.npy")
    # print(x_save_load)
    #
    # # 폴더 안에 이미지를 불러와 픽셀 배열로 저장해 하나의 Dataset로 만들기
    # path = "./code1"
    # X = file_to_array(path)
    # print(X[0:2])
    # print(f'X.shape: {np.shape(X)}')   # (11316, 64, 64, 3) = (sample수, h, w, c)
    # print(type(X))
    #
    # # # y label 만들기
    # path = "./code1"
    # y = creat_label(path)
    # print(y)
    # print(type(y))
    # print(len(y))

    # # train, test 데이터로 나누고, 데이터 정규화 하기
    # np.random.seed(210)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(len(X_train), len(X_test), len(y_train), len(y_test))
    # print('X_train:', X_train[0:2])
    # print('y_train:', y_train[0:2])
    # print('X_test', X_test[0:2])
    # print('y_test', y_test[0:2])
    #
    # xy = (X_train, X_test, y_train, y_test)
    # np.save('./xy_save.npy', xy)   # x_save.npy
    # X_train, X_test, y_train, y_test = np.load('./xy_save.npy', allow_pickle=True)
    # print(X_train.shape)
    #
    # # # Y_train, Y_test를 one-hot-encoding 변환 - 함수에서 만들어서 생략함
    # # y_train = to_categorical(y_train, 2, dtype='float16')
    # # y_test = to_categorical(y_test, 2, dtype='float16')
    #
    # # 정규화 시키기
    # X_train = X_train.astype('float16') / 255
    # X_test = X_test.astype('float16') / 255
    #
    # print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')  # X_train: (823, 64, 64, 3), y_train: (823, 2)
    # print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')  # X_test: (206, 64, 64, 3), y_test: (206, 2)
    #
    # print(f'X_train: {X_train[0:2]}, X_test: {X_test[0:2]}')



    # ### CNN 모델 만들기 ###
    # # 신경망 모델 생성 - Sequential 클래스 인스턴스 생성
    # model = Sequential()
    #
    # # 신경망 모델에 은닉층, 출력층 계층(layers)들을 추가
    # # Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense
    # # Conv2D 활성화 함수: ReLU
    # # Dense 활성화 함수: ReLU, Softmax
    # model.add(Conv2D(filters=32,         # 필터 갯수
    #                  kernel_size=(3,3),  # 필터의 height/width
    #                  activation='relu',  # 활성화 함수
    #                  input_shape=(64, 64, 3)))  # 입력데이터의 shape (h,w,c)순서임.
    #
    # model.add(MaxPool2D(pool_size=2))   # 이미지가 줄어듦.
    # model.add(Flatten())  # keras에서는 Dense 층에 넣기 전에 모두 펴줘야함.
    # model.add(Dense(128, activation='relu'))   # 완전 연결 은닉층
    # model.add(Dense(2, activation='softmax'))  # 출력층 (위에서 함수로 one-hot-encoding 해줌)
    #
    # model.summary()
    #
    # # 신경망 모델 컴파일
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # # 신경망 모델의 성능 향상이 없는 경우 중간에 epoch을 빨리 중지시키기 위해서
    # early_stop = EarlyStopping(monitor='val_loss',
    #                            verbose=1,
    #                            patience=10)
    #
    # # 신경망 학습
    # history = model.fit(X_train, y_train,
    #                     batch_size=100,  # 전체갯수를 batch_size로 나눈 만큼 반복
    #                     epochs=50,   # 에폭만큼 파라미터 업데이트
    #                     verbose=1,
    #                     callbacks=[early_stop],
    #                     validation_split=0.2)
    #                     # 충분히 데이터가 많은 땐, validation_split를 사용하는 것이 바람직.
    #                     # 학습 단계에서 test 데이터를 줘버리면 당연히 test 데이터를 검증할 때
    #                     # 정확도가 높게 나올 수 밖에 없는 구조 (따라서 train 데이터에서 떼어내서 검증하는
    #                     # validation_split를 쓰는 것이 더 정확하게 검증하는 방법임!)
    #
    # # 테스트 데이터를 사용해서 신경망 모델을 평가
    # # 테스트 데이터의 Loss, Accuracy
    # eval = model.evaluate(X_test, y_test)
    # print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')
    # # Test loss: 8.387081241607666, accuracy: 0.44999998807907104 (위의 fit 결과에서 마지막 값과 같음.)
    #
    # # 학습 데이터와 테스트 데이터의 Loss 그래프
    # train_loss = history.history['loss']  # history dictionary에 저장된 'loss' 키를 갖는 value들을 가져옴
    # test_loss = history.history['val_loss']
    #
    # x = range(len(train_loss))
    # plt.plot(x, train_loss, marker='.', color='red', label='Train loss')
    # plt.plot(x, test_loss, marker='.', color='blue', label='Test loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    #
    # # 학습 데이터, 테스트 데이터의 정확도 그래프
    # train_accuracy = history.history['accuracy']
    # test_accuracy = history.history['val_accuracy']
    #
    # x = range(len(train_accuracy))
    # plt.plot(x, train_accuracy, marker='.', color='red', label='Train loss')
    # plt.plot(x, test_accuracy, marker='.', color='blue', label='Test loss')
    # plt.xlabel('epoch')
    # plt.ylabel('accuracy')
    # plt.legend()
    # plt.show()
