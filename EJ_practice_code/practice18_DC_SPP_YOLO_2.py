import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from practice4_5cats import get_dir
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Dense
from SPPNet import SpatialPyramidPooling
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

def save_my_train_test(dir, save_path, test_size=0.2):
    image_w, image_h = 52, 52
    X = []
    y = []
    categories = os.listdir(dir)
    cat_num = len(categories)
    for idx, cat in enumerate(categories):
        label = [0 for i in range(cat_num)]
        label[idx] = 1
        cat_dir = dir + "/" + cat
        imgs = os.listdir(cat_dir)
        for i, img in enumerate(imgs):
            img_path = cat_dir + '/' + img
            raw_img = Image.open(img_path)
            raw_img = raw_img.convert("RGB")
            raw_img = raw_img.resize((image_w, image_h), Image.ANTIALIAS)
            data = np.asarray(raw_img)

            X.append(data)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    xy = (X_train, X_test, y_train, y_test)
    np.save(save_path, xy)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'

    # 1. practice7로 디렉토리 설정 후 확인
    practice7_dir = './practice7'
    get_dir(practice7_dir)
    #  practice7의 하위 폴더:['df', 'mel', 'nv', 'tsu', 'vl']
    #  df 폴더의 이미지 파일 수: 5023
    #  mel 폴더의 이미지 파일 수: 5254
    #  nv 폴더의 이미지 파일 수: 5351
    #  tsu 폴더의 이미지 파일 수: 5374
    #  vl 폴더의 이미지 파일 수: 5001

    # 2. numpy 데이터를 불러와서 shape 확인
    X_train, X_test, y_train, y_test = np.load("./practice7_data.npy", allow_pickle=True)
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')  # X_train: (20802, 64, 64, 3), y_train: (20802, 5)
    print(f'X_test: {X_test.shape}, y_train: {y_test.shape}')  # X_test: (5201, 64, 64, 3), y_train: (5201, 5)

    # 3. 정규화
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255

    # 4. DC-SPP-YOLO
    model = Sequential()
    # DC-SPP-YOLO의 default input size는 416x416x3.
    # 편의를 위해 이미지 크기 64x64를 그대로 사용
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPool2D(pool_size=2))
    # model.add(Dropout(rate=0.25))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(512, activation='relu')) # 편의를 위해 생략
    # model.add(Dense(512, activation='relu')) # 편의를 위해 생략
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # 'sgd'

    model_path = './model/practice18.h5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model.summary()
    plot_model(model, to_file='practice18.png')

    history = model.fit(X_train, y_train, batch_size=200, epochs=100, callbacks=[checkpoint, early_stop],
                        validation_split=0.2)
    # Train on 16640 samples, validate on 4160 samples

    # test 데이터의 loss, accuracy
    best_model = load_model('./model/practice18.h5', custom_objects={'SpatialPyramidPooling': SpatialPyramidPooling})
    eval = best_model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')
    # Test loss: 0.20032099020570499, accuracy: 0.9232839941978455

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
    predictions = best_model.predict(X_test)
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    print('* Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('* Classification Report')
    print(classification_report(y_test, y_pred, target_names=os.listdir(practice7_dir)))