"""
practice3
- train_test_split을 이용해서 train set와 test set으로 나눌 것이기 때문에,
    ISIC의 Train과 Test를 합치고 tsutsugamushi 폴더를 추가해 practice3라는 폴더 만듦
- practice3 하위 폴더 이름을 ["ak", "bcc", "df", "mel", "nv", "pbk", "scc", "sk", "tsu", "vl"]로 변경
- ImageDataGenerator 활용 후 최종 파일 수
    ak 이미지 파일 수: 1560
    bcc 이미지 파일 수: 4702
    df 이미지 파일 수: 1332
    mel 이미지 파일 수: 5446
    nv 이미지 파일 수: 4476
    pbk 이미지 파일 수: 5732
    scc 이미지 파일 수: 2364
    sk 이미지 파일 수: 960
    tsu 이미지 파일 수: 8061
    vl 이미지 파일 수: 1704
- https://github.com/lsjsj92/keras_basic/blob/master/7.%20predict_multi_img_with_CNN.ipynb 코드 주로 참고
"""
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

# practice1에서 만들었던, 해당 디렉토리의 파일 이름과 확장자를 일괄적으로 변경해주는 함수
def rename_files_and_change_file_extension(dir, new_name, new_ext):
    files = os.listdir(dir)
    for i, file in enumerate(files):
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name + str(i) + '.' + new_ext))

if __name__ == '__main__':
    # 0. tsu의 파일 이름을 변경해 확장자를 jpg로 바꿈
    # 파일 이름을 변경하는 이유는 확장자만 바꾸면 중복 이름 때문에 에러가 날 수 있기 때문
    tsu_dir = "./practice3/tsu"
    rename_files_and_change_file_extension(tsu_dir, 'tsu', 'jpg')

    # 1. ImageDataGenerator
    main_dir = "./practice3"
    categories = ["ak", "bcc", "df", "mel", "nv", "pbk", "scc", "sk", "tsu", "vl"]
    classes_num = len(categories)

    datagen = ImageDataGenerator(rotation_range=15,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.5,
                                 channel_shift_range=10,
                                 brightness_range=[0.5, 1.5])

    for cat in categories:
        cat_dir = main_dir + "/" + cat # 각 카테고리별 디렉토리
        files = glob.glob(cat_dir + "/*.jpg") # cat_dir의 jpg 파일들 다 가져옴
        for file in files:
            img = Image.open(file)
            if cat == "tsu": # "tsu" 폴더에서 발생하는 RGB 에러를 방지하기 위함
                img = img.convert('RGB')
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for _ in datagen.flow(x, batch_size=1, save_to_dir=cat_dir,
                                  save_prefix=os.path.basename(file).split('.')[0]+'_copy',
                                  save_format='jpg'):
                i += 1
                if i > 10: # 한 이미지당 열 장씩만 더 불리기 위함 # 원본 이미지 + 복사된 이미지 10
                    break


    # 2. 학습 데이터와 테스트 데이터로 분리해서 npy 파일로 저장
    image_w, image_h = 64, 64
    X = []
    y = []
    for idx, cat in enumerate(categories):
        # one-hot label
        label = [0 for i in range(classes_num)]
        label[idx] = 1

        cat_dir = main_dir + "/" + cat
        files = glob.glob(cat_dir+"/*.jpg")
        print(cat, "이미지 파일 수:", len(files)) # 각 카테고리별 이미지 파일 수를 파악하기 위함
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h), Image.ANTIALIAS) # Image.ANTIALIAS는 resizing을 부드럽게 처리하기 위함
            data = np.asarray(img)

            X.append(data)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    xy = (X_train, X_test, y_train, y_test) # np.load 시 데이터를 편하게 불러오기 위해 tuple로 저장
    np.save("./practice3_data.npy", xy)

    # 3. numpy 데이터를 불러와서 모델 학습
    X_train, X_test, y_train, y_test = np.load("./practice3_data.npy", allow_pickle=True)
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}') # X_train: (29069, 64, 64, 3), y_train: (29069, 10)
    print(f'X_test: {X_test.shape}, y_train: {y_test.shape}') # X_test: (7268, 64, 64, 3), y_train: (7268, 10)

    # 정규화
    X_train = X_train.astype(float) / 255
    X_test = X_test.astype(float) / 255

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_dir = "./model"

    if not os.path.exists(model_dir): # model_dir이 없을 경우 폴더 생성
        os.mkdir(model_dir)

    model_path = model_dir + '/practice3.model'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=6)
    model.summary()

    history = model.fit(X_train, y_train, batch_size=200, epochs=100, callbacks=[checkpoint, early_stop], validation_split=0.2)
    # Train on 23255 samples, validate on 5814 samples

    # test 데이터의 loss, accuracy
    eval = model.evaluate(X_test, y_test)
    print(f'Test loss: {eval[0]}, accuracy: {eval[1]}')

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