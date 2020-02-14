""" 이미지 전처리1
- 이미지 사이즈 조정 (64,64)
- RGB값을 리스트로 저장하기 """

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def img_preprocessing(path, size=(64, 64), file_name='', n=10000):
    """ 폴더 안의 파일을 불러와 파일이름을 리스트로 저장하고,
        이미지 사이즈, 확장자, 이름 변환
    :param path: 이미지가 저장되어 있는 폴더 경로
    :param size: (64, 64) 튜플형식으로 입력
    :param file_name: 저장할 파일 이름
    :param n: 폴더 안의 파일 n개 까지만 변환
    :return: 변환된 파일을 폴더 안에 저장 """

    file_list = os.listdir(path)
    print('file_list: {}'.format(file_list))
    idx = 0
    for name in file_list[ :n]:
        im = Image.open(path + name)
        im = im.convert('RGB').resize(size)  # rgb로 변환해야 jpg로 변환 가능함
        idx += 1
        im.save(f'{file_name}{idx}.jpg')



if __name__ == '__main__':
    # 폴더 안에 이미지 불러와 이미지 사이즈, 확장자(jpg), 파일이름 변환
    ### 쯔쯔가무시 데이터 변환 ###
    path = 'C:/dev/final_project/images/refined_images/eschar/'
    img_preprocessing(path, file_name='tsutsu', n=10)


    # # 변환 전 이미지 사이즈 알아보기
    # img1 = Image.open('C:/Users/image/OneDrive/final_project/images/refined_images/eschar/eschar (10).jpeg')
    # img1.save(f'resized_img.png')
    # img1 = np.array(img1)
    # print(img1.shape)

    ### 확장자(각각 차원이 다름) ###
    # gif : shape (127, 176) 2차원
    # jpg, jpeg : (181, 193, 3), 3차원
    # png : (100, 120, 4), 4차원


    #### nevus 데이터 변환 ####
    # 폴더 안에 이미지 불러오기
    path = 'C:/dev/final_project/images/skin_cancer/Train/nevus/'
    img_preprocessing(path, file_name='nevus', n=10)


