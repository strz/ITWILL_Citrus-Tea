import glob
import os
import sys

sys.path.append(os.pardir)
from PIL import Image
from keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img
import numpy as np

print('CURRENT WORKING DIRECTORY: ', os.getcwd())

"""
SIMPLE CNN NETWORK

refer : MJ 01_size, extension, name. , 02_img_preprocessing, 03_img_generating, EJ practice3_imagex11+label10.py.

IN ORDER TO RUN THIS CODE YOU MUST HAVE TARGET IMAGE DIRECTORY, WHICH HAS SUB-DIRECTORIES
WHICH HAVE NON-DUPLEX IMAGE FILES.

YOU MAY IGNORE BELOW if __name__=='__main__':.


"""


################################################################################################

# refer from MJ 01 size extension name.
def image_resizer(path, height=64, width=64, resized_name='resized_', limitation=False, maximum=10000):
    """
    Resizes image files with certain height and widths. Height and Width must be natural number. If one of H, W is negative, return ValueError.
    H, W will be rescaled to nearst natural number.

    parametres:
    path : String. Specifies where the images we want to resize are.
    height, width : (Recommended) Natural numbers. If negative or rational numbers are given, the function will handle it properly.
    resized_name : String. When resizing is complete, it will be added as prefix. Attaching by suffix is not supported.
    limitation : Boolean. Default by False. If True, then the function will resize images up to parameter 'maximum'.
    maximum : (Recommended) Natural number. Constricts numbers of resizing images.

    DOES NOT RETURN. but resized images will be stored at your own working directory.
    """
    print('INITIALIZING IMAGE RESIZING')
    if height * width <= 0 or height + width < 0:
        # 1. if one of h or w is negative / 2. if either h and w are negative.
        raise ValueError('Invalid input')
    else:
        # truncate height and width as int type.
        height = int(round(height, 0))
        width = int(round(width, 0))

    file_directory = os.listdir(path)

    if limitation:
        # If we checked limitation as true, then image_resizer function will resize files till maximum=x.
        if maximum <= 0:
            raise ValueError('Invalid input')
        maximum = int(round(maximum, 0))

        file_directory = file_directory[:maximum]

    for x in enumerate(file_directory):
        print('SELECTED FILE : ', x[1])
        img_file = Image.open(path + '/' + x[1])
        img_file_converted = img_file.convert('RGB').resize((height, width))
        img_file_converted.save(fp=resized_name + str(x[0]) + '.jpg')
        print('RESIZED AS : ', resized_name + str(x[0]) + '.jpg')


################################################################################################

def array_transform(path, limitation=False, maximum=10000):
    """
    Transforms images into array. Note that this function is not able to access subdirectories. Thus you'd better to give a path variable
    where the 'RESIZED' images which are created by image_resizer are.

    parameters:
    path : String. A directory where the targeted images are. Take it with resized image directory.
    limitation : Boolean. Same as image_resizer.
    maximum : Natural number. Same as image_resizer.

    return : Numpy Array. where width and height are decided by each image(If you unified their size, all elements will have same shape)
    """
    file_directory = os.listdir(path)

    if limitation:
        if maximum <= 0:
            raise ValueError('Invalid input')
        maximum = int(round(maximum, 0))

    transformed_array = []

    for x in file_directory:  # we don't need to make it as enumerate.
        img_file = Image.open(path + '/' + x)
        print(f'SELECTED FILE : {path + "/" + x}')
        img_file_converted = img_to_array(img_file)  # use keras module here
        transformed_array.append(img_file_converted)

    return np.array(transformed_array)


################################################################################################

def one_hot_encoding_labeler(common_path, sub_directory):
    """
Attaches One_Hot_Encoding to each sub directory.

Parameters:
common_path : String. Target upper directory of sub directories. For example, if a directory 'SD' has sub directories 'A1, A2', then
'SD' is the upper directory of 'A1',' A2'.
sub_directory : list. If 'SD' is upper directory, then sub_directory=['A1', 'A2'].

return : Numpy Array. each element will have length as len(sub_directory), quasi-sparse matrix.
    """
    print('One Hot Encoding Initialised')
    OHE = []
    cnt = 0
    for index, category in enumerate(sub_directory):
        sub_category_path = common_path + '/' + category
        print(f'sub_category_path_check ={sub_category_path}')
        print(f'Selected path :{sub_category_path}')
        files = glob.glob(sub_category_path + '/*.*')
        print('files =', files)
        for i in files:
            print(f'SELECTED FILE : {i}')
            if os.path.isfile(i):
                label = [0 for _ in range(len(sub_directory))]
                label[index] = 1
                OHE.append(label)
                print('appended!')
                cnt += 1
    print('labeled items : ', cnt)

    return np.array(OHE)


def image_generator(source_path, save_path, duplication=10):
    """
    REFERED : MJ 03 IMAGE GENERATING.PY

    Duplicates images. In order to handle generating setting, modify function itself.
    Eventhough save_path does not exist, the function will create the directory.
    Note that this function cannot access to subdirectories.

    Parameters:
    source_path : String. A directory where image files we'd like to duplicate are. MUST BE VALID PATH.
    the function will detect 'bmp',jpg', 'jpeg', 'png' files only. Other extensions are ignored.
    save_path : String. A directory where we'd like to store converted images.
    If save_path is not valid, the function will create directory with python.builtins.os module.
    duplication : Natural number. Determines the number of duplication.

    """
    print('INITIALIZING IMAGE GENERATOR...')
    source_path = source_path.replace('\\', '/')
    save_path = save_path.replace('\\', '/')

    if duplication <= 1:
        raise ValueError('Invalid Input')
    else:
        duplication = round(int(duplication), 0)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    power_generator = ImageDataGenerator(rescale=1. / 255,
                                         rotation_range=30,  # 각도 범위내 회전
                                         width_shift_range=0.1,  # 수평방향
                                         height_shift_range=0.1,  # 수직방향
                                         brightness_range=[0.2, 0.7],  # 밝기
                                         shear_range=0.7,  # 시계반대방향
                                         zoom_range=[0.9, 1.1],
                                         horizontal_flip=True,  # 수평방향 뒤집기
                                         vertical_flip=True,
                                         fill_mode='nearest')

    file_list = os.listdir(source_path)

    delition = 1

    for ext in file_list:
        if os.path.isdir(ext):
            print(f'DELETED BY IS NOT FILE : {delition, ext}')
            # remove all items which aren't files. such as directories.
            delition += 1
            file_list.remove(ext)

    for ext in file_list:
        # if an item has no extension, remove it.
        if len(ext.split('.')) == 1:
            print(f'DELETED BY EXTENSION : {delition, ext}')
            delition += 1
            file_list.remove(ext)

    for ext in file_list:
        if not ext.split('.')[1] in ['bmp', 'jpg', 'png', 'jpeg', 'BMP', 'JPG', 'PNG', 'JPEG']:
            print(f'DELETED BY UNSUPPORTED EXTENSION : {delition, ext}')
            delition += 1
            file_list.remove(ext)

    print(f'NUMBER OF FILES TO BE DUPLICATED : {len(file_list)}')

    # for filename in file_list: print(filename)

    cnt = 1
    for name in file_list:
        img = load_img(source_path + '/' + name)
        j = img_to_array(img)
        j = j.reshape((1,) + j.shape)  # j 전체를 하나의 []로 묶음
        i = 0
        for _ in power_generator.flow(j, save_to_dir=save_path, save_prefix=name.split('.')[0] + '_gen',
                                      save_format='jpg'):
            print(f' {cnt}/{len(file_list) * duplication} DUPLICATING IMAGE {i} :: {name}...')
            i += 1
            cnt += 1
            if i >= duplication:
                break
    print('END OF IMAGE DUPLICATION')


"""
Preparation
-> directory : must have each sub category.

1. resize images for each sub directory.

2. apply array_transform for each sub directory. and save it.

3. apply one_hot_encoding_labeler to main directory. save it as well.

"""

if __name__ == '__main__':
    # subdirectories.
    sd = ['actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma', 'melanoma', 'nevus',
          'pigmented_benign_keratosis', 'scrub_typhus', 'seborrheic_keratosis', 'squamous_cell_carcinoma',
          'vascular_lesion']
    for x in sd:
        print(x)

    # common path.
    LDB = os.path.abspath(os.path.join('..', '..', 'LDB')).replace('\\', '/')
    print('TARGET DIRECTORY : ', LDB)

    # 1. RESIZE IMAGES FOR EACH SUBDIRECTORY. USE FOR STATEMENT.

    # for i in sd: image_resizer(path=LDB+'/'+i, resized_name=i+'_') # DONE!

    # 2. TRANSFORM IMAGES INTO NUMPY ARRAY.

    LDB += '_RESIZED'

    LDB_array = array_transform(path=LDB) / 255.0
    # np.save(os.getcwd().replace('\\','/')+'/LDB_array.npy', LDB_array) # we have already stored LDB_array.npy.

    # apply one_hot_encoding.

    # LDB -= '_RESIZED'

    # LDB_ohe = one_hot_encoding_labeler(common_path=LDB, sub_directory=sd)
    # np.save(os.getcwd().replace('\\','/')+'/LDB_ohe.npy', LDB_ohe) # we have saved one-hot-encoded file for convenience.
