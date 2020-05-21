import os
import shutil
import cv2
from skimage import feature as ft
import joblib

from config import opt


classes_names = opt.classes_names
label_names = opt.label_names


def extract_hog_feature(in_dir, save_dir, hog_file_name='hog_feature'):
    '''
        加载图片，获取特征，存储特征信息
    '''
    imgs_file = os.listdir(in_dir)
    imgs_path = [os.path.join(in_dir, name) for name in imgs_file]
    # 不属于类别里的，视为背景(0)
    imgs_name = [name.split('_')[0] if name.split('_')[0] in classes_names else 0  for name in imgs_file]
    labels_name = [label_names[name.split('_')[0]] if name.split('_')[0] in classes_names else 0  for name in imgs_file]

    hog_feature = []

    for img_path, img_name, label_name in zip(imgs_path, imgs_name, labels_name):
        img = cv2.imread(img_path)
        feature = get_hog_feature(img)

        hog_feature.append({'file_path': img_path, 'classes_name': img_name, 'label': label_name, 'feature':feature})

    joblib.dump(hog_feature, os.path.join(save_dir, hog_file_name))


def get_hog_feature(img):
    '''
        获取hog特征
    '''
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)
    block_norm = 'L2'

    feature = ft.hog(
        img, 
        orientations=orientations, 
        pixels_per_cell=pixels_per_cell, 
        cells_per_block=cells_per_block, 
        block_norm=block_norm, 
        transform_sqrt=True
    )
    return feature


if __name__ == "__main__":
    train_dir = opt.TRAIN_AUG_DIR
    train_hog_dir = opt.TRAIN_HOG_DIR
    test_dir = opt.TEST_AUG_DIR
    test_hog_dir = opt.TEST_HOG_DIR
    hog_file_name = opt.HOG_FILE_NAME

    if os.path.exists(train_hog_dir):
        shutil.rmtree(train_hog_dir)
    if os.path.exists(test_hog_dir):
        shutil.rmtree(test_hog_dir)
    os.mkdir(train_hog_dir)
    os.mkdir(test_hog_dir)
    
    print('start extract ...')
    # for train
    extract_hog_feature(train_dir, train_hog_dir, hog_file_name)
    # for test
    extract_hog_feature(test_dir, test_hog_dir, hog_file_name)
    print('end extract!')
    