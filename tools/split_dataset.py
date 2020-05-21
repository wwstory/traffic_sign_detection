import os
import shutil
import random

from config import opt



def _copy_file(src, dst):
    '''
        文件的复制
    '''
    if os.path.isfile(src):
        shutil.copyfile(src, dst)
    else:
        print(f'! [{src}] is not exist')


def split_dataset(dataset_dir, train_dir, test_dir, rate=[0.8, 0.2], shuffle=True):
    '''
        分割训练集和测试集
    '''
    data_img_dir = os.path.join(dataset_dir, opt.IMAGE_NAME)
    data_ann_dir = os.path.join(dataset_dir, opt.ANNO_NAME)
    train_img_dir = os.path.join(train_dir, opt.IMAGE_NAME)
    train_ann_dir = os.path.join(train_dir, opt.ANNO_NAME)
    test_img_dir = os.path.join(test_dir, opt.IMAGE_NAME)
    test_ann_dir = os.path.join(test_dir, opt.ANNO_NAME)

    images_names = os.listdir(data_img_dir)
    if shuffle:
        random.shuffle(images_names)
    anns_names = [name.split('.')[0] + '.xml' for name in images_names]

    num = len(images_names)
    train_images_name = images_names[:int(num*rate[0])]
    train_anns_name = anns_names[:int(num*rate[0])]
    test_images_name = images_names[int(num*rate[0]):]
    test_anns_name = anns_names[int(num*rate[0]):]

    for img_name, ann_name in zip(train_images_name, train_anns_name):
        _copy_file(os.path.join(data_img_dir, img_name), os.path.join(train_img_dir, img_name))
        _copy_file(os.path.join(data_ann_dir, ann_name), os.path.join(train_ann_dir, ann_name))
    for img_name, ann_name in zip(test_images_name, test_anns_name):
        _copy_file(os.path.join(data_img_dir, img_name), os.path.join(test_img_dir, img_name))
        _copy_file(os.path.join(data_ann_dir, ann_name), os.path.join(test_ann_dir, ann_name))


if __name__ == "__main__":
    dataset_dir = opt.DATASET_ORGIN_DIR
    train_dir = opt.TRAIN_DATASETS_DIR
    test_dir = opt.TEST_DATASETS_DIR

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.mkdir(train_dir)
    os.mkdir(train_dir + opt.IMAGE_NAME)
    os.mkdir(train_dir + opt.ANNO_NAME)
    os.mkdir(test_dir)
    os.mkdir(test_dir + opt.IMAGE_NAME)
    os.mkdir(test_dir + opt.ANNO_NAME)

    print('start split ...')
    split_dataset(dataset_dir, train_dir, test_dir)
    print('end split!')
