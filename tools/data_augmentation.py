import os
import shutil
import cv2
import numpy as np
from random import random, randint

from config import opt

# filter_list = ['background']
filter_list = []


def augmentation(compose, in_dir):
    '''
        加载图片，调用图形变换
    '''
    imgs_file = os.listdir(in_dir)
    imgs_path = [os.path.join(in_dir, name) for name in imgs_file]
    imgs_name = [name.split('_')[0] for name in imgs_file]

    for img_path, img_name in zip(imgs_path, imgs_name):
        if img_name in filter_list: # 过滤
            continue

        img = cv2.imread(img_path)
        compose([img], [img_name])


class Compose:
    '''
        图形变换的调用，存储
    '''
    def __init__(self, transforms, save_dir='/tmp/'):
        self.transforms = transforms
        self.save_dir = save_dir
        self.count = 0
        self.t = None   # current transforms

    def __call__(self, imgs, labels):
        for t in self.transforms:
            imgs, labels = t(imgs, labels)

            self.t = t
            self._save(imgs, labels)

    def _save(self, imgs, labels):
        for img, label in zip(imgs, labels):
            file_name = label + '_t' + self.t.__class__.__name__ + str(self.count) + '.jpg'
            cv2.imwrite(os.path.join(self.save_dir, file_name), img)
            self.count += 1

class Orign:
    '''
        保存原图
    '''
    def __call__(self, imgs, labels):
        return imgs, labels


class Perspective:
    def __init__(self):
        self.count = 1
        self.warp_rate = 0.2   # 四个角斜切大小占图像的比例

    def __call__(self, imgs, labels):
        p_imgs, p_labels = [], []

        for img, label in zip(imgs, labels):
            for _ in range(self.count):
                p_img, p_label = self._perspective(img, label)
                p_imgs.append(p_img)
                p_labels.append(p_label)
        return p_imgs, p_labels
    
    def _perspective(self, img, label):
        h, w, _ = img.shape
        h_gap, w_gap = h*self.warp_rate, w*self.warp_rate

        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        pts_dst = np.float32([
            [int(w_gap*random()), int(h_gap)*random()], 
            [int(w_gap*random()), h-int(h_gap*random())], 
            [w-int(w_gap*random()), h-int(h_gap*random())], 
            [w-int(w_gap*random()), int(h_gap*random())]
        ])

        M = cv2.getPerspectiveTransform(pts, pts_dst)
        p_img = cv2.warpPerspective(img, M, (h, w))

        # cv2.imshow('', p_img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        return img, label



if __name__ == "__main__":
    train_dir = opt.TRAIN_REGION_DIR
    train_aug_dir = opt.TRAIN_AUG_DIR
    test_dir = opt.TEST_REGION_DIR
    test_aug_dir = opt.TEST_AUG_DIR

    if os.path.exists(train_aug_dir):
        shutil.rmtree(train_aug_dir)
    if os.path.exists(test_aug_dir):
        shutil.rmtree(test_aug_dir)
    os.mkdir(train_aug_dir)
    os.mkdir(test_aug_dir)

    # for train
    print('start augmentation ...')
    train_compose = Compose([
        Orign(),
        Perspective()
    ],
    save_dir=train_aug_dir)
    augmentation(train_compose, train_dir)

    # for test
    test_compose = Compose([
        Orign(),
        Perspective()
    ],
    save_dir=test_aug_dir)
    augmentation(test_compose, test_dir)
    print('end augmentation!')

