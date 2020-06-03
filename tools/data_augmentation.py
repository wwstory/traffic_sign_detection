import os
import shutil
import cv2
import numpy as np
from random import random, randint
from math import sin, cos, atan2, pi, sqrt

from config import opt

need_list = ['background', 'left', 'right']
# filter_save_list = ['background']   # 会拉大background和其它类训练集的数量差距


def augmentation(compose, in_dir, out_dir):
    '''
        加载图片，调用图形变换
    '''
    imgs_file = os.listdir(in_dir)
    imgs_path = [os.path.join(in_dir, name) for name in imgs_file]
    imgs_name = [name.split('_')[0] for name in imgs_file]

    for img_path, img_name in zip(imgs_path, imgs_name):
        if img_name not in need_list:
            continue
        # if img_name in filter_save_list: # 过滤但需要复制一份
        #     img = cv2.imread(img_path)
        #     cv2.imwrite(os.path.join(out_dir, img_path.split('/')[-1]), img)
        #     continue

        img = cv2.imread(img_path)
        compose([img], [img_name])


class Compose:
    '''
        图形变换的调用，存储
    '''
    def __init__(self, transforms, save_dir='/tmp/', use_new_img=False):
        self.transforms = transforms
        self.save_dir = save_dir
        self.use_new_img = use_new_img
        self.count = 0
        self.t = None   # current transforms

    def __call__(self, imgs, labels):
        for t in self.transforms:
            self.t = t
            if self.use_new_img:
                _imgs, _labels = [], []
                for img, label in zip(imgs, labels):
                    _imgs.append(img.copy())
                    _labels.append(label)
                _imgs, _labels = t(_imgs, _labels)
                self._save(_imgs, _labels)
            else:
                imgs, labels = t(imgs, labels)
                self._save(imgs, labels)

    def _save(self, imgs, labels):
        for img, label in zip(imgs, labels):
            file_name = label + '_' + self.t.__class__.__name__ + str(self.count) + '.jpg'
            cv2.imwrite(os.path.join(self.save_dir, file_name), img)
            self.count += 1

class Orign:
    '''
        保存原图
    '''
    def __call__(self, imgs, labels):
        return imgs, labels


class Perspective:
    '''
        图形斜切变换
    '''
    def __init__(self):
        self.count = 1
        self.warp_rate = 0.2   # 四个角斜切大小占图像的比例

    def __call__(self, imgs, labels):
        _imgs, _labels = [], []

        for img, label in zip(imgs, labels):
            for _ in range(self.count):
                _img, _label = self._perspective(img, label)
                _imgs.append(_img)
                _labels.append(_label)
        return _imgs, _labels
    
    def _perspective(self, img, label):
        h, w, _ = img.shape
        hr, wr = h*self.warp_rate, w*self.warp_rate

        # 左上，右上，左下，右下
        pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts_dst = np.float32([
            [wr*random(), hr*random()], 
            [w-wr*random(), hr*random()], 
            [wr*random(), h-hr*random()], 
            [w-wr*random(), h-hr*random()]
        ])

        M = cv2.getPerspectiveTransform(pts, pts_dst)
        p_img = cv2.warpPerspective(img, M, (h, w))

        # cv2.imshow('', p_img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        return p_img, label


class LostPart:
    '''
        丢失部分信息
    '''
    count = 1
    rate = 0.3

    def __call__(self, imgs, labels):
        _imgs, _labels = [], []

        for img, label in zip(imgs, labels):
            for _ in range(self.count):
                _img, _label = self._lost(img, label)
                _imgs.append(_img)
                _labels.append(_label)
        
        return _imgs, _labels
    
    def _lost(self, img, label):
        h, w, _ = img.shape
        x1, y1 = int(w*random()), int(h*random())
        x2, y2 = int(x1+w*self.rate*random()), int(y1+h*self.rate*random())

        img[y1:y2, x1:x2] = 0

        # cv2.imshow('', img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        return img, label


class DistortingMirror:
    '''
        部分位置进行凸镜效果
    '''
    count = 1

    def __call__(self, imgs, labels):
        _imgs, _labels = [], []

        for img, label in zip(imgs, labels):
            for _ in range(self.count):
                h, w, _ = img.shape
                _img = self._distorting_mirror(img, int(w*random()), int(h*random()), R=random()/4, Z=random()/4)
                _imgs.append(_img)
                _labels.append(label)
        
        return _imgs, _labels

    def _distorting_mirror(self, img, x, y, R=0.1, Z=0.1):
        '''
            x, y: 坐标
            
            R: 缩放范围 (0~1)

            Z: 缩放距离 (0~1)
        '''
        img_dst = np.zeros(img.shape)
        img_dst = img.copy()

        h, w, ch = img.shape
        midX, midY = x, y
        R = Z if Z > R else R
        R = min(h, w) * R
        Z = min(h, w) * Z

        for i in range(h):
            for j in range(w):
                offsetX = j - midX
                offsetY = i - midY
                radian = atan2(offsetY, offsetX)
                radius = sqrt(offsetX**2 + offsetY**2)
                
                if 1 < radius <= R:
                    k = sqrt(radius / R) * radius / R * Z
                    X = int(cos(radian) * k) + midX
                    Y = int(sin(radian) * k) + midY
                    X = 0 if X < 0 else X
                    Y = 0 if Y < 0 else Y
                    X = w - 1 if X >= w else X
                    Y = h - 1 if Y >= h else Y

                    img_dst[i, j] = img[Y, X]   # copy
                else:
                    img_dst[i, j] = img[i, j]
        
        # cv2.imshow('', img_dst)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        return img_dst


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
        Perspective(),
        LostPart(),
        DistortingMirror(),
    ],
    save_dir=train_aug_dir,
    use_new_img=True)
    augmentation(train_compose, train_dir, train_aug_dir)

    # for test
    test_compose = Compose([
        Orign(),
        Perspective(),
        LostPart(),
        DistortingMirror(),
    ],
    save_dir=test_aug_dir)
    augmentation(test_compose, test_dir, test_aug_dir)
    print('end augmentation!')

