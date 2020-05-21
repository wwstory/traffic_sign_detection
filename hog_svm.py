import cv2
import numpy as np
from skimage import feature as ft
import joblib


class HogSvm:
    '''
        传入目标图片，返回预测的类别名
    '''

    classes_names = ['background', 'left', 'right']

    def __init__(self, model_path='./svm_model.pkl'):
        self.clf = joblib.load(model_path)


    def predict(self, img):
        '''
            img: (type cv2 Mat)

            返回预测类别名称
        '''
        feature = self._get_hog_feature(img)
        return self._predict(feature)


    def predict_proba(self, img):
        '''
            img: (type cv2 Mat)
            
            返回预测的类别名称和概率
        '''
        feature = self._get_hog_feature(img)
        return self._predict_proba(feature)


    def _get_hog_feature(self, img):
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

        feature = np.reshape(feature, (1, -1))
        return feature


    def _predict(self, feature):
        p = self.clf.predict(feature)
        index = p[0]
        return self.classes_names[index]


    def _predict_proba(self, feature):
        p = self.clf.predict_proba(feature)
        p = p[0]
        index = np.argmax(p)
        proba = p[index]
        return self.classes_names[index], proba
