from sklearn.svm import SVC
import joblib

import os
import numpy as np

from config import opt


def load_hog_feature(file_path):
    '''
        加载hog特征
    '''
    features = []
    labels = []

    hog_feature = joblib.load(file_path)

    for hf in hog_feature:
        features.append(hf['feature'])
        labels.append(hf['label'])

    return np.array(features), np.array(labels)


def train_svm(model_path, features, labels, classes_num=2):
    '''
        训练svm
    '''
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        clf = SVC(C=classes_num, probability=True, tol=1e-3)
    clf.fit(features, labels)
    joblib.dump(clf, model_path)


def test_svm(model_path, features, labels):
    '''
        测试svm
    '''
    if not os.path.exists(model_path):
        print('! no exists')
        return
    
    clf = joblib.load(model_path)
    accuracy = clf.score(features, labels)
    print(f'test accuracy: {accuracy}')
    return accuracy


def train(**kwargs):
    features, labels = load_hog_feature(opt.TRAIN_HOG_DIR + opt.HOG_FILE_NAME)
    train_svm(opt.model_path, features, labels, classes_num=opt.classes_num)


def test(**kwargs):
    features, labels = load_hog_feature(opt.TEST_HOG_DIR + opt.HOG_FILE_NAME)
    test_svm(opt.model_path, features, labels)


if __name__ == "__main__":
    print('start svm ...')
    import fire
    fire.Fire()
    print('end!')
