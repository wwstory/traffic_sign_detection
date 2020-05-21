import warnings

class Config:
    # classes
    classes_num = 3

    # control
    is_train = True

    # dir path
    HOME_DIR = '/home/nevc/'
    DATASET_DIR = HOME_DIR + 'datasets/'
    TRAIN_DATASETS_DIR = DATASET_DIR + 'train/'
    TEST_DATASETS_DIR = DATASET_DIR + 'test/'
    # hog_extract
    TRAIN_HOG_DIR = TRAIN_DATASETS_DIR + 'hog/'
    TEST_HOG_DIR = TEST_DATASETS_DIR + 'hog/'
    HOG_FILE_NAME = 'hog_feature.pkl'
    # train_svm
    model_path = './model/svm_model.pkl'


    def parse(self, **kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = Config()
