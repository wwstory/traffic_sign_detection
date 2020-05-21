import warnings

class Config:
    # classes
    classes_names = ['background', 'left', 'right']
    label_names = {'background':0, 'left': 1, 'right': 2}

    # dir path
    # split_dataset
    IMAGE_NAME = 'IMAGES'
    ANNO_NAME = 'ANNOS'
    HOME_DIR = '/home/nevc/'
    DATASET_DIR = HOME_DIR + 'datasets/'
    DATASET_ORGIN_DIR = DATASET_DIR + 'data/'
    TRAIN_DATASETS_DIR = DATASET_DIR + 'train/'
    TEST_DATASETS_DIR = DATASET_DIR + 'test/'
    # crop_region
    TRAIN_REGION_DIR = TRAIN_DATASETS_DIR + 'region/'
    TEST_REGION_DIR = TEST_DATASETS_DIR + 'region/'
    filter_list = []
    random_crop_background = True
    # data_augment
    TRAIN_AUG_DIR = TRAIN_DATASETS_DIR + 'augment/'
    TEST_AUG_DIR = TEST_DATASETS_DIR + 'augment/'
    # hog_extract
    TRAIN_HOG_DIR = TRAIN_DATASETS_DIR + 'hog/'
    TEST_HOG_DIR = TEST_DATASETS_DIR + 'hog/'
    HOG_FILE_NAME = 'hog_feature.pkl'
    


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
