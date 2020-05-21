#!/bin/sh

cd tools

python3 split_dataset.py
python3 crop_region.py
python3 data_augmentation.py
python3 extract_hog_feature.py

cd ..

t=$(date '+%y-%m-%d_%H-%M')
cp model/svm_model.pkl model/svm_model_${t}.pkl

python3 train_svm.py train
python3 train_svm.py test

