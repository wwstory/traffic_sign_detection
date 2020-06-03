#!/bin/sh

cd tools

python3 split_dataset.py
python3 region_proposal.py
python3 data_augmentation.py
python3 extract_hog_feature.py

cd ..

if [ -f 'model/svm_model.pkl' ]; then
    t=$(date '+%y-%m-%d_%H-%M')
    cp model/svm_model.pkl model/svm_model_${t}.pkl
fi

python3 train_svm.py train
python3 train_svm.py test

