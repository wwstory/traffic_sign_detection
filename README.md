
# introduce

定位目标 -> 特征提取 -> 分类

- 定位：使用颜色，白平衡，膨胀腐蚀，面积，长宽比
- 特征提取：HOG
- 分类：SVM

# run

1.修改`tools/config.py`中的`HOME_DIR`为自己的home目录。

2.下载数据集放在指定的目录。（图片放于：`/home/<username>/datasets/data/IMAGES/`， 标签放于：`/home/<username>/datasets/data/ANNOS/`）

3.将测试图片放于`/tmp/test.jpg`。

4.运行
```sh
# train
./train.sh

# test
./run
```

# 手动执行的步骤

split_dataset.py

crop_region.py

data_augmentation.py

extract_hog_feature.py

train_svm.py

test_image.py

# ref

[hog][1]

[object-detection][2]

[orgin github][3]

[标注工具][4]

[dataset][5]


---

[1]: https://www.learnopencv.com/histogram-of-oriented-gradients/

[2]: https://www.learnopencv.com/image-recognition-and-object-detection-part1/

[3]: https://github.com/ZhouJiaHuan/traffic-sign-detection/

[4]: https://github.com/tzutalin/labelImg

[5]: https://pan.baidu.com/s/1Q0cqJI9Dnvxkj7159Be4Sw
