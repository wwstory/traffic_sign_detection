
# introduce

定位目标 -> 特征提取 -> 分类

- 定位：白平衡，HSV颜色过滤，膨胀腐蚀，面积，长宽比
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
./run.sh
```

# 扩展类别

1. 将新的类别图片放入`IMAGES/`，标注放入`ANNOS/`。
2. 向`tools/config.py`中的`classes_names`和`label_names`添加新类别。
3. 修改`config.py`中的`classes_num`，改为类别数量。
4. 向`hog_svm.py`中的`classes_names`添加新类别。

# ref

[hog][1]

[object-detection][2]

[ref github][3]

[标注工具][4]

[dataset][5]


---

[1]: https://www.learnopencv.com/histogram-of-oriented-gradients/

[2]: https://www.learnopencv.com/image-recognition-and-object-detection-part1/

[3]: https://github.com/ZhouJiaHuan/traffic-sign-detection/

[4]: https://github.com/tzutalin/labelImg

[5]: https://pan.baidu.com/s/1Q0cqJI9Dnvxkj7159Be4Sw
