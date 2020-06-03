import os
import shutil
import cv2
import xml.etree.ElementTree as ET
from random import randint

from config import opt


def region_proposal(in_dir, save_dir, filter_list=[], random_crop_background=True):
    img_dir = os.path.join(in_dir, opt.IMAGE_NAME)
    ann_dir = os.path.join(in_dir, opt.ANNO_NAME)
    anns_name = os.listdir(ann_dir)
    anns_path = [os.path.join(ann_dir, name) for name in anns_name]


    for i, ann_path in enumerate(anns_path):
        obj_dict = parse_xml(ann_path)

        img_path = os.path.join(img_dir, obj_dict['filename'])
        img = None
        for j, obj in enumerate(obj_dict['object']):
            img = cv2.imread(img_path)

            obj_name = obj[0]

            if obj_name in filter_list: # 过滤
                continue

            img_name = obj_name + '_' + str(i+j) + '.jpg'

            img_crop = _crop_image(img, obj)

            # cv2.imshow('', img_crop)
            # cv2.waitKey(500)
            # cv2.destroyAllWindows()

            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img_crop)
        
        if random_crop_background and img is not None:
            background_obj = []
            background_obj = generate_random_background(img, obj_dict['object'])
            if background_obj is None:
                continue
            img_crop = _crop_image(img, background_obj)
            img_name = 'background' + '_b' + str(i) + '.jpg'
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, img_crop)


def parse_xml(xml_path):
    '''
        获取xml中的标签名称，类别

        (一个图片中也许有多个标签)
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()

    obj_dict = {'filename':'', 'object':[]}

    for item in root:
        if item.tag == 'filename':
            obj_dict['filename'] = item.text
        elif item.tag == 'object':
            name = item[0].text
            xmin = int(item[4][0].text)
            ymin = int(item[4][1].text)
            xmax = int(item[4][2].text)
            ymax = int(item[4][3].text)
            obj_dict['object'].append([name, xmin, ymin, xmax, ymax])

    return obj_dict
            


def _crop_image(img, obj):
    '''
        裁剪图像的目标区域

        obj = [name, xmin, ymin, xmax, ymax]
    '''
    xmin = obj[1]
    ymin = obj[2]
    xmax = obj[3]
    ymax = obj[4]
    img_crop = img[ymin:ymax, xmin:xmax]
    return img_crop


def generate_random_background(img, objs, wh_rate=2, count=3, background_name='background'):
    '''
        随机裁剪与标签重合度小于30%的作为背景。

        objs = [[name, xmin, ymin, xmax, ymax], ...]
        （为避免整张图都是标签，导致一直随机不到合适的坐标，仅尝试3次）
    '''
    h, w, _ = img.shape
    
    for _ in range(count):
        x1 = randint(0, w)
        y1 = randint(0, h)
        for obj in objs:
            xmin = obj[1]
            ymin = obj[2]
            xmax = obj[3]
            ymax = obj[4]
            area = (xmax - xmin) * (ymax - ymin)

            x2 = x1 + (xmax - xmin)
            x2 = w if x2 > w else x2
            y2 = y1 + (ymax - ymin)
            y2 = h if y2 > h else y2

            _w = x2 - x1
            _h = y2 - y1

            if _w == 0 or _h == 0:
                continue
            elif _w / _h > wh_rate or _h / _w > wh_rate:  # 长宽比不合适
                continue

            iou = iou_calculate(obj[1:], [x1, y1, x2, y2])

            if iou > 0.3:   # iou不合适
                continue
            else:
                return [background_name, x1, y1, x2, y2]
        

def iou_calculate(rect1, rect2):
    '''
        计算iou
    '''
    gxmin, gymin, gxmax, gymax = rect1
    pxmin, pymin, pxmax, pymax = rect2

    xmin = max(gxmin, pxmin)
    ymin = max(gymin, pymin)
    xmax = min(gxmax, pxmax)
    ymax = min(gymax, pymax)

    S_cross = (xmax - xmin) * (ymax - ymin)
    S1 = (gxmax - gxmin) * (gymax - gymin)
    S2 = (pxmax - pxmin) * (pymax - pymin)

    iou = (S_cross) / (S1 + S2 - S_cross)

    return iou


if __name__ == "__main__":
    train_dir = opt.TRAIN_DATASETS_DIR
    train_save_dir = opt.TRAIN_REGION_DIR
    test_dir = opt.TEST_DATASETS_DIR
    test_save_dir = opt.TEST_REGION_DIR
    filter_list = opt.filter_list
    random_crop_background = opt.random_crop_background

    if os.path.exists(train_save_dir):
        shutil.rmtree(train_save_dir)
    if os.path.exists(test_save_dir):
        shutil.rmtree(test_save_dir)
    os.mkdir(train_save_dir)
    os.mkdir(test_save_dir)
    
    print('start crop ...')
    # for train
    region_proposal(train_dir, train_save_dir, filter_list, random_crop_background)
    # for test
    region_proposal(test_dir, test_save_dir, filter_list, random_crop_background)
    print('end crop!')
