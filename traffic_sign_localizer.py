import numpy as np
import cv2


class TrafficSignLocalizer:
    '''
        传入图片，获取目标的边界框。
    '''

    # _color_filter
    B_min = np.array([100, 50, 50])
    B_max = np.array([120, 255, 255])
    R_min1 = np.array([0, 50, 50])
    R_max1 = np.array([5, 255, 255])
    R_min2 = np.array([170, 50, 50])
    R_max2 = np.array([180, 255, 255])
    # _dilate_erode
    # erode:1 -> dilate:2  dilate:1 -> erode:3
    # near distance: iter_erode:3   far: iter_erode:1
    kernel_dilate = np.ones((3, 3), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)
    iter_dilate = 1
    iter_erode = 1
    # _area_filter
    rate_min_area = .0005
    rate_max_area = 1.
    # _contours_select
    wh_rate = 3.

    def __init__(self, use_erode_dilate=True):
        self.use_erode_dilate = use_erode_dilate


    def locate(self, img):
        '''
            img: (type cv2 Mat)

            return:
                bbox: [[x, y, w, h], ...]
        '''
        img = self._color_white_balance(img)
        img_bin = self._color_filter(img)
        if self.use_erode_dilate:
            img_bin = self._dilate_erode(img_bin)
        contours = self._area_filter(img_bin)
        contours = self._shape_filter(contours)
        bbox = self._contours_select(contours)

        return bbox

    def _color_white_balance(self, img):
        '''
            白平衡
        '''
        bgr = cv2.split(img)
        b = np.mean(bgr[0])
        g = np.mean(bgr[1])
        r = np.mean(bgr[2])

        kb = (b + g + r) / b / 3
        kg = (b + g + r) / g / 3
        kr = (b + g + r) / r / 3

        bgr[0] = bgr[0] * kb
        bgr[1] = bgr[1] * kg
        bgr[2] = bgr[2] * kr

        img = cv2.merge(bgr)
        img = img.clip(0, 255).astype(np.uint8)
        return img

    
    def _color_filter(self, imgBGR):
        '''
            颜色过滤
        '''
        imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)

        img_B_bin = cv2.inRange(imgHSV, self.B_min, self.B_max)
        img_R_bin1 = cv2.inRange(imgHSV, self.R_min1, self.R_max1)
        img_R_bin2 = cv2.inRange(imgHSV, self.R_min2, self.R_max2)

        img_R_bin = np.maximum(img_R_bin1, img_R_bin2)
        img_bin = np.maximum(img_B_bin, img_R_bin)

        return img_bin


    def _dilate_erode(self, img_bin):
        '''
            腐蚀膨胀
        '''
        img_bin = cv2.dilate(img_bin, self.kernel_dilate, iterations=self.iter_dilate)
        img_bin = cv2.erode(img_bin, self.kernel_erode, iterations=self.iter_erode)

        return img_bin


    def _area_filter(self, img_bin):
        '''
            面积过滤

            return:
                返回contours
        '''
        contour_pass = []

        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        h, w = img_bin.shape
        max_area = w*h*self.rate_max_area if self.rate_max_area > 0 else w*h
        min_area = w*h*self.rate_min_area if self.rate_min_area > 0 and self.rate_max_area > self.rate_min_area else w*h*0.001

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                contour_pass.append(contour)
        
        return contour_pass


    # TODO
    def _shape_filter(self, contours):
        '''
            形状过滤
        '''
        contours_pass = []
        contours_pass = contours
        return contours_pass


    def _contours_select(self, contours):
        '''
            边框选择，长宽比过滤

            return:
                返回矩形坐标 (x, y, w, h)
        '''
        bbox = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w/h <= self.wh_rate and h/w <= self.wh_rate:
                bbox.append([x, y, w, h])

        return bbox





if __name__ == "__main__":
    img = cv2.imread('/tmp/test.jpg')

    tsl = TrafficSignLocalizer()
    bbox = tsl.locate(img)

    for b in bbox:
        x, y, w, h = b
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow('final', img)

    while True:
        if cv2.waitKey(1000) == 27:
            break
    cv2.destroyAllWindows()


