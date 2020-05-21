import cv2

from traffic_sign_localizer import TrafficSignLocalizer
from hog_svm import HogSvm


if __name__ == "__main__":
    img = cv2.imread('/tmp/test.jpg')

    tsl= TrafficSignLocalizer()
    hs = HogSvm('./model/svm_model.pkl')

    img_b = img.copy()
    bbox = tsl.locate(img)
    for b in bbox:
        x1 = b[0]
        y1 = b[1]
        x2 = b[0] + b[2]
        y2 = b[1] + b[3]

        classes_name, proba = hs.predict_proba(img[y1:y2, x1:x2])

        if classes_name is not 'background':
            cv2.rectangle(img_b, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_b, f'{classes_name}:{proba}', (x1, y1), 1, 1, (0, 0, 255), 2)
    
    print('end')
    cv2.imshow('', img_b)
    while True:
        if cv2.waitKey(1000) != -1:
            break
    cv2.destroyAllWindows()
