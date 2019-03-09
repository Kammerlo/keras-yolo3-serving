import numpy as np
import cv2
import base64
import requests
# from predict_client.prod_client import ProdClient
import json
import time

import tensorflow as tf

from utils.colors import get_color

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def draw_boxes(image,bb,confidence,classnum):
    # num = int(classnum)
    xmin = bb[1]
    ymin = bb[0]
    xmax = bb[3]
    ymax = bb[2]
    label_str = (labels[classnum] + ' ' + str(round(confidence*100, 2)) + '%')
    text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
    width, height = text_size[0][0], text_size[0][1]
    region = np.array([[xmin-3,        ymin],
                       [xmin-3,        ymin-height-26],
                       [xmin+width+13, ymin-height-26],
                       [xmin+width+13, ymin]], dtype='int32')

    cv2.rectangle(img=image, pt1=(xmin,ymin), pt2=(xmax,ymax), color=get_color(classnum), thickness=5)
    cv2.fillPoly(img=image, pts=[region], color=get_color(classnum))
    cv2.putText(img=image,
                text=label_str,
                org=(xmin+13, ymin - 13),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1e-3 * image.shape[0],
                color=(0,0,0),
                thickness=2)

    return image


if __name__ == '__main__':
    img = cv2.imread("dog.jpg")
    image_h, image_w = img.shape[:2]

    retr, buffer = cv2.imencode(".jpg", img)
    jpg_str = base64.urlsafe_b64encode(buffer).decode("utf-8")

    payload = {
        "instances": [{'image': jpg_str}]
    }

    # sending post request to TensorFlow Serving server
    startTime = time.time()
    r = requests.post('http://127.0.0.1:8501/v1/models/voc:predict', json=payload)
    endTime = time.time()
    print(endTime - startTime)
    print(r)
    pred = json.loads(r.content.decode('utf-8'))
    for preds in pred['predictions']:
        bb = preds['box']
        confidence = preds['confidence']
        classes = preds['class']
        img = draw_boxes(img,bb,confidence,classes)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", img)
    cv2.waitKey(0)

    print("Done")
