import numpy as np
import cv2
import base64
import requests
# from predict_client.prod_client import ProdClient
import json
import time

import tensorflow as tf

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

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
        img = cv2.rectangle(img, (int(bb[0]), int(bb[1])), (abs(int(bb[2])), int(bb[3])), (255, 0, 0), 10, 1)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test", img)
    cv2.waitKey(0)

    print("Done")
