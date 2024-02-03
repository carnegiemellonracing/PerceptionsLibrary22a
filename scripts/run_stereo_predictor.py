import sys
import os
import numpy as np
from ultralytics import YOLO
import cv2

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.data.utils.dataloader import DataLoader

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value > 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] += np.uint8(value)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increaseContrast(image):
    img = image
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10, tileGridSize=(32,32))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def main():
    #sp = YOLOv5Predictor('ultralytics/yolov8', 'perc22a/predictors/stereo/yolo8n.pt')
    dl = DataLoader("/Users/michael/Desktop/PerceptionsLibrary22a/perc22a/data/raw/track-testing-09-29")

    model = YOLO('/Users/michael/Desktop/PerceptionsLibrary22a/perc22a/predictors/stereo/best164.onnx')

    

    for i in range(len(dl)):
        data = dl[i]
        imgb = data["left_color"]
        results = model(imgb[:,:,:3])
        annotated_frame = results[0].plot()

        img = increaseContrast(imgb)
        results_contrast = model(img[:,:,:3])
        annotated_frame_contrast = results_contrast[0].plot()


        result = np.hstack((annotated_frame, annotated_frame_contrast))

        cv2.imshow("YOLOv8 Inference", result)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
