import cv2
import numpy as np

from typing import List
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor 
from perc22a.data.utils.DataType import DataType

class CombinedYolo:
    def init(self):
        self.YOLOModel = YOLOv5Predictor()
        self.combinedImage = None

        self.leftCameraImage = None
        self.rightCameraImage = None

        self.leftDepthImage = None
        self.rightDepthImage = None

    def required_data(self) -> List[DataType]:
        return [DataType.ZED_LEFT_COLOR, DataType.ZED_LEFT_COLOR, DataType.ZED_DEPTH_IMG]
    
    def predict(self, data):
        self.leftCameraImage = data[DataType.ZED_LEFT_COLOR]
        self.rightCameraImage = data[DataType.ZED_RIGHT_COLOR]

        self.leftCameraImage = Utils.increase_brightness(self.leftCameraImage, Utils.getBrightnessDelta(self.leftCameraImage))
        self.leftCameraImage = Utils.increaseContrast(self.leftCameraImage)

        self.rightCameraImage = Utils.increase_brightness(self.rightCameraImage, Utils.getBrightnessDelta(self.rightCameraImage))
        self.rightCameraImage = Utils.increaseContrast(self.rightCameraImage)

        self.leftDepthImage = data[DataType.ZED_LEFT_DEPTH]
        self.rightDepthImage = data[DataType.ZED_RIGHT_DEPTH]

        self.combinedImage = np.concatenate((self.leftCameraImage, self.rightCameraImage), axis=1)

        return self.YOLOModel.predict(data)

    def display(self):
        return self.YOLOModel.display()

class Utils:
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
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(32,32))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    def getBrightnessDelta(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b_img = hsv[int(len(hsv)*0.55):]
        h, s, v = cv2.split(b_img)
        # import pdb; pdb.set_trace()
        return int(125 - np.average(v).item())
