import cv2
import numpy as np
import time
from perc22a.predictors.utils.cones import Cones

class ThresholdPredictor:
    def __init__(self, data):
        self.left_img = data['left_color']
        self.depth_img = data['depth_image']
        self.cones = Cones()

    def predict(self):
        detectCones(self, self.left_img, self.depth_img)
        return self.cones
    
    def visualize(self, image):
        for cone in self.cones:
            cv2.circle(image, (cone[0], cone[1]), 5, (0, 255, 0), -1)
        return image


def triangleOptimization(self, cone_contours, coneRange, shownCounter, counter, color, depth_img):
    while shownCounter < coneRange:
        cnt = cone_contours[counter]

        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) >= 2.95 or len(approx) <= 3.05:
            shownCounter += 1
            counter += 1

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                counter += 1
                shownCounter += 1
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            depth = depth_img[cY][cX]
            
            if not (depth == float("inf") or depth ==float("-inf") or np.isnan(depth)):
                if depth < 20:
                    cone = [cX, cY, depth, color]
                    self.cones.append(cone)
        if counter == len(cone_contours):
            break

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


def detectCones(self, image, depth_img):
    overallStart = time.time_ns()
    start = time.time_ns()

    mask = np.zeros(image.shape[:2], dtype="uint8")
    # cv2.rectangle(mask, (0, 0), (image.shape[1], int(image.shape[0]*0.88)), 255, -1)
    mask[-120:, 470:930] = 255
    mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=mask)


    image = increase_brightness(image, getBrightnessDelta(image))
    image = increaseContrast(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b_img = img_hsv[int(len(img_hsv)*0.55):]
    
    print(type(b_img))
    print("Convert Image: " + str((time.time_ns() - start) / 1000000))
    start = time.time_ns()
    blue_mask = cv2.inRange(b_img, (80,90,50), (115, 255, 255))
    orange_mask = cv2.inRange(b_img, (0,100,100), (10, 255, 255))
    black_mask = cv2.inRange(img_hsv, (179, 255, 255), (179, 255, 255))
    yellow_mask = cv2.inRange(b_img, (10,100,100), (50, 255, 255))

    blue_mask = cv2.vconcat([black_mask[:int(len(black_mask)*0.55)], blue_mask])
    yellow_mask = cv2.vconcat([black_mask[:int(len(black_mask)*0.55)], yellow_mask])
    orange_mask = cv2.vconcat([black_mask[:int(len(black_mask)*0.55)], orange_mask])

    actual_mask = black_mask + yellow_mask + blue_mask + orange_mask


    getCones(self, blue_mask, image, "blue", depth_img)
    getCones(self, yellow_mask, image, "yellow", depth_img)
    getCones(self, orange_mask, image, "orange", depth_img)

    start = time.time_ns()
    # cv2.imshow("blue??", actual_mask)
    # cv2.waitKey(0)

    print("Showing Image: " + str((time.time_ns() - start) / 1000000))
    start = time.time_ns()
    print("Final Image: " + str((time.time_ns() - overallStart) / 1000000))

    return Cones()