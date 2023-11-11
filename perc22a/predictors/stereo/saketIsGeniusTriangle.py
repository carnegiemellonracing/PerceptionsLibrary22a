import cv2
import numpy as np
import time
import os
import csv


cones = list()

def triangleOptimization(cone_contours, image, coneRange, shownCounter, counter, color, depth_img):
    while shownCounter < coneRange:
        cnt = cone_contours[counter]

        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        if len(approx) >= 2.95 or len(approx) <= 3.05:
            shownCounter += 1
            counter += 1

            drawColor = (0, 0, 0)

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                counter += 1
                shownCounter += 1
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            depth = depth_img[cY][cX]

            if color == "blue":
                showColor = (255, 0, 0)
            elif color == "yellow":
                showColor = (0, 255, 255)
            elif color == "orange":
                showColor = (0, 165, 255)
            
            if not (depth == float("inf") or depth ==float("-inf") or np.isnan(depth)):
                if depth < 20:
                    image = cv2.drawContours(image, [cnt], -1, showColor, 20)
                    cone = [cX, cY, depth, color]
                    cones.append(cone)

                    cv2.putText(image, str(depth), (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    print("Center: " + str(cX) + ", " + str(cY) + ", " + str(depth))
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

def getCones(actual_mask, image, color, depth_img):
    actual_mask = cv2.blur(actual_mask, (1, 1), 0)


    # cv2.imshow("blueee", img_hsv)
    # cv2.waitKey(0)

    cone_contours, _ = cv2.findContours(actual_mask,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    start = time.time_ns()
    # cone_contours_yellow = sorted(cone_contours_yellow, key = cv2.contourArea, reverse=True)
    cone_contours = sorted(cone_contours, key = cv2.contourArea, reverse=True)
    print("Sorting Contours: " + str((time.time_ns() - start) / 1000000))
    start = time.time_ns()
    # if contours have been detected, draw them
    coneRange = 0
    if len(cone_contours) >= 20: coneRange = 20
    if len(cone_contours) < 20: coneRange = len(cone_contours)
    shownCounter = 0
    counter = 0
    cv2.drawContours(image, cone_contours, -1, 255, 2)
    triangleOptimization(cone_contours, image, coneRange, shownCounter, counter, color, depth_img)

    # import pdb; pdb.set_trace()

    print("Drawing Bounding Boxes: " + str((time.time_ns() - start) / 1000000))

def detectCones(image, depth_img):
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
    
    getCones(blue_mask, image, "blue", depth_img)
    getCones(yellow_mask, image, "yellow", depth_img)
    getCones(orange_mask, image, "orange", depth_img)

    start = time.time_ns()

    cv2.imshow("Saket Is Genius?????????????????????", image)
    cv2.waitKey(0)
    # cv2.imshow("blue??", actual_mask)
    # cv2.waitKey(0)

    print("Showing Image: " + str((time.time_ns() - start) / 1000000))
    start = time.time_ns()
    print("Final Image: " + str((time.time_ns() - overallStart) / 1000000))
    add_cones_to_file(cones)

def add_cones_to_file(cones):
    with open('output.csv', mode='a') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for cone in cones:
            output_writer.writerow(cone)
    

for i in range(23, 155):
    array = np.load(f"track-testing-09-29/instance-{i}.npz")
    left_img = array['left_color'] 
    depth_img = array['depth_image']

# for i in range(30):
#     array = np.load(f"np-data/instance-{i}.npz")
#     left_img = array['left']

# for i in range(30):
#     array = np.load(f"9-30/instance-{i}.npz")
#     left_img = array['left']

# for i in range(82):
#     array = np.load(f"arthur/instance-{i}.npz")
#     left_img = array['left_color']

# directory = "FSOCO-MIT/test/images"
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg" or ".npz"):
#         left_img = cv2.imread(os.path.join(directory, filename))
    detectCones(left_img, depth_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()