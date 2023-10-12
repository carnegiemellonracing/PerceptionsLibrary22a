from PredictorInterface import Predictor
from ..utils import stereo as utils

import torch
import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

#Hardcoded config info for now, potentially will pull from some constants.py in the future
class CFG_COLORS(Enum):
    BLUE = 1
    YELLOW = 2
    ORANGE = 3
    UNKNOWN = 4

COLORS = {
    1: (255, 191, 0),
    0: (0, 150, 255),
}

CV2_COLORS = {
    CFG_COLORS.BLUE: [255, 191, 0],
    CFG_COLORS.YELLOW: [7, 238, 255],
    CFG_COLORS.ORANGE: [0, 150, 255]
}



class StereoPredictor(Predictor):
#Implements Predictor interface

    def __init__(self, repo, path, sim=False):
        #Initializes pytorch model using given path and repository

        self.model = torch.hub.load(repo, 'custom', path=path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        #Used for visualization in display()
        self.predictions = []
        self.boxes_with_depth = []
        

    def predict(self, data):
        #access left_img and zed_pts from data dict(just hardcoded for now)
        left_img = data['left_img']
        zed_pts = data['zed_pts']

        print(np.mean(left_img), np.std(left_img))

        pad = 5

        blue_cones, yellow_cones, orange_cones = [], [], [], []
        #Resets predictions arr and boxes w/ depth arr for visualization
        self.predictions, self.boxes_with_depth = [], []

        # model expects RGB, convert BGR to RGB
        boxes = self.model(left_img[:,:,[2,1,0]], size=640)
        pred_color_dict = boxes.names

        nr, nc = left_img.shape[:2]

        print("performing prediction")
        num_cones = len(boxes.xyxy[0])
        for i, box in enumerate(boxes.xyxy[0]):
            #depth_y = get_object_depth(box, padding=1) # removing get_object_depth because need depth map (not in DataFrame)
            # and also not as accurate of an indicator of position as the point cloud which is just some func(depth, cp)
            # where cp is the camera parameters
            center_x, center_z = utils.calc_box_center(box)
            color_id = int(box[-1].item())

            xl = max(0, center_x - pad)
            xr = min(center_x + pad, nr)
            zt = max(0, center_z - pad)
            zb = min(center_z + pad, nc)

            #coord = zed_pts[center_x, center_z]
            #world_x, world_y, world_z = coord['x'], coord['y'], coord['z']

            coords = zed_pts[xl:xr, zt:zb]
            # # if zed is None:
            # #     world_xs, world_ys, world_zs = coords['x'], coords['y'], coords['z']
            # # else:
            # world_xs, world_ys, world_zs = coords[:,:,0].reshape(-1), coords[:,:,1].reshape(-1), coords[:,:,2].reshape(-1)

            try:
                world_x, world_y, world_z = utils.get_world_coords(coords)
            except Exception:
                print(f"\t[PERCEPTIONS WARNING] (cone {i} of {num_cones}) detected cone but no depth; throwing away")
                break

            color_str = pred_color_dict[color_id]
            if color_str == "yellow_cone":
                color = CFG_COLORS.YELLOW
            elif color_str == "blue_cone":
                color = CFG_COLORS.BLUE
            elif color_str == "orange_cone" or color_str == "large_orange_cone":
                color = CFG_COLORS.ORANGE
            else:
                print("stereo-vision YOLO: Found unknown cone -- ignoring")
                color = COLORS.UNKNOWN

            color = utils.get_cone_color(left_img, box, padding=2)

            prediction = [world_x,
                        world_y,
                        world_z,
                        color]
            
            #add most recent prediction to arrays and call display to visualize
            self.boxes_with_depth.append(box)
            self.predictions.append(prediction)
            self.display()

            if color == CFG_COLORS.YELLOW:
                yellow_cones.append(prediction)
            elif color == CFG_COLORS.BLUE:
                blue_cones.append(prediction)
            elif color == CFG_COLORS.ORANGE:
                orange_cones.append(prediction)

        return orange_cones, blue_cones, yellow_cones



    def display(self):
        #code for visualizePrediction() in og
        for i, box in enumerate(self.boxes_with_depth):
            _, _, depth_z, color = self.predictions[i]
            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            c = CV2_COLORS[color]
            image = cv2.rectangle(self.left_image.copy(), top_left, bottom_right, c, 3)
            cv2.imshow('left w/ predictions', image)
            cv2.waitKey(1)
        return