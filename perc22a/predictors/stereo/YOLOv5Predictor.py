from typing import List

# input and output datatypes for data and cones respectively
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType
from perc22a.predictors.utils.cones import Cones

from perc22a.predictors.interface.PredictorInterface import Predictor
import perc22a.predictors.utils.stereo as utils

import os
import torch
import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import perc22a.predictors.stereo.cfg as cfg

# get for allowing access to parameter files associated with predictor
STEREO_DIR_NAME = os.path.dirname(__file__)

# Hardcoded config info for now, potentially will pull from some constants.py in the future
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
    cfg.COLORS.BLUE: [255, 191, 0],
    cfg.COLORS.YELLOW: [7, 238, 255],
    cfg.COLORS.ORANGE: [0, 150, 255],
}

DEBUG = False


class YOLOv5Predictor(Predictor):
    # Implements Predictor interface

    def __init__(self, param_file="yolov5_model_params.pt"):
        ''' param_file = the parameter file for the predictor
        '''
        # Initializes pytorch model using given path and repository

        self.param_file = param_file
        self.repo = "ultralytics/yolov5"
        self.path = os.path.join(STEREO_DIR_NAME, self.param_file)

        self.model = torch.hub.load(self.repo, "custom", path=self.path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # Used for visualization in display()
        self.predictions = []
        self.boxes_with_depth = []

    def required_data(self) -> List[DataType]:
        return [DataType.ZED_LEFT_COLOR, DataType.ZED_XYZ_IMG]

    def predict(self, data: DataInstance) -> Cones:

        # initialize return type for cones
        cones = Cones()

        #access left_img and zed_pts from data dict(just hardcoded for now)
        self.left_img = data[DataType.ZED_LEFT_COLOR]
        self.zed_pts = data[DataType.ZED_XYZ_IMG]

        pad = 5

        # Resets predictions arr and boxes w/ depth arr for visualization
        self.predictions, self.boxes_with_depth = [], []

        # model expects RGB, convert BGR to RGB
        boxes = self.model(self.left_img[:, :, [2, 1, 0]], size=640)
        pred_color_dict = boxes.names

        nr, nc = self.left_img.shape[:2]

        num_cones = len(boxes.xyxy[0])
        for i, box in enumerate(boxes.xyxy[0]):
            # depth_y = get_object_depth(box, padding=1) # removing get_object_depth because need depth map (not in DataFrame)
            # and also not as accurate of an indicator of position as the point cloud which is just some func(depth, cp)
            # where cp is the camera parameters
            center_x, center_z = utils.calc_box_center(box)
            color_id = int(box[-1].item())

            xl = max(0, center_x - pad)
            xr = min(center_x + pad, nr)
            zt = max(0, center_z - pad)
            zb = min(center_z + pad, nc)

            # coord = zed_pts[center_x, center_z]
            # world_x, world_y, world_z = coord['x'], coord['y'], coord['z']

            coords = self.zed_pts[xl:xr, zt:zb]
            # # if zed is None:
            # #     world_xs, world_ys, world_zs = coords['x'], coords['y'], coords['z']
            # # else:
            # world_xs, world_ys, world_zs = coords[:,:,0].reshape(-1), coords[:,:,1].reshape(-1), coords[:,:,2].reshape(-1)

            # utils.get_world_coords(coords)
            try:
                world_x, world_y, world_z = utils.get_world_coords(coords)
                if DEBUG:
                    print(f"\t success in (cone {i} of {num_cones})")
            except Exception:
                if DEBUG:
                    print(
                        f"\t[PERCEPTIONS WARNING] (cone {i} of {num_cones}) detected cone but no depth; throwing away"
                    )
                break

            # use YOLO model color prediction
            color_str = pred_color_dict[color_id]
            if color_str == "yellow_cone":
                color = cfg.COLORS.YELLOW
            elif color_str == "blue_cone":
                color = cfg.COLORS.BLUE
            elif color_str == "orange_cone" or color_str == "large_orange_cone":
                color = cfg.COLORS.ORANGE
            else:
                if DEBUG:
                    print("stereo-vision YOLO: Found unknown cone -- ignoring")
                color = cfg.COLORS.UNKNOWN

            # package information into a single prediction
            prediction = [world_x, world_y, world_z, color]

            # add most recent prediction to arrays and call display to visualize
            self.boxes_with_depth.append(box)
            self.predictions.append(prediction)

            x, y, z, c = prediction
            if c == cfg.COLORS.YELLOW:
                cones.add_yellow_cone(x, y, z)
            elif c == cfg.COLORS.BLUE:
                cones.add_blue_cone(x, y, z)
            elif c == cfg.COLORS.ORANGE:
                cones.add_orange_cone(x, y, z)

        return cones

    def display(self):
        # code for visualizePrediction() in og
        image = self.left_img.copy()
        for i, box in enumerate(self.boxes_with_depth):
            _, _, depth_z, color = self.predictions[i]

            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            c = CV2_COLORS[color]
            image = cv2.rectangle(image.copy(), top_left, bottom_right, c, 3)
        cv2.imshow("left w/ predictions", image)
        cv2.waitKey(1)
        return
