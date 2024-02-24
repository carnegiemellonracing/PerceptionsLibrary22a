from typing import List

# input and output datatypes for data and cones respectively
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform.transform import PoseTransformations

from perc22a.predictors.interface.PredictorInterface import Predictor
import perc22a.predictors.utils.stereo as utils
import perc22a.predictors.stereo.cfg as cfg

import os
import torch
import cv2
import numpy as np
from enum import Enum

# get for allowing access to parameter files associated with predictor
STEREO_DIR_NAME = os.path.dirname(__file__)

# Hardcoded config info for now, potentially will pull from some constants.py in the future
class CFG_COLORS(Enum):
    BLUE = 1
    YELLOW = 2
    ORANGE = 3
    UNKNOWN = 4

ZED_STR = "zed"
ZED2_STR = "zed2"

CAMERA_TO_DATATYPE = {
    ZED_STR: [DataType.ZED_LEFT_COLOR, DataType.ZED_XYZ_IMG],
    ZED2_STR: [DataType.ZED2_LEFT_COLOR, DataType.ZED2_XYZ_IMG]
}

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

    def __init__(self, param_file="yolov5_model_params.pt", camera=ZED_STR):
        ''' Prediction using YOLOv5 cone detection and ZED stereocamera depth

        Arguments:
            param_file (str): parameter file to load YOLOv5 model
                - "yolov5_model_params.pt"

            camera (str): which camera stream to perform prediction on
                - "zed"
                - "zed2"
        '''
        self.sensor_name = camera
        self.transformer = PoseTransformations()


        # Initializes pytorch model using given path and repository
        self.param_file = param_file
        self.repo = "ultralytics/yolov5"
        self.path = os.path.join(STEREO_DIR_NAME, self.param_file)

        self.model = torch.hub.load(self.repo, "custom", path=self.path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        # get camera datatypes
        self.camera = camera
        self.img_datatype = CAMERA_TO_DATATYPE[self.camera][0]
        self.xyz_datatype = CAMERA_TO_DATATYPE[self.camera][1]

        # Used for visualization in display()
        self.predictions = []
        self.boxes_with_depth = []

    def required_data(self) -> List[DataType]:
        return [self.img_datatype, self.xyz_datatype]

    def predict(self, data: DataInstance) -> Cones:

        # initialize return type for cones
        cones = Cones()

        #access left_img and zed_pts from data dict(just hardcoded for now)
        self.left_img = data[self.img_datatype]
        self.zed_pts = data[self.xyz_datatype]

        pad = 5

        # Resets predictions arr and boxes w/ depth arr for visualization
        self.predictions, self.boxes_with_depth = [], []

        # model expects RGB, convert BGR to RGB
        boxes = self.model(self.left_img[:, :, [2, 1, 0]], size=640)
        pred_color_dict = boxes.names

        nr, nc = self.left_img.shape[:2]

        num_cones = len(boxes.xyxy[0])
        if DEBUG:
            print(f"[YOLOv5Predictor] [DEBUG] {num_cones} cones detected")

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
                    print(f"\t [YOLOv5Predictor] [DEBUG] success in (cone {i} of {num_cones})")
            except Exception:
                if DEBUG:
                    print(
                        f"\t [YOLOv5Predictor] [DEBUG]  (cone {i} of {num_cones}) detected cone but no depth; throwing away"
                    )
                continue

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



        return self.transformer.transform_cones(self.sensor_name, cones)

    def display(self):
        # code for visualizePrediction() in og
        image = self.left_img.copy()
        for i, box in enumerate(self.boxes_with_depth):
            _, _, depth_z, color = self.predictions[i]

            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            c = CV2_COLORS[color]
            image = cv2.rectangle(image.copy(), top_left, bottom_right, c, 3)
        cv2.imshow(f"yolov5 predictions ({self.camera})", image)
        cv2.waitKey(1 if not DEBUG else 0)
        return
