from PredictorInterface import Predictor
from ..utils import stereo as utils

import torch
import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt


class StereoPredictor(Predictor):
    def __init__(self, repo, path, sim=False):
        self.model = torch.hub.load(repo, 'custom', path=path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.predictions = []
        self.boxes_with_depth = []
        

    def predict(self, data):
        self.left_img = data['left']
        self.zed_pts = data['point']

        print(np.mean(self.left_img), np.std(self.left_img))

        pad = 5

        blue_cones, yellow_cones, orange_cones, self.predictions = [], [], [], []
<<<<<<< HEAD
        boxes_with_depth = []
=======
        self.boxes_with_depth = []
>>>>>>> ff3fdde (initial commit)

        # model expects RGB, convert BGR to RGB
        boxes = self.model(self.left_img[:,:,[2,1,0]], size=640)
        pred_color_dict = boxes.names

        nr, nc = self.left_img.shape[:2]

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

            coords = self.zed_pts[xl:xr, zt:zb]
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
                color = cfg_perceptions.COLORS.YELLOW
            elif color_str == "blue_cone":
                color = cfg_perceptions.COLORS.BLUE
            elif color_str == "orange_cone" or color_str == "large_orange_cone":
                color = cfg_perceptions.COLORS.ORANGE
            else:
                print("stereo-vision YOLO: Found unknown cone -- ignoring")
                color = cfg_perceptions.COLORS.UNKNOWN

            color = utils.get_cone_color(self.left_img, box, padding=2)

            prediction = [world_x,
                        world_y,
                        world_z,
                        color]
            
            self.boxes_with_depth.append(box)
            self.predictions.append(prediction)
            self.display()

            if color == cfg_perceptions.COLORS.YELLOW:
                yellow_cones.append(prediction)
            elif color == cfg_perceptions.COLORS.BLUE:
                blue_cones.append(prediction)
            elif color == cfg_perceptions.COLORS.ORANGE:
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