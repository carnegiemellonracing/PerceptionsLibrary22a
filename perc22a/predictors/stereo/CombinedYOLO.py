import cv2
import numpy as np

from typing import List
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor 
from perc22a.data.utils.DataType import DataType
from perc22a.predictors.utils.cones import Cones
import perc22a.predictors.utils.stereo as utils
import perc22a.predictors.stereo.cfg as cfg

DEBUG = False


CV2_COLORS = {
    cfg.COLORS.BLUE: [255, 191, 0],
    cfg.COLORS.YELLOW: [7, 238, 255],
    cfg.COLORS.ORANGE: [0, 150, 255],
}

class CombinedYolo:
    def init(self):
        self.YOLOModel = YOLOv5Predictor()
        self.combinedImage = None

        self.stereoCameraImage_01 = None
        self.stereoCameraImage_02 = None

        self.depthImage_01 = None
        self.depthImage_02 = None

        self.cones = Cones()

    def required_data(self) -> List[DataType]:
        return [DataType.ZED_LEFT_COLOR, DataType.ZED_LEFT_COLOR, DataType.ZED_XYZ_IMG, DataType.ZED_XYZ_IMG]
    
    def predict(self, data):
        self.stereoCameraImage_01 = data[DataType.ZED_LEFT_COLOR]
        self.stereoCameraImage_02 = data[DataType.ZED_RIGHT_COLOR]

        self.stereoCameraImage_01 = utils.increase_brightness(self.stereoCameraImage_01, utils.getBrightnessDelta(self.stereoCameraImage_01))
        self.stereoCameraImage_01 = utils.increaseContrast(self.stereoCameraImage_01)

        self.stereoCameraImage_02 = utils.increase_brightness(self.stereoCameraImage_02, utils.getBrightnessDelta(self.stereoCameraImage_02))
        self.stereoCameraImage_02 = utils.increaseContrast(self.stereoCameraImage_02)

        self.depthImage_01 = data[DataType.ZED_DEPTH_01]
        self.depthImage_02 = data[DataType.ZED_DEPTH_02]

        self.combinedImage = np.hstack((self.stereoCameraImage_01, np.zeros((self.stereoCameraImage_01.shape[0], 30, 3), dtype=np.uint8), self.stereoCameraImage_02))

        pad = 5

        # Resets predictions arr and boxes w/ depth arr for visualization
        self.predictions, self.boxes_with_depth = [], []

        # model expects RGB, convert BGR to RGB
        boxes = self.model(self.combinedImage[:, :, [2, 1, 0]], size=640)
        pred_color_dict = boxes.names

        nr, nc = self.combinedImage.shape[:2]

        num_cones = len(boxes.xyxy[0])
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

            if center_x < self.stereoCameraImage_01.shape[1]:
                coords = self.depthImage_01[xl:xr, zt:zb]
            else:
                xl = max(0, center_x - pad) - self.stereoCameraImage_01.shape[1] - 30
                xr = min(center_x + pad, nr) - self.stereoCameraImage_01.shape[1] - 30
                coords = self.depthImage_02[xl:xr, zt:zb]
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
                self.cones.add_yellow_cone(x, y, z)
            elif c == cfg.COLORS.BLUE:
                self.cones.add_blue_cone(x, y, z)
            elif c == cfg.COLORS.ORANGE:
                self.cones.add_orange_cone(x, y, z)

        return self.cones

    def display(self):
        image = self.combinedImage.copy()
        for i, box in enumerate(self.boxes_with_depth):
            _, _, depth_z, color = self.predictions[i]

            top_left = (int(box[0]), int(box[1]))
            bottom_right = (int(box[2]), int(box[3]))
            c = CV2_COLORS[color]
            image = cv2.rectangle(image.copy(), top_left, bottom_right, c, 3)
        cv2.imshow("yolov5 predictions", image)
        cv2.waitKey(1 if not DEBUG else 0)
        return
