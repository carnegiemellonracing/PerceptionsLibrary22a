

from typing import List
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.interface.PredictorInterface import Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

from perc22a.predictors.utils.transform.transform import PoseTransformations, WorldImageTransformer
from perc22a.predictors.utils.cones import Cones

import perc22a.predictors.stereo.cfg as cfg

class LidarColorPredictor(Predictor):
    '''performs lidar cone prediction, but colors via projection onto images'''

    def __init__(self):
        
        # initialize lidar predictor
        self.lp = LidarPredictor()
        self.yp = YOLOv5Predictor(camera="zed2")
        # initialize for moving between global and camera frame
        self.transformer = PoseTransformations()

        # initialize for projecting cone positions onto images
        self.wi_transformer = WorldImageTransformer(340.72, 340.72, 352.99, 192.22)
        # self.wi_transformer = WorldImageTransformer(687.14, 687.14, 676.84, 369.63)

        #TODO: STUPID SOLUTION
        # self.wi_transformer = WorldImageTransformer(340.72, 340.72, 352.99 * 0.915, 192.22 * 0.915)

        return

    def required_data(self) -> List[DataType]:
        return self.lp.required_data() + [DataType.ZED_LEFT_COLOR]

    def rectanglesOverlap(self, left1, top1, width1, height1,
                      left2, top2, width2, height2):
        return ((top1 + height1 >= top2) and (top2 + height2 >= top1) and (left1 + width1 >= left2) and (left2 + width2 >= left1))

    def get_color_from_pixel(self, boxes, centerX, centerY, window):
        color = [255, 255, 255]
        cone_left = centerX - window
        cone_right = centerX + window
        cone_top = centerY - window
        cone_bottom = centerY + window

        for box in boxes:
            print(centerX, box[0][0], box[1][0], centerY, box[0][1], box[1][1])
            box_left = box[0][0]
            box_right = box[1][0]
            box_top = box[0][1]
            box_bottom = box[1][1]
            if(self.rectanglesOverlap(cone_left, cone_top, cone_right-cone_left, cone_bottom-cone_top, box_left, box_top, box_right-box_left, box_bottom-box_top)):
                print("KASHGDKASHDOAISHDAHIJLS") 
                return box[2] 
            # if(not (cone_left > box_right or cone_right < box_left or cone_top < box_bottom or cone_bottom > box_top)): return box[2]
        return [255, 255, 255, 255]
               
        
    def predict(self, data: DataInstance) -> Cones:

        # get cone positions
        # lp_cones = self.lp.predict(data)
        # blue_arr, yellow_arr, orange_arr = lp_cones.to_numpy()

        # points = self.transformer.to_origin("zed", blue_arr, inverse=True)
        # coords = self.wi_transformer.world_to_image(points)

        lp_cones = self.lp.predict(data)
        orig_lp_cones = self.transformer.transform_cones("zed2", lp_cones, inverse=True)
        blue_arr, yellow_arr, orange_arr = orig_lp_cones.to_numpy()

        # lp_cones = self.yp.predict(data)
        # self.yp.display()

        import numpy as np
        yellow_arr = np.vstack([blue_arr, yellow_arr])

        yellow_arr = yellow_arr[:,[0, 2, 1]]        
        coords = self.wi_transformer.world_to_image(yellow_arr)
        H, W, _ = data[DataType.ZED_LEFT_COLOR].shape
        coords = coords[:, [1, 0]]
        coords[:, 0] = H - coords[:, 0]


        # plotting
        img = data[DataType.ZED_LEFT_COLOR]
        boxes = self.yp.get_bounding_boxes(data)
        img = self.yp.get_bounding_image(data)
        for i in range(coords.shape[0]):
            x = int(coords[i,0])
            y = int(coords[i,1])
            window = 5
            # color = self.get_color_from_pixel(data, x, y, window)
            # if color == "blue":
            #     img[x:x+window, y:y+window, :] = [255, 0, 0, 255]
            # elif color == "yellow":
            #     img[x:x+window, y:y+window, :] = [0, 165, 255, 255]
            # else:
            #     img[x:x+window, y:y+window, :] = [255, 255, 255, 255]
            img[x:x+window, y:y+window, :] = self.get_color_from_pixel(boxes, x, y, window)

        import cv2
        cv2.imshow("color", img)
        cv2.waitKey()

        return None
    
    def display(self):
        pass
