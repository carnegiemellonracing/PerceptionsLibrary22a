

from typing import List
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.interface.PredictorInterface import Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

from perc22a.predictors.utils.transform.transform import PoseTransformations, WorldImageTransformer
from perc22a.predictors.utils.cones import Cones

class LidarColorPredictor(Predictor):
    '''performs lidar cone prediction, but colors via projection onto images'''

    def __init__(self):
        
        # initialize lidar predictor
        self.lp = LidarPredictor()
        self.yp = YOLOv5Predictor()

        # initialize for moving between global and camera frame
        self.transformer = PoseTransformations()

        # initialize for projecting cone positions onto images
        self.wi_transformer = WorldImageTransformer(340.72, 340.72, 352.99, 192.22)
        
        return

    def required_data(self) -> List[DataType]:
        return self.lp.required_data() + [DataType.ZED_LEFT_COLOR]

    def predict(self, data: DataInstance) -> Cones:

        # get cone positions
        # lp_cones = self.lp.predict(data)
        # blue_arr, yellow_arr, orange_arr = lp_cones.to_numpy()

        # points = self.transformer.to_origin("zed", blue_arr, inverse=True)
        # coords = self.wi_transformer.world_to_image(points)

        yp_cones = self.lp.predict(data)
        orig_yp_cones = self.transformer.transform_cones("zed", yp_cones, inverse=True)
        blue_arr, yellow_arr, orange_arr = orig_yp_cones.to_numpy()

        import numpy as np
        yellow_arr = np.vstack([blue_arr, yellow_arr])

        yellow_arr = yellow_arr[:,[0, 2, 1]]        
        coords = self.wi_transformer.world_to_image(yellow_arr)
        H, W, _ = data[DataType.ZED_LEFT_COLOR].shape
        coords = coords[:, [1, 0]]
        coords[:, 0] = H - coords[:, 0]

        img = data[DataType.ZED_LEFT_COLOR]
        for i in range(coords.shape[0]):
            x = int(coords[i,0])
            y = int(coords[i,1])
            img[x:x+10, y:y+10, :] = [255, 0, 0, 255]

        import cv2
        cv2.imshow("color", img)
        cv2.waitKey()

        return None
    
    def display(self):
        pass