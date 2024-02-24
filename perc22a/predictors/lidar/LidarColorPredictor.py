

from typing import List
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.interface.PredictorInterface import Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

from perc22a.predictors.utils.transform.transform import PoseTransformations, WorldImageTransformer
from perc22a.predictors.utils.cones import Cones

class utils:
    
    def increase_brightness(img, value=30):
        import cv2
        import numpy as np
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
        import cv2
        import numpy as np
        img = image
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(32,32))
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl,a,b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced_img

    def getBrightnessDelta(image):
        import cv2
        import numpy as np
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b_img = hsv[int(len(hsv)*0.55):]
        h, s, v = cv2.split(b_img)
        # import pdb; pdb.set_trace()
        return int(125 - np.average(v).item())

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

        #TODO: STUPID SOLUTION
        # self.wi_transformer = WorldImageTransformer(340.72, 340.72, 352.99 * 0.915, 192.22 * 0.915)

        return

    def required_data(self) -> List[DataType]:
        return self.lp.required_data() + [DataType.ZED_LEFT_COLOR]

    def get_color_from_pixel(self, image, pixel_x, pixel_y, window):
        import numpy as np
        x = pixel_x
        y = pixel_y
        window = window
        average_color = np.mean(image[x:x+window, y:y+window, :], axis=(0, 1))
        print(average_color[0], average_color[1], average_color[2])

        if average_color[0] > 150 and average_color[1] < 160 and average_color[2] < 160:
            return "blue"
        if average_color[0] < 150 and average_color[1] > 130 and average_color[2] > 130:
            return "yellow"
        return "unknown"

    def predict(self, data: DataInstance) -> Cones:

        # get cone positions
        # lp_cones = self.lp.predict(data)
        # blue_arr, yellow_arr, orange_arr = lp_cones.to_numpy()

        # points = self.transformer.to_origin("zed", blue_arr, inverse=True)
        # coords = self.wi_transformer.world_to_image(points)

        yp_cones = self.lp.predict(data)
        orig_yp_cones = self.transformer.transform_cones("zed", yp_cones, inverse=True)
        blue_arr, yellow_arr, orange_arr = orig_yp_cones.to_numpy()

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
        for i in range(coords.shape[0]):
            x = int(coords[i,0])
            y = int(coords[i,1])
            window = 20
            color = self.get_color_from_pixel(img, x, y, window)
            if color == "blue":
                img[x:x+window, y:y+window, :] = [255, 0, 0, 255]
            elif color == "yellow":
                img[x:x+window, y:y+window, :] = [0, 165, 255, 255]
            else:
                img[x:x+window, y:y+window, :] = [255, 255, 255, 255]

        img = utils.increase_brightness(img, utils.getBrightnessDelta(img))
        img = utils.increaseContrast(img)

        import cv2
        cv2.imshow("color", img)
        cv2.waitKey()

        return None
    
    def display(self):
        pass
