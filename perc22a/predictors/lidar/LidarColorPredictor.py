

from typing import List
from perc22a.data.utils.DataInstance import DataInstance
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.interface.PredictorInterface import Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

from perc22a.predictors.utils.transform.transform import PoseTransformations, WorldImageTransformer
from perc22a.predictors.utils.cones import Cones

class LidarColorPredictor(Predictor):
    '''performs lidar cone prediction, but colors via projection onto images'''

    def __init__(self):
        
        # initialize lidar predictor
        self.lp = LidarPredictor()

        # initialize for moving between global and camera frame
        self.transformer = PoseTransformations()

        # initialize for projecting cone positions onto images
        self.wi_transformer = WorldImageTransformer(687.14, 687.14, 676.74, 369.63)
        
        return

    def required_data(self) -> List[DataType]:
        return self.lp.required_data() + [DataType.ZED_LEFT_COLOR]

    def predict(self, data: DataInstance) -> Cones:

        # get cone positions
        lp_cones = self.lp.predict(data)
        blue_arr, yellow_arr, orange_arr = lp_cones.to_numpy()

        points = self.transformer.to_origin("zed", blue_arr, inverse=True)
        coords = self.wi_transformer.world_to_image(points)

        import pdb; pdb.set_trace()


        return lp_cones
    
    def display():
        pass