
# allows us to simply import via `from perc22a.predictors import *Predictor`
# add a line here for any new predictor created
from perc22a.predictors.interface.PredictorInterface import Predictor 
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor