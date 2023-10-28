#take params -> make transformation class, initialize stereo_predictor and lidar_predictor -> call predicts and get two sets of cone outputs
#apply transformation to both cone outputs
#use open3d to visualize
#make new issue & branch
#what if display returned some open3d project
from perc22a.predictors.interface.PredictorInterface import Predictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor
from perc22a.predictors.utils.cones import Cones
from perc22a.predictors.utils.transform import PoseTransformations
import perc22a.predictors.utils.lidar.filter as filter


import perc22a.predictors.utils.lidar.visualization as vis
import open3d as o3d
import numpy as np


class AggregatePredictor(Predictor):
    def __init__(self, path):
        self.predictions = []
        self.transformer = PoseTransformations(path)
        self.Lidar = LidarPredictor()
        self.Stereo = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/stereo/model_params.pt')
        self.transformed_lidar = Cones()
        self.transformed_stereo = Cones()
        self.points_cluster = []
        self.all_cones = []

        self.window = vis.init_visualizer_window()

    
    def predict(self, data) -> Cones:
        lidar_cones = self.Lidar.predict(data)
        stereo_cones = self.Stereo.predict(data)
        #self.all_cones = []
        self.transformed_lidar_list = []
        self.transformed_stereo_list = []
        self.transformed_lidar = Cones()
        self.transformed_stereo = Cones()

        # self.points_cluster = self.calc_point_cluster(data)
        self.points_cluster = self.Lidar.points_cluster

        #PoseTransformations takes in numpy arr
        lidar_blue, lidar_yellow, lidar_orange = lidar_cones.get_cones()
        stereo_blue, stereo_yellow, stereo_orange = stereo_cones.get_cones()
        
        
        self.transformed_lidar_list += self.transformed_lidar.add_blue_points_cone(self.transformer.to_origin('lidar', lidar_blue, inverse=False))
        #print(self.all_cones)
        self.transformed_lidar_list += self.transformed_lidar.add_yellow_points_cone(self.transformer.to_origin('lidar', lidar_yellow, inverse=False))
        self.transformed_lidar_list += self.transformed_lidar.add_orange_points_cone(self.transformer.to_origin('lidar', lidar_orange, inverse=False))

        self.transformed_stereo_list += self.transformed_stereo.add_blue_points_cone(self.transformer.to_origin('stereo', stereo_blue, inverse=False))
        self.transformed_stereo_list += self.transformed_stereo.add_yellow_points_cone(self.transformer.to_origin('stereo', stereo_yellow, inverse=False))
        self.transformed_stereo_list += self.transformed_stereo.add_orange_points_cone(self.transformer.to_origin('stereo', stereo_orange, inverse=False))

        # print(self.all_cones)
        self.display()
        #self.display()

        return self.transformed_lidar, self.transformed_stereo
    
    def _transform_points(self, points):
            points = points[:, :3]
            points = points[:, [1, 0, 2]]
            points[:,0] = -points[:,0]

            return points

    def calc_point_cluster(self, data):
        points = self._transform_points(data["points"])

        # remove all points with nan values
        points = points[~np.any(np.isnan(points), axis=-1)]

        # perform a box range on the data
        points_ground_plane = filter.box_range(
            points, xmin=-20, xmax=20, ymin=-20, ymax=20, zmin=-1, zmax=1)
        
        # vis.update_visualizer_window(None, points=points_ground_plane)

        # perform a plane fit and remove ground points
        xbound = 10
        points_filtered_ground, _, ground_planevals = filter.plane_fit(
            points, points_ground_plane, return_mask=True, boxdim=2, height_threshold=0.1)
        
        # perform another filtering algorithm to dissect boxed-region
        points_cluster, mask_cluster = filter.box_range(
            points_filtered_ground, xmin=-xbound, xmax=xbound, ymin=-10, ymax=50, zmin=-10, zmax=100, return_mask=True)

        return points_cluster


    def display(self):
            lidar_color = [1, 0, 0]
            stereo_color = [1, 0, 1]

            n_lidar_cones = len(self.transformed_lidar_list)
            n_stereo_cones = len(self.transformed_stereo_list)

            lidar_list = np.array(self.transformed_lidar_list).reshape((n_lidar_cones, 3))
            stereo_list = np.array(self.transformed_stereo_list).reshape((n_stereo_cones, 3))

            # create color arrays
            lidar_color_arr = np.zeros((n_lidar_cones, 3))
            lidar_color_arr[:, [0, 1, 2]] = lidar_color
            stereo_color_arr = np.zeros((n_stereo_cones, 3))
            stereo_color_arr[:, [0, 1, 2]] = stereo_color
            cone_colors = np.vstack([lidar_color_arr, stereo_color_arr])

            # create cone array
            cones = np.vstack([lidar_list, stereo_list])
            
            vis.update_visualizer_window(self.window, points=self.points_cluster, pred_cones=cones, colors_cones=cone_colors)
