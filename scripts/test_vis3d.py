from perc22a.predictors.utils.vis.Vis3D import Vis3D

from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType

from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor

import open3d as o3d
import numpy as np
import time

def demo():
    # create visualizer and window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=480, width=640)

    # initialize pointcloud instance.
    pcd = o3d.geometry.PointCloud()
    # *optionally* add initial points
    points = np.random.rand(10000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)

    # include it in the visualizer before non-blocking visualization.
    vis.add_geometry(pcd)

    # to add new points each dt secs.
    dt = 0.01
    # number of points that will be added
    n_new = 10000

    previous_t = time.time()

    # run non-blocking visualization. 
    # To exit, press 'q' or click the 'x' of the window.
    keep_running = True
    while keep_running:
        
        if time.time() - previous_t > dt:
            # Options (uncomment each to try them out):
            # 1) extend with ndarrays.
            # pcd.points.clear()
            # pcd.points.extend(np.random.rand(n_new, 3) * 1)
            pcd.points = o3d.utility.Vector3dVector(np.random.rand(10000, 3) * 4)
            
            # 2) extend with Vector3dVector instances.
            # pcd.points.extend(
            #     o3d.utility.Vector3dVector(np.random.rand(n_new, 3)))
            
            # 3) other iterables, e.g
            # pcd.points.extend(np.random.rand(n_new, 3).tolist())
            
            vis.update_geometry(pcd)
            previous_t = time.time()

        keep_running = vis.poll_events()
        vis.update_renderer()

    vis.destroy_window()

def main():

    dl = DataLoader("perc22a/data/raw/track-testing-09-29")
    yp = YOLOv5Predictor(camera="zed")
    vis = Vis3D()

    while True:
        for i in range(50, len(dl)):
            data = dl[i]
            points = data[DataType.HESAI_POINTCLOUD]

            points = points[:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] = -points[:, 0]

            cones = yp.predict(data)

            vis.set_points(points)
            vis.set_cones(cones)
            vis.update()


if __name__ == "__main__":
    main()