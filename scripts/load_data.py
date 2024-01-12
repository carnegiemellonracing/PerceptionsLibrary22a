# perceptions specific imports
from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType

import perc22a.predictors.utils.lidar.visualization as vis

# general python imports
import time
import cv2


def main():
    window = vis.init_visualizer_window()
    dl = DataLoader("perc22a/data/raw/12-02-ecg-track-test")

    i = 745
    print(i)
    data = dl[i]
    img = data[DataType.ZED_LEFT_COLOR]
    pointcloud = data[DataType.HESAI_POINTCLOUD]

    vis.update_visualizer_window(None, points=pointcloud, pred_cones=[])

    


if __name__ == "__main__":
    main()
