# perceptions specific imports
from data.utils.dataloader import DataLoader
from predictors.utils.lidar import *

# general python imports
import time
import cv2


def main():
    dl = DataLoader("data/raw/track-testing-09-29")

    window = init_visualizer_window()

    for i in range(len(dl)):
        # load the i-th image from track testing run
        data = dl[i]
        points = data["points"][:, :3]
        points = points[:, [1,0,2]]
        points[:,0] = -points[:,0]

        # display the point cloud
        update_visualizer_window(window, points)

if __name__ == "__main__":
    main()