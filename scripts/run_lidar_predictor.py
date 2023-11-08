import cProfile
import pstats

# perceptions specific imports
# general python imports
import time

import cv2

from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor


def main():
    # initialize data loader and lidar predictor
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")
    lp = LidarPredictor()

    for i in range(30):
        # load the i-th image from track testing run
        cones = lp.predict(dl[i])
        print(cones)
        lp.display()


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats("profile_results.prof")  # Save to a file

#! have to run "snakeviz profile_results.prof" in the terminal to view the results
