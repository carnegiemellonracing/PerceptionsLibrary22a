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
    dl = DataLoader("perc22a/data/raw/12-02-ecg-track-test")
    lp = LidarPredictor()

    # Create a profiler object
    profiler = cProfile.Profile()

    for i in range(len(dl)):
        # load the i-th image from track testing run
        # profiler.enable()
        # cones = lp.predict(dl[i])
        # profiler.disable()
        # profiler.print_stats()
        start = time.time()
        cones, profiler = lp.profile_predict(dl[i])
        end = time.time()
        print(f"Predict Time Elapsed: {end-start}")
        # profiler.print_stats()
        print(cones)
        lp.display()

    # start = time.time()
    # cones, profiler = lp.profile_predict(dl[750])
    # end = time.time()
    # print(f"Predict Time Elapsed: {end-start}")
    # # profiler.print_stats()
    # print(cones)
    # lp.display()
    # time.sleep(10000)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.dump_stats("profile_results.prof")  # Save to a file

#! have to run "snakeviz profile_results.prof" in the terminal to view the results
