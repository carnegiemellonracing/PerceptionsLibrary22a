import cProfile
import io
import pstats

# perceptions specific imports
from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor

# general python imports
>>>>>>> 95d76f1972d95e8a237583562bc67835e90c3ede
import time
import cv2
from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors import LidarPredictor


def main():
    # initialize data loader and lidar predictor
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")
    lp = LidarPredictor()
    # Initialize a StringIO object to collect profiling stats
    s = io.StringIO()
    for i in range(len(dl)):
        # load the i-th image from track testing run
        cones, profiler = lp.profile_predict(dl[i])  # Corrected to expect a tuple
        print(cones)
        lp.display()
        # Collect the profiling information
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats()
    # Optionally, print the profiling information or save it to a file
    profiling_info = s.getvalue()
    print(profiling_info)
    with open("profile_stats.txt", "w") as f:
        f.write(profiling_info)
if __name__ == "__main__":
    main()











