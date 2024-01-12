<<<<<<< HEAD
import cProfile  # Import cProfile for profiling
import os
import pstats  # Import pstats to format and save the profile statistics
import sys

import numpy as np

=======
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor
>>>>>>> main
from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor


def main():
    sp = StereoPredictor(
        "ultralytics/yolov5", "perc22a/predictors/stereo/model_params.pt"
    )
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(len(dl)):
        cones = sp.predict(dl[i])
        print(cones)
        sp.display()


if __name__ == "__main__":
<<<<<<< HEAD
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()  # Stop profiling
    stats = pstats.Stats(profiler).sort_stats("cumtime")  # Format the stats
    stats.dump_stats("stereo_profile_results.prof")  # Save stats to a file

    # Optional: You can print the stats to the console as well
    # stats.print_stats()

#! run 'snakeviz stereo_profile_results.prof' in the terminal
=======
    main()
>>>>>>> main
