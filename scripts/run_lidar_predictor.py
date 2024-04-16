import cProfile
import pstats

# perceptions specific imports
# general python imports

from perc22a.utils.Timer import Timer
from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor


def main():
    # initialize data loader and lidar predictor
    dl = DataLoader("perc22a/data/raw/tt-4-6-lidar")
    lp = LidarPredictor()
    timer = Timer()

    # Create a profiler object
    profiler = cProfile.Profile()

    #import pdb; pdb.set_trace();
 
    for i in range(44, len(dl)):
        print(i)
        # load the i-th image from track testing run
        # profiler.enable()
        # cones = lp.predict(dl[i])
        # profiler.disable()
        # profiler.print_stats()
        timer.start("Predict Time Elapsed")
        cones, profiler = lp.profile_predict(dl[i])
        timer.end("Predict Time Elapsed")
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
