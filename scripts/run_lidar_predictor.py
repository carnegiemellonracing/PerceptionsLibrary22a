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
    lp = LidarPredictor(debug=True)
    timer = Timer()

    # Create a profiler object
    profiler = cProfile.Profile()

    #import pdb; pdb.set_trace();
 
    while True:
        # originally 44
        for i in range(103, 235):
            # load the i-th image from track testing run
            print(i)
            timer.start("Predict Time Elapsed")
            cones = lp.predict(dl[i])
            timer.end("Predict Time Elapsed")
            # profiler.print_stats()
            # print(cones)
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
    main()

#! have to run "snakeviz profile_results.prof" in the terminal to view the results
