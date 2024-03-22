import time

from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors.lidar.LidarColorPredictor import LidarColorPredictor


def main():
    # initialize data loader and lidar predictor
    dl = DataLoader("perc22a/data/raw/three-laps-reverse-subset")
    lp = LidarColorPredictor()


    for i in range(500):
        # load the i-th image from track testing run
        start = time.time()
        cones = lp.predict(dl[i])
        end = time.time()
        print(f"Predict Time Elapsed: {end-start}")

        lp.display()
        
if __name__ == "__main__":
    main()
