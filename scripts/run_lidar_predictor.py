
# perceptions specific imports
from perc22a.data.utils.dataloader import DataLoader
from perc22a.predictors import LidarPredictor

# general python imports
import time
import cv2


def main():
    # initialize data loader and lidar predictor
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")
    lp = LidarPredictor()

    for i in range(50, len(dl)):
        # load the i-th image from track testing run
        data = dl[i]
        lp.predict(data)
        lp.display()

        
if __name__ == "__main__":
    main()