import numpy as np

from perc22a.predictors.stereo.ThresholdPredictor import ThresholdPredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    tp = ThresholdPredictor()
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(len(dl)):
        cones = tp.predict(dl[i])
        print(cones)

if __name__ == "__main__":
    main()
