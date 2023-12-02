import numpy as np

from perc22a.predictors.stereo.ThresholdPredictor import ThresholdPredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    tp = ThresholdPredictor()

    for i in range(23, 155):
        array = np.load(f"perc22a/data/raw/track-testing-09-29/instance-{i}.npz")
        cones = tp.predict(array)
        print(cones)
        tp.visualize(array)

if __name__ == "__main__":
    main()
