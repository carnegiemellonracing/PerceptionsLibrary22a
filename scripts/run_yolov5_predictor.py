from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.data.utils.dataloader import DataLoader


def main():
    sp = YOLOv5Predictor()
    dl = DataLoader("perc22a/data/raw/three-laps-large")

    for i in range(len(dl)):
        cones = sp.predict(dl[i])
        print(cones)
        sp.display()


if __name__ == "__main__":
    main()
