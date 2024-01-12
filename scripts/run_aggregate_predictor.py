from perc22a.predictors.aggregate.AggregatePredictor import AggregatePredictor
from perc22a.data.utils.dataloader import DataLoader

def main():
    ap = AggregatePredictor('sensor_config.yaml')
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(40, len(dl)):
        cones = ap.predict(dl[i])
        # print(cones)
        # ap.display()


if __name__ == "__main__":
    main()
