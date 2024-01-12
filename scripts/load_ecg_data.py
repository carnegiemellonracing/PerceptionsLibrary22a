
from perc22a.data.utils.dataloader import DataLoader

from perc22a.predictors.aggregate.AggregatePredictor import AggregatePredictor
from perc22a.predictors.lidar.LidarPredictor import LidarPredictor
from perc22a.predictors.stereo.StereoPredictor import StereoPredictor

SHIFT = 5


def main():
    
    dl = DataLoader("perc22a/data/raw/ecg-12-02-full")
    predictor = AggregatePredictor("/home/chip/Desktop/Documents/driverless-packages/PerceptionsLibrary22a/sensor_config.yaml")

    lp = LidarPredictor()
    sp = StereoPredictor('ultralytics/yolov5', 'perc22a/predictors/stereo/model_params.pt')

    for i in range(SHIFT + 15, len(dl)):
        lidar_data = dl[i]
        stereo_data = dl[i - SHIFT]

        lp.predict(lidar_data)
        sp.predict(stereo_data)

        sp.display()

        # data = {}
        # data["points"] = lidar_data["points"]
        # data["left_color"] = stereo_data["left_color"]
        # data["xyz_image"] = stereo_data["xyz_image"]

        # predictor.predict(data)
        # predictor.display()
        input()
        

if __name__ == "__main__":
    main()
