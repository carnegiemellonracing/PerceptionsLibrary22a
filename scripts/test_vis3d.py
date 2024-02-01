from perc22a.predictors.utils.vis.Vis3D import Vis3D

from perc22a.data.utils.dataloader import DataLoader
from perc22a.data.utils.DataType import DataType


def main():

    dl = DataLoader("perc22a/data/raw/track-testing-09-29")
    vis = Vis3D()

    while True:
        for i in range(len(dl)):
            data = dl[i]
            points = data[DataType.HESAI_POINTCLOUD]

            points = points[:, :3]
            points = points[:, [1, 0, 2]]
            points[:, 0] = -points[:, 0]


            vis.set_points(points)
            vis.update()


if __name__ == "__main__":
    main()