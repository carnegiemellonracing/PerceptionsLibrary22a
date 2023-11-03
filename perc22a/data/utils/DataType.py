'''config.py

String definitions for each sensor data-type that perc22a uses.

Use to access data from the DataLoader (e.g. something like dl[i][LEFT_COLOR])
'''

from enum import Enum

class DataType(Enum):

    # zed stereocamera data
    ZED_LEFT_COLOR = "left_color"
    ZED_RIGHT_COLOR = "right_color"
    ZED_XYZ_IMG = "xyz_image"
    ZED_DEPTH_IMG = "depth_image"

    # hesai pointcloud data
    HESAI_POINTCLOUD = "points"

    # add other data here ...