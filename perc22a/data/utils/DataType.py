'''config.py

String definitions for each sensor data-type that perc22a uses.

Use to access data from the DataLoader (e.g. something like dl[i][LEFT_COLOR])

Datatypes:
    ZED_LEFT_COLOR: left camera color image from ZED stereocamera
    ZED_XYZ_IMG: depth image from ZED stereocamera
    ZED2_LEFT_COLOR: left camera color image from ZED2 stereocamera
    ZED2_XYZ_IMG: depth image from ZED2 stereocamera
    HESAI_POINTCLOUD: pointcloud data from Hesai lidar
    GPS_POSE: pose data from GPS sensor
'''

from enum import Enum

class DataType(Enum):

    # zed stereocamera data
    ZED_LEFT_COLOR = "left_color"
    ZED_XYZ_IMG = "xyz_image"
   
    ZED2_LEFT_COLOR = "zed2_left_color"
    ZED2_XYZ_IMG = "zed2_xyz_image"

    # hesai pointcloud data
    HESAI_POINTCLOUD = "points"

    # gps related data
    GPS_POSE = "pose"

    # add other data here ...
