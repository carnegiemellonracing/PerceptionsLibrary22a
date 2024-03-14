'''defines an enum representing the type of pipelines perceptions returns'''

from enum import Enum

class PipelineType(Enum):

    LIDAR = "lidar"
    ZED_PIPELINE = "zed"
    ZED2_PIPELINE = "zed2"