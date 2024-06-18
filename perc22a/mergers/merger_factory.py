'''factory.py

Functions for constructing mergers with specific properties

Functions:
    create_any_merger: simple merger that requires a single pipeline
    create_lidar_merger: simple merger that requires the LiDAR pipeline
    create_zed_merger: simple merger that requires cones from both ZEDs
    create_all_merger: simple merger that requires cones from all pipelines
'''

from perc22a.mergers.BaseMerger import BaseMerger
from perc22a.mergers.PipelineType import PipelineType

def create_any_merger():
    '''simple merger that requires a single pipeline'''
    return BaseMerger(
        required_pipelines=[]
    )

# TODO: add in parameters to influence the ZED filter distance
def create_lidar_merger():
    '''simple merger that requires the LiDAR pipeline'''

    return BaseMerger(
        required_pipelines=[PipelineType.LIDAR]
    )

def create_zed_merger():
    '''simple merger that requires cones from both ZEDs'''

    return BaseMerger(
        required_pipelines=[
            PipelineType.ZED_PIPELINE, 
            PipelineType.ZED2_PIPELINE
        ]
    )

def create_all_merger():
    '''simple merger that requires cones from all pipelines'''
    return BaseMerger(
        required_pipelines=[
            PipelineType.LIDAR,
            PipelineType.ZED_PIPELINE, 
            PipelineType.ZED2_PIPELINE
        ]
    )