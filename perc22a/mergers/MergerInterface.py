from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType 


class Merger:

    def __init__(self):
        '''initialize all variables for empty cone policy'''
        pass

    def add(cones: Cones, pipeline: PipelineType):
        '''add cones from a specific pipeline'''
        pass

    def sufficient() -> bool:
        '''returns whether there are enough cones to merge'''
        pass

    def merge() -> Cones:
        '''returns merged cones, requires sufficiency'''
        pass

    def reset():
        '''resets merger as if new-ly initialized'''
        pass