'''
Merger Interface

Functions:
    add: add cones from a specific pipeline
    sufficient: returns whether there are enough cones to merge
    merge: returns merged cones, requires sufficiency
    reset: resets merger as if new-ly initialized
'''

from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType 


class Merger:

    def __init__(self):
        '''initialize all variables for empty cone policy'''
        pass

    def add(self, cones: Cones, pipeline: PipelineType):
        '''add cones from a specific pipeline'''
        pass

    def sufficient(self) -> bool:
        '''returns whether there are enough cones to merge'''
        pass

    def merge(self) -> Cones:
        '''returns merged cones, requires sufficiency'''
        pass

    def reset(self):
        '''resets merger as if new-ly initialized'''
        pass