'''Naive Merger

sufficient() -> true iff non-zero number of cones have been added
merge() -> returns all cones added since last reset() (doesn't remove dups)
'''

from perc22a.mergers.MergerInterface import Merger
from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType

class NaiveMerger(Merger):

    def __init__(self):
        self.reset() 

        return

    def reset(self):
        self.total_cones = 0
        self.pipeline_cones = {p: Cones() for p in PipelineType}

        return

    def add(self, cones: Cones, pipeline: PipelineType):
        self.total_cones += len(cones)
        self.pipeline_cones[pipeline].add_cones(cones)

        return

    def sufficient(self) -> bool:
        return len(self.pipeline_cones) > 0

    def merge(self) -> Cones:
        all_cones = Cones()
        for p, cones in self.pipeline_cones.items():
            all_cones.add_cones(cones)

        return all_cones