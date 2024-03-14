'''Naive Merger

sufficient() -> true iff non-zero number of cones have been added 
    and required pipelines are obtained
merge() -> returns all cones added since last reset() (doesn't remove dups)
'''

from perc22a.mergers.MergerInterface import Merger
from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType

from types import List

class NaiveMerger(Merger):

    def __init__(self, required_pipelines: List[PipelineType] = []):
        self.required_pipelines_set = set(required_pipelines)

        self.reset() 

        return

    def reset(self):
        self.total_cones = 0
        self.pipeline_cones = {p: Cones() for p in PipelineType}
        self.seen_pipelines_set = set()

        return

    def add(self, cones: Cones, pipeline: PipelineType):
        self.total_cones += len(cones)
        self.pipeline_cones[pipeline].add_cones(cones)
        self.seen_pipelines_set.add(pipeline)

        return

    def sufficient(self) -> bool:
        non_zero_cones = len(self.pipeline_cones) > 0
        seen_required_cones = self.required_pipelines_set.issubset(self.seen_pipelines_set)
        return non_zero_cones and seen_required_cones

    def merge(self) -> Cones:
        # TODO: implement height zero-ing and distance limiting on ZED pipelines

        all_cones = Cones()
        for p, cones in self.pipeline_cones.items():
            all_cones.add_cones(cones)

        return all_cones