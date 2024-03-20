'''Base Merger

sufficient() -> true iff non-zero number of cones have been added 
    and required pipelines are obtained
merge() -> returns all cones added since last reset() (doesn't remove dups)
'''

from perc22a.mergers.MergerInterface import Merger
from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType
from perc22a.predictors.utils.vis.Vis3D import Vis3D

from typing import List

MAX_TOLERATED_DIFFERENCE = 1

class custom_cone:
    def __init__(self, xPos, yPos, zPos, color, pipeline) -> None:
        self.x = xPos
        self.y = yPos
        self.z = zPos
        self.color = color
        self.p = pipeline

class BaseMerger(Merger):

    def __init__(self, required_pipelines: List[PipelineType] = []):
        self.required_pipelines_set = set(required_pipelines)
        self.vis = Vis3D()
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
    
    def dist(self, cone1, cone2):
        return ((cone1.x - cone2.x)**2 + (cone1.y - cone2.y)**2) ** 0.5

    def merge(self) -> Cones:
        # TODO: implement height zero-ing and distance limiting on ZED pipelines
        # TODO: do duplication removal

        # all_cones = Cones()
        # result_cones = Cones()
        # for p, cones in self.pipeline_cones.items():
        #     all_cones.add_cones(cones)

        # all_cones_list = all_cones.blue_cones + all_cones.yellow_cones + all_cones.orange_cones
        # blue_length = len(all_cones.blue_cones)
        # yellow_length = len(all_cones.yellow_cones)
        # counter = 0
        # result_cones_list = []

        # for i in all_cones_list:
        #     isDuplicate = False
        #     for j in result_cones_list:
        #         if self.dist(i, j) < MAX_TOLERATED_DIFFERENCE: isDuplicate = True
        #     if not isDuplicate: 

        #         if 0 <= counter and counter < blue_length:
        #             result_cones.add_blue_cone(i[0], i[1], 0)
        #         elif blue_length <= counter and counter < (blue_length + yellow_length):
        #             result_cones.add_yellow_cone(i[0], i[1], 0)
        #         else:
        #             result_cones.add_orange_cone(i[0], i[1], 0)
                 
        #         result_cones_list.append(i)
        #     counter+=1

        # return result_cones

        all_cones = []
        for  p, cones in self.pipeline_cones.items():
            for cone in cones.blue_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "blue", p))
            for cone in cones.yellow_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "yellow", p))
            for cone in cones.blue_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "orange", p))
    
        result_cones = Cones()
        for i in all_cones:
            duplicateCones = [i]
            for j in all_cones:
                if self.dist(i, j)  < MAX_TOLERATED_DIFFERENCE:
                    duplicateCones.append(j)

            # TODO: could speed up this computation for computing cone distances 
            xPos = i.x
            yPos = i.y
            hasLidar = False
            lidarX = 0
            lidarY = 0
            color = i.color
            for d in duplicateCones:
                if d.p == PipelineType.LIDAR: 
                    hasLidar = True
                    lidarX = d.x
                    lidarY = d.y
                else:
                    xPos += d.x
                    yPos += d.y
                    color = d.color
            
            if hasLidar: finalCone = custom_cone(lidarX, lidarY, 0, color, PipelineType.LIDAR)
            else: finalCone = custom_cone(xPos/len(duplicateCones), yPos/len(duplicateCones), 0, color, PipelineType.ZED2_PIPELINE)

            if finalCone.color == "blue":
                result_cones.add_blue_cone(finalCone.x, finalCone.y, finalCone.z)
            elif finalCone.color == "yellow":
                result_cones.add_yellow_cone(finalCone.x, finalCone.y, finalCone.z)
            else:
                result_cones.add_orange_cone(finalCone.x, finalCone.y, finalCone.z)

        return result_cones


    def display(self):
        self.vis.set_cones(self.merge())
        self.vis.update()


# sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

# X (n, 2), ... np.sqrt(X @ X.T)

# '''Base Merger

# sufficient() -> true iff non-zero number of cones have been added 
#     and required pipelines are obtained
# merge() -> returns all cones added since last reset() (doesn't remove dups)
# '''

# from perc22a.mergers.MergerInterface import Merger
# from perc22a.predictors.utils.cones import Cones
# from perc22a.mergers.PipelineType import PipelineType

# from types import List

# class BaseMerger(Merger):

#     def __init__(self, required_pipelines: List[PipelineType] = []):
#         self.required_pipelines_set = set(required_pipelines)

#         self.reset() 

#         return

#     def reset(self):
#         self.total_cones = 0
#         self.pipeline_cones = {p: Cones() for p in PipelineType}
#         self.seen_pipelines_set = set()

#         return

#     def add(self, cones: Cones, pipeline: PipelineType):
#         self.total_cones += len(cones)
#         self.pipeline_cones[pipeline].add_cones(cones)
#         self.seen_pipelines_set.add(pipeline)

#         return

#     def sufficient(self) -> bool:
#         non_zero_cones = len(self.pipeline_cones) > 0
#         seen_required_cones = self.required_pipelines_set.issubset(self.seen_pipelines_set)
#         return non_zero_cones and seen_required_cones

#     def merge(self) -> Cones:
#         # TODO: implement height zero-ing and distance limiting on ZED pipelines
#         # TODO: do duplication removal

#         all_cones = Cones()
#         for p, cones in self.pipeline_cones.items():
#             all_cones.add_cones(cones)

#         return all_cones
    

# # sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

# # X (n, 2), ... np.sqrt(X @ X.T)