'''
Base Merger: Merges cones from different pipelines

Classes:
    custom_cone: custom class to store cone information
        - __init__ : initializes custom cone object
            - xPos (float): x position of cone
            - yPos (float): y position of cone
            - zPos (float): z position of cone
            - color (str): color of cone
            - pipeline (PipelineType): pipeline cone came from
        - __repr__ : returns string representation of custom cone
        - __str__ : returns string representation of custom cone
    BaseMerger: class to merge cones from different pipelines
        - __init__ : initializes BaseMerger object
            - required_pipelines (List[PipelineType]): list of required pipelines
            - zed_dist_limit (float): distance limit for ZED cones
            - lidar_dist_limit (float): distance limit for LIDAR cones
            - debug (bool): debug flag
        - reset : resets all tracking information for sufficiency
        - add : adds cones to BaseMerger object
            - cones (Cones): cones to add
            - pipeline (PipelineType): pipeline cones came from
        - sufficient : returns true iff non-zero number of cones have been added 
            and required pipelines are obtained
        - dist : returns distance between two cones
        - _naive_merge : merges all cones from all pipelines
        - merge : returns all cones added since last reset() (doesn't remove dups)
            - Goes through the entire cone list, trusting color values from cameras and position information form LIDAR
        - display : displays merged cones if debug flag is set

'''

from perc22a.mergers.MergerInterface import Merger
from perc22a.predictors.utils.cones import Cones
from perc22a.mergers.PipelineType import PipelineType
from perc22a.predictors.utils.vis.Vis3D import Vis3D
from perc22a.predictors.utils.vis.Vis2D import Vis2D

from perc22a.mergers.constants import \
    MAX_MERGE_DISTANCE, \
    ZED_DIST_LIMIT, \
    LIDAR_DIST_LIMIT

import numpy as np
from typing import List


def create_dist_filter(dist):
    def dist_filter(tup):
        return np.linalg.norm(np.array(tup)[:2]) < dist
    return dist_filter

class custom_cone:
    def __init__(self, xPos, yPos, zPos, color, pipeline) -> None:
        self.x = xPos
        self.y = yPos
        self.z = zPos
        self.color = color
        self.p = pipeline

    def __repr__(self):
        return f"x: {self.x} y: {self.y} z: {self.z} c: {self.color} p: {self.p}"

    def __str__(self):
        return self.__repr__()

class BaseMerger(Merger):

    def __init__(
            self, 
            required_pipelines: List[PipelineType] = [],
            zed_dist_limit=ZED_DIST_LIMIT,
            lidar_dist_limit=LIDAR_DIST_LIMIT,
            debug=False
        ):
        self.required_pipelines_set = set(required_pipelines)
        self.zed_filter = create_dist_filter(zed_dist_limit)
        self.lidar_filter = create_dist_filter(lidar_dist_limit)

        self.debug = debug
        if debug:
            self.vis = Vis2D()

        # reset all tracking information for sufficiency
        self.reset() 

        return

    def reset(self):
        self.total_cones = {p: 0 for p in PipelineType}
        self.pipeline_cones = {p: Cones() for p in PipelineType}
        self.seen_pipelines_set = set()

        return

    def add(self, cones: Cones, pipeline: PipelineType):
        self.total_cones[pipeline] = len(cones)
        self.pipeline_cones[pipeline] = cones
        self.seen_pipelines_set.add(pipeline)

        return

    def sufficient(self) -> bool:
        non_zero_cones = sum(self.total_cones.values()) > 0
        seen_required_cones = self.required_pipelines_set.issubset(self.seen_pipelines_set)
        return non_zero_cones and seen_required_cones
    
    def dist(self, cone1, cone2):
        return ((cone1.x - cone2.x)**2 + (cone1.y - cone2.y)**2) ** 0.5
    
    def _naive_merge(self):
        merged_cones = Cones()
        for pipeline, cones in self.pipeline_cones.items():
            merged_cones.add_cones(cones)
        return merged_cones

    def merge(self) -> Cones:

        # filter out the cones from the ZED pipelines that are too far away
        self.pipeline_cones[PipelineType.ZED_PIPELINE].filter(self.zed_filter)
        self.pipeline_cones[PipelineType.ZED2_PIPELINE].filter(self.zed_filter)
        self.pipeline_cones[PipelineType.LIDAR].filter(self.lidar_filter)

        # merge all cones from all pipelines sorted by color
        all_cones = []
        for  p, cones in self.pipeline_cones.items():
            for cone in cones.blue_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "blue", p))
            for cone in cones.yellow_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "yellow", p))
            for cone in cones.orange_cones:
                all_cones.append(custom_cone(cone[0], cone[1], 0, "orange", p))
    
        merged_cones = []
        for i in all_cones:
            duplicateCones = [i]

            # find all cones that are close enough to the current cone indicating they are most likely the same cone
            for j in all_cones:
                if i != j and self.dist(i, j)  < MAX_MERGE_DISTANCE:
                    duplicateCones.append(j)

            # TODO: could speed up this computation for computing cone distances 
            # Evaluates the most likely cone position based on the average of all the cones and color
            xPos = 0
            yPos = 0
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
            
            # if there is a lidar cone, use that as the final cone position
            if hasLidar: finalCone = custom_cone(lidarX, lidarY, 0, color, PipelineType.LIDAR)
            else: finalCone = custom_cone(xPos/len(duplicateCones), yPos/len(duplicateCones), 0, color, PipelineType.ZED2_PIPELINE)

            merged_cones.append(finalCone) 

        # convert custom cones to Cones() object
        result_cones = Cones()
        for cone in merged_cones:
            if cone.color == "blue":
                result_cones.add_blue_cone(cone.x, cone.y, 0)
            elif cone.color == "yellow":
                result_cones.add_yellow_cone(cone.x, cone.y, 0)
            else:
                result_cones.add_orange_cone(cone.x, cone.y, 0)

        return result_cones


    def display(self):
        if self.debug:
            self.vis.set_cones(self.merge())
            self.vis.update()