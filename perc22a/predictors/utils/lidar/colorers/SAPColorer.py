'''SAPColorer.py

SAP stands for Seed And Propogate. This colorer will perform coloring as follows
    1. using a seeding fn, initialize starting seeds for blue and yellow cones
    2. using an propagating fn, iteratively find next blue and yellow cone
'''

from perc22a.predictors.utils.cones import Cones

from perc22a.predictors.utils.lidar.colorers.ColorerInterface import Colorer
import perc22a.predictors.utils.lidar.colorers.seed as seed
from perc22a.predictors.utils.lidar.colorers.propagate import PropagatorNaive

from perc22a.predictors.utils.lidar.colorers.constants import *

import numpy as np

BLUE_STR = "blue"
YELLOW_STR = "yellow"


class SAPColorer(Colorer):
    # TODO: desirable to have track that satisfies propogator properties

    def __init__(self, seed=SEED_NAIVE_NAME, propogate=PROPOGATE_NAIVE_NAME):
        '''Initializes seed and propogate colorer

        seed: can be "naive" or "svm"
        propogate: can be "naive" or "direction"
        '''
        assert(seed == SEED_NAIVE_NAME or seed == SEED_SVM_NAME)
        assert(propogate == PROPOGATE_NAIVE_NAME or propogate == PROPOGATE_DIRECTION_NAME)

        self.svm_model = None

        self.propogator_name = propogate
        self.seed_name = seed

        return
    
    def _seed(self, cones_pos):
        if self.seed_name == SEED_NAIVE_NAME or self.svm_model is None:
            return seed.seed_cones_naive(cones_pos)
        else:
            return seed.seed_cones_svm(cones_pos, self.svm_model)
        
    def _closer_seed_color(self, seed_cones: Cones):
        blue, yellow, orange = seed_cones.to_numpy()

        if blue.shape[0] == 0:
            return YELLOW_STR
        elif yellow.shape[0] == 0:
            return BLUE_STR
        
        # otherwise, find closest of two cones to determine which to seed first
        blue_dist_sq = np.sum(blue ** 2)
        yellow_dist_sq = np.sum(yellow ** 2)
        return BLUE_STR if blue_dist_sq < yellow_dist_sq else YELLOW_STR
        
    def _init_propogator(self, seed_cone_pos):
        if self.seed_name == PROPOGATE_NAIVE_NAME:
            return PropagatorNaive(seed_cone_pos)
        else:
            return None
        
    def _propogate(self, cones: Cones, seed_cone_pos, remaining_cones_pos, color):

        propogator = self._init_propogator(seed_cone_pos)

        while True:
            idx = propogator.propogate(remaining_cones_pos)

            if idx is None:
                break
            
            # if propogated, update remaining_cone_pos and cones
            selected_cone = remaining_cones_pos[idx, :]
            remaining_cones_pos = np.delete(remaining_cones_pos, idx, axis=0)

            if color == BLUE_STR:
                cones.add_blue_cone(selected_cone[0], selected_cone[1], selected_cone[2])
            else:
                cones.add_yellow_cone(selected_cone[0], selected_cone[1], selected_cone[2])

        return cones, remaining_cones_pos        

    def update_svm(self, svm_model):
        self.svm_model = svm_model

    def color(self, cones_pos):
        # return empty cones if necessary
        if cones_pos.shape[0] == 0:
            return Cones()

        # initialize the seed
        seed_cones, remaining_cones_pos = self._seed(cones_pos)
        blue, yellow, orange = seed_cones.to_numpy()

        # determine which of the two seeds are closer
        closer_color = self._closer_seed_color(seed_cones)
        other_color = YELLOW_STR if closer_color == BLUE_STR else BLUE_STR

        closer_seed = blue if closer_color == BLUE_STR else yellow
        other_seed = yellow if closer_color == BLUE_STR else blue

        # start propagating with the closer color
        if closer_seed is not None:
            cones, remaining_cones_pos = self._propogate(
                seed_cones,
                closer_seed,
                remaining_cones_pos,
                closer_color
            )

        # propogate non-closer color
        if other_color is not None:
            cones, remaining_cones_pos = self._propogate(
                cones,
                other_seed,
                remaining_cones_pos,
                closer_color
            )

        return cones