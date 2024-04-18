'''propogate.py

This file contains utility classes for propogating cone colors from an
initial starting cone to the next cone in that side of the track

All propogator classes will have an __init__ method which takes in
as input a seed-cone position as a (1, 3) vectors

All propogator classes will have a .propogate method which takes
in as input a np.ndarray of size (k, 3) of cones to propogate to. 
Note that None is returned if no cones can be propogated to.

All propogate functions will return an index of the next cone among the k
to propogate the color to.
'''

import numpy as np

class PropagatorNaive:
    def __init__(self, seed_cone_pos, max_prop_dist=4):
        '''initialize naive propogator, when propogating, next iterate
        must be within max_prop_dist in order for propgation to succeed'''        
        assert(seed_cone_pos.shape == (1, 3))

        self.seed_cone_pos = seed_cone_pos
        self.max_prop_dist = max_prop_dist

    def propogate(self, cones_pos):
        '''propogate by grabbing closest cone to seed'''
        assert(cones_pos.ndim == 2 and cones_pos.shape[1] == 3)
        if cones_pos.shape[0] == 0:
            return None

        # find the closest cone to cones_pos within appropriate distance
        diff = cones_pos - self.seed_cone_pos
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        min_dist_idx = np.argmin(dists)

        # update seed if closest cone is within appropriate distance
        if dists[min_dist_idx] < self.max_prop_dist:
            self.seed_cone_pos = cones_pos[min_dist_idx, :].reshape((1, 3))
            return min_dist_idx
        else:
            return None