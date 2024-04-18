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

import matplotlib.pyplot as plt
import numpy as np

class PropagatorNaive:
    def __init__(self, seed_cone_pos, max_prop_dist=4):
        '''initialize naive propogator, when propogating, next iterate
        must be within max_prop_dist in order for propgation to succeed'''        
        assert(seed_cone_pos.shape == (1, 3))

        self.seed_cone_pos = seed_cone_pos
        self.max_prop_dist = max_prop_dist

    def propagate(self, cones_pos):
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
        
    def debug(self, seed_cone_pos, cones_pos, idx):
        plt.scatter(seed_cone_pos[0, 0], seed_cone_pos[0, 1], c="red") 
        plt.scatter(cones_pos[:,0], cones_pos[:,1], c="black")
        plt.scatter([cones_pos[idx,0]], [cones_pos[idx,1]], c="orange")
        plt.show()

        pass


class PropagateDirection:
    def __init__(self, seed_cone_pos):
        '''uses estimation of direction of track to guide cone selection'''
        assert(seed_cone_pos.shape == (1, 3))

        self.seed_cone_pos = seed_cone_pos
        self.max_angle_diff = np.pi / 2.5
        self.curr_angle = np.pi / 2
        self.max_dist = 8

    def next_point_simple(self, curr_point, points, max_angle_diff=np.pi / 3.5):
        """
        assume that curr_point is (3,) and points is N x 3 (idx, x, y)
        dir is an angle in radians

        looks for a point in from of the point at some max radius
        and at some thresholded value of how far off of the radius to consider
        """


        #change angle to y axis not x axis
        dir = self.curr_angle

        max_dist = 8

        # to ignore the index
        currburr = (np.cos(dir), np.sin(dir))

        # compute the distances and angles of all points relative to curr_point
        deltas = points[:, 1:] - curr_point[1:]

        deep_currburr = np.array(currburr).reshape((-1, 1))
        deep_cos_thetas = np.arccos((deltas @ deep_currburr) / np.linalg.norm(deltas, axis=-1, keepdims=True)).reshape(-1)
        deep_cross_prods = np.cross(deep_currburr.T, deltas, axis=-1)
        raangles = np.where(deep_cross_prods > 0, -deep_cos_thetas, deep_cos_thetas)

        dists = np.sqrt(np.sum(deltas**2, axis=1))
        angles2 = np.arctan2(deltas[:, 1], deltas[:, 0])
        angles = np.where(angles2 < 0, angles2 + 2 * np.pi, angles2)
        anglesbangles = np.where(angles <= dir + np.pi, dir - angles2, dir + 2 * np.pi - angles)
        angle_diffs = np.abs(raangles)

        # get all points within angle range and distance range
        print(dists)
        print(angle_diffs)
        is_close = np.logical_and(dists < max_dist, angle_diffs < max_angle_diff)

        if np.any(is_close):
            points_close = points[is_close]
            dists_close = dists[is_close]
            angles_close = angles[is_close]

            angle_diffs_close = angle_diffs[is_close]

            ridx = np.argmin(dists_close)
            angle_diffy = angle_diffs_close[ridx]
            anglles = np.logical_and(angle_diffs_close != angle_diffy, abs(angle_diffs_close - angle_diffy) < 0.15)

            self.curr_angle = angles_close[ridx]
            return int(points_close[ridx, 0].item())
        else:
            return None

    def propagate(self, cones_pos):
        curr_point = self.seed_cone_pos.reshape(3)
        
        points = np.array(cones_pos)
        points[:, 1:] = points[:, 0:2]
        points[:, 0] = np.arange(points.shape[0])

        idx = self.next_point_simple(curr_point, points)
        if idx is not None:
            self.seed_cone_pos = cones_pos[idx, :].reshape((1, 3))

        return idx
