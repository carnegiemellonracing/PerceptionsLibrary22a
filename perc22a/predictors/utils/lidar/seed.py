'''seed.py

Contains utility functions for getting the seed that will initialize the 
coloring process for Colorers.

Functions for getting the seed can be found below.

All seed functions take in at least a numpy array of cone positions and they
return a (Cones, np.ndarray) tuple containing the seeded cones (at most
1 blue and 1 yellow, if any for either) and the remaining unseeded cones
respectively.
'''

from perc22a.predictors.utils.cones import Cones
from perc22a.svm.SVM import BLUE_LABEL, YELLOW_LABEL
from perc22a.predictors.utils.vis.Vis2D import Vis2D

import numpy as np
import matplotlib.pyplot as plt


def debug_seed(cones_pos, seed_cones: Cones, remaining_cone_pos):

    plt.scatter(remaining_cone_pos[:, 0], remaining_cone_pos[:, 1], c="black")
    
    blue_cones, yellow_cones, _ = seed_cones.to_numpy()
    assert(blue_cones.shape[0] <= 1 and yellow_cones.shape[0] <= 1)

    if blue_cones.shape[0] == 1:
        plt.scatter([blue_cones[0, 0]], [blue_cones[0, 1]], c="blue")
    if yellow_cones.shape[0] == 1:
        plt.scatter([yellow_cones[0, 0]], [yellow_cones[0, 1]], c="gold")

    plt.scatter([0], [0], c="red")
    
    plt.grid()
    plt.show()

    return


def closest_to_origin(points):
    '''returns index of point in (N, D) matrix closest to origin along with distance'''
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    idx = np.argmin(dists)
    return idx, dists[idx]

def split_by_xsign(points):
    '''returns two np array of points, those with negative x and positive x
    If no such points with negative y or no such points with positive y,
    then will return empty (0, d) array for respective return value
    '''
    _, D = points.shape
    left_points = points[np.where(points[:, 0] < 0)]
    right_points = points[np.where(points[:, 0] >= 0)]

    return left_points.reshape((-1, D)), right_points.reshape((-1, D))

def seed_cones_naive(cones_pos, max_seed_dist=15):
    '''Naive implementation of getting the seed

    Closest cone to the left of the car (if any) is blue.
    Closest cone to the right of the car (if any) is yellow. 
    '''
    assert(cones_pos.ndim == 2 and cones_pos.shape[1] == 3)
    # NOTE: max_seed_dist needs to be modified based off of xscale
    xscale = 1.1

    left_cones_pos, right_cones_pos = split_by_xsign(cones_pos)

    # initialize Cone object for returning seeds
    seed_cones = Cones()

    # determine the closest cones on eithe rside from origin
    if left_cones_pos.shape[0] > 0:
        modified_left_cones_pos = np.array(left_cones_pos)
        modified_left_cones_pos[:, 0] *= xscale
        closest_idx, closest_dist = closest_to_origin(modified_left_cones_pos)

        # add only if within trustworthy distance
        if closest_dist <= max_seed_dist:
            closest_pos = left_cones_pos[closest_idx, :]
            seed_cones.add_blue_cone(closest_pos[0], closest_pos[1], closest_pos[2])
            left_cones_pos = np.delete(left_cones_pos, closest_idx, axis=0)

    if right_cones_pos.shape[0] > 0:
        modified_right_cones_pos = np.array(right_cones_pos)
        modified_right_cones_pos[:, 0] *= xscale
        closest_idx, closest_dist = closest_to_origin(modified_right_cones_pos)

        # add only if within trustworthy distance
        if closest_dist <= max_seed_dist:
            closest_pos = right_cones_pos[closest_idx, :]
            seed_cones.add_yellow_cone(closest_pos[0], closest_pos[1], closest_pos[2])
            right_cones_pos = np.delete(right_cones_pos, closest_idx, axis=0)

    # remerge the remaining cones
    remaining_cones_pos = np.concatenate([left_cones_pos, right_cones_pos], axis=0)
    return seed_cones, remaining_cones_pos


def seed_cones_svm(cones_pos, svm_model, max_seed_dist=10):
    '''uses SVM to determine seed by predicting yellow and blue cones and
    taking closest blue and closest yellow within reasonable distance'''

    # predict the colors of all cones
    pred_labels = svm_model.predict(cones_pos[:, :2])
    blue_cones_pos = cones_pos[np.where(pred_labels == BLUE_LABEL)]
    yellow_cones_pos = cones_pos[np.where(pred_labels == YELLOW_LABEL)]

    # initialize Cone object for returning seeds
    seed_cones = Cones()

    # determine the closest cones for each color predicted by SVM
    if blue_cones_pos.shape[0] > 0:
        closest_blue_idx, closest_blue_dist = closest_to_origin(blue_cones_pos)

        # add only if within trustworthy distance
        if closest_blue_dist <= max_seed_dist:
            closest_pos = blue_cones_pos[closest_blue_idx, :]
            seed_cones.add_blue_cone(closest_pos[0], closest_pos[1], closest_pos[2])
            blue_cones_pos = np.delete(blue_cones_pos, closest_blue_idx, axis=0)

    if yellow_cones_pos.shape[0] > 0:
        closest_yellow_idx, closest_yellow_dist = closest_to_origin(yellow_cones_pos)

        # add only if within trustworthy distance
        if closest_yellow_dist <= max_seed_dist:
            closest_pos = yellow_cones_pos[closest_yellow_idx, :]
            seed_cones.add_yellow_cone(closest_pos[0], closest_pos[1], closest_pos[2])
            yellow_cones_pos = np.delete(yellow_cones_pos, closest_yellow_idx, axis=0)

    # remerge the remaining cones
    remaining_cones_pos = np.concatenate([blue_cones_pos, yellow_cones_pos], axis=0)
    return seed_cones, remaining_cones_pos