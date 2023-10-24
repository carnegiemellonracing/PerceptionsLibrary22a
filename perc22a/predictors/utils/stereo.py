import statistics
import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import perc22a.predictors.stereo.cfg as cfg

def calc_box_center(box):
    """
    Args:
        bounding_box: Box containing object of interest
                        bound[0] = x1 (top left - x)
                        bound[1] = y1 (top left - y)
                        bound[2] = x2 (bottom right - x)
                        bound[3] = y2 (bottom right - y)
        Returns:
            center_x, center_y - center x, y pixel of bounding box 
    """
    x1, y1 = box[0], box[1]
    x2, y2 = box[2], box[3]
    center_y = int((x1+x2)/2)  # x-coordinate of bounding box center
    center_x = int((y1+y2)/2)  # y-coordinate of bounding box center
    return center_x, center_y

def get_object_depth(depth_map, box, padding=2):
    """
    Calculates the median z position of supplied bounding box
    Args:
        box: Box containing object of interest
                      bound[0] = x1 (top left - x)
                      bound[1] = y1 (top left - y)
                      bound[2] = x2 (bottom right - x)
                      bound[3] = y2 (bottom right - y)
        padding: Num pixels around center to include in depth calculation
                       REQUIREMENT: padding <= (x2-x1)/2 and padding <= (y2-y1)/2 
                       Default = 2
    Returns:
        float - Depth estimation of object in interest 
    """
    z_vect = []
    center_x, center_y = calc_box_center(box)

    # Iterate through center pixels and append them to z_vect
    for i in range(max(center_x - padding, 0), min(center_x + padding, 720)):
        for j in range(max(center_y - padding, 0), min(center_y + padding, 1280)):
            z = depth_map[i, j]
            if not np.isnan(z) and not np.isinf(z):
                z_vect.append(z)

    # Try calculating the mean of the depth of the center pixels
    # Catch all exceptions and return -1 if mean is unable to be calculated
    try:
        z_mean = statistics.mean(z_vect)
    except Exception:
        print("Unable to compute z_mean")
        z_mean = -1
    return z_mean

def get_world_coords(coords):
    world_xs, world_ys, world_zs = coords[:,:,0].reshape(-1), coords[:,:,1].reshape(-1), coords[:,:,2].reshape(-1)

    idxs = np.logical_and(~np.isnan(world_xs), ~np.isinf(world_xs))
    world_xs = world_xs[idxs]
    world_ys = world_ys[idxs]
    world_zs = world_zs[idxs]

    if world_xs.shape[0] == 0:
        raise Exception(f"\t[PERCEPTIONS WARNING] cone detected cone but no depth; throwing away")

    world_x = np.mean(world_xs)
    world_y = np.mean(world_ys)
    world_z = np.mean(world_zs)

    return world_x, world_y, world_z


class CFG_PERCEPTIONS:
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720
    RED_THRESHOLD = 100

class CFG_COLORS(Enum):
    BLUE = 1
    YELLOW = 2
    ORANGE = 3
    UNKNOWN = 4


def get_cone_color(left_frame, box, padding=2):
    # return config.COLORS.BLUE
    center_x, center_y = calc_box_center(box)
    min_x = max(center_x - padding, 0)
    max_x = min(center_x + padding, CFG_PERCEPTIONS.IMAGE_HEIGHT)

    min_y = max(center_y - padding, 0)
    max_y = min(center_y + padding, CFG_PERCEPTIONS.IMAGE_WIDTH)

    rgbs = left_frame[min_x:max_x, min_y:max_y, :3].reshape(-1, 3)
    avg_rgbs = np.average(rgbs, axis=0)
    if avg_rgbs[2] < CFG_PERCEPTIONS.RED_THRESHOLD:
        return cfg.COLORS.BLUE
    else:
        return cfg.COLORS.YELLOW