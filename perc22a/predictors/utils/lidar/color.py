import numpy as np
import matplotlib.pyplot as plt
import perc22a.predictors.utils.lidar.visualization as vis
from perc22a.predictors.utils.lidar.seed import seed_cones_svm, debug_seed

from perc22a.utils.Timer import Timer
from perc22a.predictors.utils.cones import Cones

C2RGB = {
    "blue": [7, 61, 237],  # cone color: Blue RYB
    "yellow": [255, 209, 92],  # cone color: Maize Crayola
    "nocolor": [0, 0, 0],  # undetermined cone color: Black
    "white": [255, 255, 255],  # undetermined cone color: White
    "red": [232, 49, 81],  # midline spline color: Amaranth
}


def split_by_y(points):
    """
    assumes that points is N x 3 (idx, x, y)
    returns left_arr and right_arr where left_arr is all points in points
    with y < 0 and right_arr is all points in points with y >= 0
    """
    right_idxs = points[:, 1] >= 0
    return points[np.logical_not(right_idxs)], points[right_idxs]


def next_point_simple(curr_point, yellow, points, dir, max_angle_diff=np.pi / 3.5):
    """
    assume that curr_point is (3,) and points is N x 3 (idx, x, y)
    dir is an angle in radians

    looks for a point in from of the point at some max radius
    and at some thresholded value of how far off of the radius to consider
    """


    #change angle to y axis not x axis

    max_dist = 5

    # to ignore the index
    points_index = points[:, 0].reshape((-1, 1))
    currburr = (np.cos(dir), np.sin(dir))

    # compute the distances and angles of all points relative to curr_point
    deltas = points[:, 1:] - curr_point[1:]

    deep_currburr = np.array(currburr).reshape((-1, 1))
    deep_cos_thetas = np.arccos((deltas @ deep_currburr) / np.linalg.norm(deltas, axis=-1, keepdims=True)).reshape(-1)
    deep_cross_prods = np.cross(deep_currburr.T, deltas, axis=-1)
    raangles = np.where(deep_cross_prods > 0, -deep_cos_thetas, deep_cos_thetas)

    dists = np.sqrt(np.sum(deltas**2, axis=1))
    angles2 = np.arctan2(deltas[:, 1], deltas[:, 0])
    angles3 = np.arccos
    angles = np.where(angles2 < 0, angles2 + 2 * np.pi, angles2)
    anglesbangles = np.where(angles <= dir + np.pi, dir - angles2, dir + 2 * np.pi - angles)
    #angle_diffs = np.abs(angles - dir)
    angle_diffs = np.abs(raangles)
    # for i in range(len(points)):
    #     print(angles2[i], points[i], curr_point, deltas[i])

    # get all points within angle range and distance range
    is_close = np.logical_and(dists < max_dist, angle_diffs < max_angle_diff)
    #is_close = dists < max_dist

    if np.any(is_close):
        points_close = points[is_close]
        dists_close = dists[is_close]
        angles_close = angles[is_close]
        angles_diffs_close = angle_diffs[is_close]
        angles2_close = angles2[is_close]
        #angles3_close = angles3[is_close]
        #import pdb; pdb.set_trace();
        anglesbangles_close = anglesbangles[is_close]
        raangles_close = raangles[is_close]
        #idx = np.argmin(dists_close)
        #idx = np.argmax(angles_close)
        # if yellow: 
        #     peepeepoopoo = np.where(raangles_close > 0, anglesbangles_close / dists_close ** 4, anglesbangles_close * dists_close ** 4)
        #     idx = np.argmax(peepeepoopoo)
        # else: 
        #     peepeepoopoo = np.where(raangles_close < 0, anglesbangles_close / dists_close ** 4, anglesbangles_close * dists_close ** 4)
        #     idx = np.argmin(peepeepoopoo)
        
        # anglesbangles_diffs = abs(raangles_close - raangles_close[idx])
        # is_closeblose = raangles_close < 1
        # is_closeblose_anglesbangles = dists_close[is_closeblose]
        # ridx = np.argmin(is_closeblose_anglesbangles)
        angle_diffs_close = angle_diffs[is_close]

        # print(dists_close, angles_close, dir, angle_diffs_close, raangles_close)
        ridx = np.argmin(dists_close)
        angle_diffy = angle_diffs_close[ridx]
        # print(angle_diffy)
        anglles = np.logical_and(angle_diffs_close != angle_diffy, abs(angle_diffs_close - angle_diffy) < 0.15)
        if abs(angle_diffy) > np.pi / 3 and not np.any(anglles): return None, None
        #pdb.set_trace()
        # print(points_close)
        # print(idx)
        # print(points_close[idx])
        return points_close[ridx], angles_close[ridx]
    else:
        # for i in range(len(dists)):
        #     if dists[i] < max_dist:
        #         print(angle_diffs[i], dir, points[i])
        return None, None


def plot_dir(start, dir, scale=3):
    vec_dir = scale * np.array([np.cos(dir), np.sin(dir)])
    end = start + vec_dir
    plt.arrow(start[0], start[1], vec_dir[0], vec_dir[1], width=0.25)


def plot(centers, colors):
    """
    assumes that centers is N x 2
    and colors is N x 1 and corresponds to centers
    """
    plt.scatter(centers[:, 0], centers[:, 1], c=colors)
    plt.scatter([0], [0], c="red")
    plt.xlim([-20, 10])
    plt.ylim([-5, 30])
    plt.gca().set_aspect("equal")


def color_cones(centers):
    """
    assumes that centers is N x 3

    algorithm should check track bounds so that we are not creating
    incorrect predictions
    """
    
    # TODO: get a better algorithm for selecting the first point!!!
    # TODO: get a better algorithm for selecting the next point!!!
    # TODO: when performing a scan, should we rotate the centers for a better direction?
    #import pdb; pdb.set_trace()
    if centers.shape[0] == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    max_angle_diff = np.pi / 2.5

    # NOTE: these center filtering steps should be center filtering stages
    centers = centers[centers[:, 1] >= 0]

    all_centers = centers
    centers = centers[:, :2]

    # pdb.set_trace()

    N = centers.shape[0]

    # add index to centers
    idxs = np.arange(N).reshape((-1, 1))
    centers = np.hstack([idxs, centers])

    # default colors
    colors = ["nocolor"] * N
    centers_remaining = centers

    # seeding points
    seed_yellow_point = None
    seed_blue_point = None

    # seed the yellow and blue point
    _, right_points = split_by_y(centers_remaining)

    def get_seed(points, centers_remaining, colors, color):
        if points.shape[0] > 0:
            y_avg = np.average(centers_remaining[:, 2])

            # seed is based on distribution of points -- to help with turns
            y_avg = 0
            if y_avg < 4:
                # get closest point by lowest on y-axis
                i = np.argmin(points[:, 2])
            else:
                # get closest point by distance from origin (car)
                # scale points to discourage selecting points far from x-axis
                S = np.array([[2, 0], [0, 1]])
                dists = np.sqrt(np.sum((points[:, 1:3] @ S) ** 2, axis=1))
                i = np.argmin(dists)

            point_curr = points[i, :]
            cidx = int(point_curr[0])
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

            # angle of approach
            angle = np.pi / 2
            colors[cidx] = color

            return point_curr, centers_remaining, colors
        else:
            return None, centers_remaining, colors

    seed_yellow_point, centers_remaining, colors = get_seed(
        right_points, centers_remaining, colors, "yellow"
    )

    # YELLOW cone path
    # get closest, right point and update
    if seed_yellow_point is not None:
        # init path search
        point_curr = seed_yellow_point
        angle = np.pi / 2

        while True:
            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, True, centers_remaining, angle, max_angle_diff=max_angle_diff
            )

            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = "yellow"

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

    # BLUE cone path
    left_points, _ = split_by_y(centers_remaining)
    seed_blue_point, centers_remaining, colors = get_seed(
        left_points, centers_remaining, colors, "blue"
    )
    if seed_blue_point is not None:
        # init path search
        point_curr = seed_blue_point
        angle = np.pi / 2

        while True:

            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, False, centers_remaining, angle, max_angle_diff=max_angle_diff
            )
            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = "blue"

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]
            

    # create colors as final output
    c2id = {"yellow": 1, "blue": 0, "nocolor": -1}

    color_ids = np.array([c2id[c] for c in colors]).reshape((-1, 1))
    colors = np.array([C2RGB[c] for c in colors]) / 255

    # plot_dir(point_curr[1:], angle)
    # plot_dir(point_curr[1:], angle - np.pi / 1.75)
    # plot_dir(point_curr[1:], angle + np.pi / 1.75)
    # plot(all_centers[:, :2], colors)
    # plt.show()
    
    #pdb.set_trace()

    cone_output = np.hstack([all_centers[:, :2], color_ids])

    return cone_output, all_centers, colors



def recolor_cones_with_svm(cones: Cones, svm_model):
    """
    assumes that centers is N x 3

    very similar to the above algorithm, but seeds are first found
    using the SVM model to make propagation more consistent.

    algorithm should check track bounds so that we are not creating
    incorrect predictions
    """

    
    # TODO: get a better algorithm for selecting the first point!!!
    # TODO: get a better algorithm for selecting the next point!!!
    # TODO: when performing a scan, should we rotate the centers for a better direction?
    #import pdb; pdb.set_trace()
    if len(cones) == 0:
        return Cones()

    # convert the cones to center positions
    blue, yellow, _ = cones.to_numpy()
    centers = np.concatenate([blue, yellow], axis=0) 

    max_angle_diff = np.pi / 2.5

    # NOTE: these center filtering steps should be center filtering stages
    centers = centers[centers[:, 1] >= 0]

    # get the seed positions from the svm model
    seed_cones, remaining_centers = seed_cones_svm(centers, svm_model, max_seed_dist=8.5)

    all_centers = remaining_centers
    centers = remaining_centers[:, :2]

    N = centers.shape[0]

    # add index to centers
    idxs = np.arange(N).reshape((-1, 1))
    centers = np.hstack([idxs, centers])

    # default colors
    colors = ["nocolor"] * N
    centers_remaining = centers

    # seeding points
    blue_seed, yellow_seed, _ = seed_cones.to_numpy()

    seed_yellow_point = None if yellow_seed.shape[0] == 0 else np.array([-1, yellow_seed[0,0], yellow_seed[0,1]])
    seed_blue_point = None if blue_seed.shape[0] == 0 else np.array([-1, blue_seed[0,0], blue_seed[0,1]])

    # start propagation from the closest seed
    if seed_blue_point is None and seed_yellow_point is None:
        return Cones()
    elif seed_blue_point is None:
        seed_closer_point = seed_yellow_point
        seed_closer_color = "yellow"
        seed_farther_point = None
    elif seed_yellow_point is None:
        seed_closer_point = seed_blue_point
        seed_closer_color = "blue"
        seed_farther_point = None
    elif np.sum(seed_blue_point[1:] ** 2) < np.sum(seed_yellow_point[1:] ** 2):
        seed_closer_point = seed_blue_point
        seed_closer_color = "blue"
        seed_farther_point = seed_yellow_point
        seed_farther_color = "yellow"
    else:
        seed_closer_point = seed_yellow_point
        seed_closer_color = "yellow"
        seed_farther_point = seed_blue_point
        seed_farther_color = "blue"


    # YELLOW cone path
    if seed_closer_point is not None:
        # init path search
        point_curr = seed_closer_point
        angle = np.pi / 2

        while True:
            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, True, centers_remaining, angle, max_angle_diff=max_angle_diff
            )

            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = seed_closer_color

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]

    # BLUE cone path
    if seed_farther_point is not None:
        # init path search
        point_curr = seed_farther_point
        angle = np.pi / 2

        while True:

            # get new point
            point_new, angle_new = next_point_simple(
                point_curr, False, centers_remaining, angle, max_angle_diff=max_angle_diff
            )
            if point_new is None:
                break

            # update color and state
            cidx = int(point_new[0])
            colors[cidx] = seed_farther_color

            point_curr = point_new
            angle = angle_new

            # remove from points
            centers_remaining = centers_remaining[centers_remaining[:, 0] != cidx]
            

    # create colors as final output
    c2id = {"yellow": 1, "blue": 0, "nocolor": -1}

    color_ids = np.array([c2id[c] for c in colors]).reshape((-1, 1))
    colors = np.array([C2RGB[c] for c in colors]) / 255

    cone_output = np.hstack([all_centers[:, :2], color_ids])

    cones = seed_cones
    for i in range(cone_output.shape[0]):
        x, y, c = cone_output[i, :]
        z = all_centers[i, 2]
        if c == 1:
            cones.add_yellow_cone(x, y, z)
        elif c == 0:
            cones.add_blue_cone(x, y, z)

    return cones    
