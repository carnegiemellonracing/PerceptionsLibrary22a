import math

import numpy as np
from skspatial.objects import Plane


def trim_cloud(points, return_mask=False):
    """
    Trims a cloud of points to reduce to a point cloud of only cone points
    by performing a naive implementation that goes as follows
        1. mask out all points that exceed a specific radius from the center
        2. mask out all points that are too close to the lidar (likely car)
        3. mask out all points that are in the ground or too high
            - this is done by measuring point distance from a
              pre-defined plane
        4. mask out all points that are outside of a specific FOV angle

    Input: points - np.array of shape (N, M) where N is the number of points
                    and M is at least 3 and the first three columns
                    correspond to the (X, Y, Z) coordinates of the point
           return_mask - boolean whether to return the mask that filters out the points
    Output: if return_mask False
                np.array of shape (N', M) where N' <= N and the resulting array
                is from the result of filtering points according to the algo
            else (return_mask True)
                (np.array described in False case, and numpy mask of length N)
    """

    # # hyperparameters for naive ground filtering
    max_height = 5
    height = -1
    r_min = 1.5
    r_max = 10
    ang_cut = math.pi / 2
    scaling = 0.015

    # [DIST] select points within a radius of the center (0,0)
    distance = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))
    mask_r_min = r_min < distance
    mask_r_max = distance < r_max
    mask_r = np.logical_and(mask_r_min, mask_r_max)

    # [ANGLE] select points that are within a specified angle +x-direction
    angle = np.abs(np.arccos(np.divide(points[:, 0], distance)))
    mask_angle = angle < ang_cut

    # [HEIGHT] select points that are above the ground but below some max height
    # after performing some transformations to consider a tilted ground
    distance = np.subtract(distance, 3.1)
    slope = np.multiply(distance, scaling)
    ground = np.add(height, slope)

    mask_z_l = points[:, 2] > ground
    mask_z_u = points[:, 2] < max_height
    mask_z = np.logical_and(mask_z_l, mask_z_u)

    # combine all masks to create a single mask and then select points
    mask = np.logical_and(mask_r, mask_z)
    mask = np.logical_and(mask, mask_angle)

    if return_mask:
        return points[mask], mask
    else:
        return points[mask]


def remove_ground(
    points, boxdim=0.5, height_threshold=0.01, xmin=-100, xmax=100, ymin=-100, ymax=100
):
    all_points = points
    points = box_range(points, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    xmax, ymax = points[:, :2].max(axis=0)
    xmin, ymin = points[:, :2].min(axis=0)
    # print(xmax, ymax, xmin, ymin)
    LPR = []
    grid_points = []

    # iterate over all cells in the 2D grid overlayed on the x and y dimensions
    for i in range(int((xmax - xmin) // boxdim)):
        for j in range(int((ymax - ymin) // boxdim)):
            # find all points within the grid cell
            bxmin, bxmax = xmin + i * boxdim, xmin + (i + 1) * boxdim
            bymin, bymax = ymin + j * boxdim, ymin + (j + 1) * boxdim
            mask_x = np.logical_and(points[:, 0] < bxmax, bxmin < points[:, 0])
            mask_y = np.logical_and(points[:, 1] < bymax, bymin < points[:, 1])
            mask = np.logical_and(mask_x, mask_y)
            box = points[mask]

            grid_points.append(box)

            # find lowest point in cell if exists
            if box.size != 0:
                minrow = np.argmin(box[:, 2])
                boxLP = box[minrow].tolist()
                LPR.append(boxLP)

    if len(LPR) > 0:
        # fit lowest points to plane and use to classify ground points
        # P, C = vis.color_matrix(fns=None, pcs=[points, np.array(LPR)])
        # vis.update_visualizer_window(None, P, colors=C)

        plane = Plane.best_fit(LPR)
        A, B, C = tuple([val for val in plane.vector])
        D = np.dot(plane.point, plane.vector)

        dist_from_plane = (
            A * all_points[:, 0] + B * all_points[:, 1] + C * all_points[:, 2] - D
        )

        # store ground plane vals here
        pc_mask = height_threshold <= dist_from_plane
    else:
        pc_mask = np.ones(all_points.shape[0], dtype=np.uint8)
        plane = None

    return all_points[pc_mask], plane


import cupy as cp
from skspatial.objects import Plane


def plane_fit(
    pointcloud, planecloud=None, return_mask=False, boxdim=0.5, height_threshold=0.01
):
    # Convert the pointclouds to GPU arrays
    pointcloud = cp.asarray(pointcloud)
    planecloud = cp.asarray(planecloud) if planecloud is not None else pointcloud

    # Ensure xmin, xmax, ymin, ymax, and boxdim are CuPy compatible
    xmin, ymin = cp.min(planecloud[:, :2], axis=0).get()
    xmax, ymax = cp.max(planecloud[:, :2], axis=0).get()
    boxdim = cp.asarray(boxdim)

    # Create grid points for each box
    xgrid = cp.arange(xmin, xmax, boxdim)
    ygrid = cp.arange(ymin, ymax, boxdim)
    xgrid, ygrid = cp.meshgrid(xgrid, ygrid)

    # Flatten the grid for vectorized operations
    xflat = xgrid.ravel()
    yflat = ygrid.ravel()
    bxmax = xflat + boxdim
    bymax = yflat + boxdim

    LPR = []

    # Vectorize the box computation using broadcasting
    for bxmin, bymin, bxmax, bymax in zip(xflat, yflat, bxmax, bymax):
        in_box = (
            (planecloud[:, 0] >= bxmin)
            & (planecloud[:, 0] < bxmax)
            & (planecloud[:, 1] >= bymin)
            & (planecloud[:, 1] < bymax)
        )

        box = planecloud[in_box]

        # Vectorize the min z computation
        if box.size > 0:
            min_z = cp.min(box[:, 2])
            boxLP = box[box[:, 2] == min_z][0].tolist()
            LPR.append(boxLP)

    # Compute the plane from the LPR points
    plane_vals = cp.array([1, 2, 3, 4])
    pc_mask = cp.ones(pointcloud.shape[0], dtype=bool)  # Default to all true

    if LPR:
        # Convert LPR back to a NumPy array for Plane fitting (skspatial not GPU compatible)
        LPR = cp.array(LPR).get()
        plane = Plane.best_fit(LPR)
        A, B, C = plane.vector
        D = cp.dot(cp.asarray(plane.point), cp.asarray(plane.vector))
        pc_compare = A * pointcloud[:, 0] + B * pointcloud[:, 1] + C * pointcloud[:, 2]
        plane_vals = cp.array([A, B, C, D])
        pc_mask = (D + height_threshold) < pc_compare

    if return_mask:
        return pointcloud[pc_mask].get(), pc_mask.get(), plane_vals.get()
    else:
        return pointcloud[pc_mask].get()


def box_range(
    pointcloud,
    return_mask=False,
    xmin=-100,
    xmax=100,
    ymin=-100,
    ymax=100,
    zmin=-100,
    zmax=100,
):
    """return points that are within the boudning box specified by the optional input parameters"""
    xrange = np.logical_and(xmin <= pointcloud[:, 0], pointcloud[:, 0] <= xmax)
    yrange = np.logical_and(ymin <= pointcloud[:, 1], pointcloud[:, 1] <= ymax)
    zrange = np.logical_and(zmin <= pointcloud[:, 2], pointcloud[:, 2] <= zmax)
    mask = np.logical_and(np.logical_and(xrange, yrange), zrange)

    points_filtered = pointcloud[mask]
    if return_mask:
        return points_filtered, mask
    else:
        return points_filtered


def circle_range(pointcloud, return_mask=False, radiusmin=0, radiusmax=100):
    """return points that are within the radius plane in the x-y dimensions only, not in the z dimension!"""
    # get everything within radius
    dists = np.sqrt(np.sum(pointcloud[:, :2] ** 2, axis=1))
    mask = np.logical_and(radiusmin <= dists, dists <= radiusmax)

    points_filtered = pointcloud[mask]
    if return_mask:
        return points_filtered, mask
    else:
        return points_filtered


def random_subset(pointcloud, p):
    """return p% of the point cloud randomly selected"""
    assert 0 < p <= 1
    n = pointcloud.shape[0]
    n_select = int(p * n)
    rand_idxs = np.random.choice(np.arange(n), n_select, replace=False)
    return pointcloud[rand_idxs, :]


def covered_centroid(pointcloud, centroids, radius=0.75, height=0.5, threshold=5):
    """filters out CENTROIDS that have some points above them with some threshold"""

    centroids_filtered = []

    for i in range(centroids.shape[0]):
        center = centroids[i, :]

        # get all points within a radius along x and y dimensions
        dists = np.sqrt(np.sum((pointcloud[:, :2] - center[:2]) ** 2, axis=-1))
        cone_points = pointcloud[dists < radius]
        high_points = cone_points[cone_points[:, 2] > height]

        if high_points.shape[0] < threshold:
            centroids_filtered.append(centroids[i, :])

    if len(centroids_filtered) > 0:
        centroids_filtered = np.vstack(centroids_filtered)
    centroids_filtered = np.array(centroids_filtered)
    return centroids_filtered
