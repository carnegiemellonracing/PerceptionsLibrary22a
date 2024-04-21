'''icp.py

Implementation of ICP by Clay Flannigan
Goat: https://github.com/ClayFlannigan/icp

Adopted by CMR Driverless for 22a
'''

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def debug_correspondences(src_pos, target_pos, correspondences):

    # get positions without correspondences
    src_uncorr_mask = np.ones(src_pos.shape[0], dtype=bool)
    src_uncorr_mask[correspondences[:, 0]] = False
    target_uncorr_mask = np.ones(target_pos.shape[0], dtype=bool)
    target_uncorr_mask[correspondences[:, 1]] = False

    # plot the correspondences in colorful colors
    for i in range(len(correspondences)):
        src_idx, target_idx = correspondences[i,:]

        points = np.array([src_pos[src_idx], target_pos[target_idx]])
        plt.scatter(points[:, 0], points[:, 1])

    # plot points without correspondences in black
    for i in range(src_uncorr_mask.shape[0]):
        if src_uncorr_mask[i]:
            plt.scatter([src_pos[i,0]], [src_pos[i,1]], c="black")
    
    for i in range(target_uncorr_mask.shape[0]):
        if target_uncorr_mask[i]:
            plt.scatter([target_pos[i,0]], [target_pos[i,1]], c="black")

    plt.show()

    return


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor_corr(src, dst, max_corr_dist=None):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Pxm array of points
        max_corr_dist: maximum distance for two points to correspond
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
        corr: Kx2 integer array of indices describing correspondences
    '''

    n, _ = src.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)

    corr = np.concatenate([np.arange(n).reshape(-1,1), indices], axis=1)
    distances = distances.reshape(-1)

    # filter away correspondences that are too far away
    if max_corr_dist is not None:
        corr = corr[distances <= max_corr_dist, :]

    return corr, distances


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001, max_corr_dist=0.5):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Pxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
        max_corr_dist: max distance correspondences can have while fitting
    Output:
        corr: (K, 2) array of indices of corresponding points between A and B (K <= min(A, B))
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
        tA: transformed A after performing ICP (approx. result of transforming A with T)
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        corr, distances = nearest_neighbor_corr(src[:m,:].T, dst[:m,:].T, max_corr_dist)
        if corr.shape[0] == 0:
            return None, None, None, None, None

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,corr[:,0]].T, dst[:m,corr[:,1]].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    transformed_A = src[:m, :].T
    T,_,_ = best_fit_transform(A, transformed_A)

    # TODO: maybe return the transformed positions from source to target
    # TODO: ensure that the transformed_A are not NaN!!!
    return corr, T, distances, i, transformed_A