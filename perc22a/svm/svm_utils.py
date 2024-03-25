
from perc22a.predictors.utils.cones import Cones
from perc22a.utils.Timer import Timer

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

DEBUG_SVM = False
DEBUG_PRED = False
DEBUG_POINTS = False

def debug_svm(cones: Cones, X, y, clf):

    # Settings for plotting
    _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = np.min(X[:,0]), np.max(X[:,0]), np.min(X[:,1]), np.max(X[:,1])
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": clf, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    # Plot bigger circles around samples that serve as support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=250,
        facecolors="none",
        edgecolors="k",
    )
    # Plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=150, edgecolors="k")
    # ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    # ax.set_title(f" Decision boundaries of {kernel} kernel in SVC")

    _ = plt.show()

    pass

def debug_points(points):

    points = np.array(points)
    print(points.shape)

    plt.plot(points[:, 0], points[:, 1], c="red")
    plt.scatter(points[:, 0], points[:, 1], c="orange", s=10)
    plt.scatter([0], [0], c="green")
    plt.xlim(-6, 6)
    plt.ylim(-3, 10)

    plt.show()

    pass

def debug_pred(pred):
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            print(int(pred[i,j]), end="")
        print()
    
    return

def augment_dataset(X, mult=4, var=0.5):
    '''duplicates a dataset by multiplier and adds random noise to points'''
    X = np.concatenate([X] * mult).astype(np.float64)
    X += np.random.randn(X.shape[0], X.shape[1]) * var

    return X

def supplement_cones(cones: Cones):
    '''does in place, adds cones around origin to ground an SVM classifier'''
    cones.add_blue_cone(-1, 0, 0)
    cones.add_yellow_cone(1, 0, 0)

    cones.add_blue_cone(-1, 1, 0)
    cones.add_yellow_cone(1, 1, 0) 

def augment_cones(cones: Cones, mult=8, var=0.35):
    '''duplicates the cones by some multiplier and adds Gaussian noise with 
    some varaince'''
    blue, yellow, orange = cones.to_numpy()

    blue = augment_dataset(blue, mult=mult, var=var)
    yellow = augment_dataset(yellow, mult=mult, var=var)
    orange = augment_dataset(orange, mult=mult, var=var)

    return Cones.from_numpy(blue, yellow, orange)

def augment_dataset_circle(X, deg=20, radius=2):
    '''for each sample in X, adds additional points on circle of specified radius
    where circle lies on the first two dimensions of sample'''
    DEG_TO_RAD = np.pi / 180
    radian = deg * DEG_TO_RAD
    angles = np.arange(0, 2 * np.pi, step=radian)

    N = X.shape[0]

    # create duplicate points
    num_angles = angles.shape[0]
    X_extra = np.concatenate([X] * num_angles)
    angles = np.repeat(angles, N)

    X_extra[:, 0] += radius * np.cos(angles)
    X_extra[:, 1] += radius * np.sin(angles)
    return np.concatenate([X, X_extra])

def augment_cones_circle(cones: Cones, deg=20, radius=2):
    '''for each cone in ones, adds additional cones of same color on circle
    around cone with specified radius, separated by degrees'''
    blue, yellow, orange = cones.to_numpy()

    blue = augment_dataset_circle(blue, deg=deg, radius=radius) 
    yellow = augment_dataset_circle(yellow, deg=deg, radius=radius) 
    orange = augment_dataset_circle(orange, deg=deg, radius=radius) 
    
    return Cones.from_numpy(blue, yellow, orange)

def cones_to_xy(cones: Cones):
    '''Converts cones to a dataset representation (X, y) where y is vector
    of 0/1 labels where 0 corresponds to blue and 1 corresponds to yellow
    '''
    blue_cones, yellow_cones, orange_cones = cones.to_numpy()
    blue_cones[:, 2] = 0
    yellow_cones[:, 2] = 1

    data = np.vstack([blue_cones, yellow_cones]) 
    return data[:, :2], data[:, -1]

def get_spline_start_idx(points):
    '''gets index of point with lowest y-axis value in points'''
    # get points that are all the lowest
    min_y = np.min(points[:, 1])
    idxs = np.where(points[:, 1] == min_y)[0]

    # take the point that is closest to x = 0
    closest_x_idx = np.argmin(abs(points[idxs, 0]))
    return idxs[closest_x_idx]

def get_closest_point_idx(points, curr_point):
    '''gets index of point in points closest to curr_point and returns the dist'''
    assert(points.shape[1] == curr_point.shape[0])
    sq_dists = np.sum((points - curr_point) ** 2, axis=1)
    idx = np.argmin(sq_dists)
    return idx, np.sqrt(sq_dists[idx])

def sort_boundary_points(points, max_spline_length=17.5):
    '''sorts boundary points by starting from the lowest point and 
    iteratively takes closest point from iteration's current point
    takes approx: 7-8ms

    can additionallyn limit the number of points that are being ran on 
    '''

    # TODO: recalculating distances each iteration
    # might be better to calculate all pair-wise distances at start
    # and then iteratively removing from the dataset for each iteration

    # TODO: integrate spacing of 50cm here instead of repeating it
    # TODO: integrate maximum length of spline

    spline_length = 0
    points = np.array(points)
    sorted_points = []

    # start from the lowest point along the y-axis
    idx = get_spline_start_idx(points)
    curr_point = points[idx, :]
    rem_points = np.delete(points, idx, axis=0)

    print("in sort_boundary_points")

    # add starting point to sorted points
    sorted_points.append(curr_point)

    while rem_points.shape[0] > 0 and spline_length < max_spline_length:

        # find closest point to curr_point
        idx, d = get_closest_point_idx(rem_points, curr_point)
        spline_length += d

        # update iterates
        curr_point = rem_points[idx, :]
        rem_points = np.delete(rem_points, idx, axis=0)

        # add closest point to sorted points
        sorted_points.append(curr_point)

    print(spline_length < max_spline_length)
    print(spline_length)
    print(len(sorted_points))
    return np.array(sorted_points)

def cones_to_midline(cones: Cones):

    timer = Timer()
    timer.start("\taugtrain")
    blue_cones, yellow_cones, _ = cones.to_numpy()
    if len(blue_cones) == 0 or len(yellow_cones) == 0:
        return []
    
    # augment dataset to make it better for SVM training  
    supplement_cones(cones)
    cones = augment_cones_circle(cones, deg=10, radius=1.2) 

    X, y = cones_to_xy(cones)

    model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
    model.fit(X, y)

    timer.end("\taugtrain")

    if DEBUG_SVM:
        debug_svm(cones, X, y, model)

    # TODO: prediction takes 20-30+ ms, need to figure out how to optimize
    step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))

    svm_input = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(svm_input)
    Z = Z.reshape(xx.shape)


    if DEBUG_PRED:
        debug_pred(Z)

    timer.start("\tboundary")

    boundary_points = []
    for i in range(1, len(xx) - 1):
        for j in range(1, len(xx[0]) - 1):
            if Z[i][j] != Z[i-1][j-1] or Z[i][j] != Z[i+1][j-1]:
                boundary_points.append([xx[i][j], yy[i][j]])

    timer.end("\tboundary")

    # sort the points in the order of a spline
    boundary_points = sort_boundary_points(boundary_points)

    # downsample the points
    downsampled = []
    accumulated_dist = 0
    for i in range(1, len(boundary_points)):
        p1 = boundary_points[i]
        p0 = boundary_points[i-1]
        curr_dist = np.sqrt((p1[0] - p0[0])**2 + (p1[1] - p0[1])**2)
        accumulated_dist += curr_dist
        if np.abs(accumulated_dist - 0.5) < 0.1: # TODO: make this 50cm
            downsampled.append(p1)
            accumulated_dist = 0
        
        if accumulated_dist > 0.55:
            accumulated_dist = 0

    if DEBUG_POINTS:
        debug_points(boundary_points) 
    
    downsampled = np.array(list(downsampled))
    # print(downsampled)

    return downsampled