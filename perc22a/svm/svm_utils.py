
from perc22a.predictors.utils.cones import Cones

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

DEBUG_SVM = False

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

def augment_dataset(X, mult=4, var=0.5):
    X = np.concatenate([X] * mult).astype(np.float64)
    X += np.random.randn(X.shape[0], X.shape[1]) * var

    return X

def supplement_cones(cones: Cones):
    '''does in place'''
    cones.add_blue_cone(-1, 0, 0)
    cones.add_yellow_cone(1, 0, 0)

    cones.add_blue_cone(-1, 1, 0)
    cones.add_yellow_cone(1, 1, 0) 

def augment_cones(cones: Cones, mult=8, var=0.35):
    blue, yellow, orange = cones.to_numpy()

    blue = augment_dataset(blue, mult=mult, var=var)
    yellow = augment_dataset(yellow, mult=mult, var=var)
    orange = augment_dataset(orange, mult=mult, var=var)

    return Cones.from_numpy(blue, yellow, orange)

def augment_dataset_circle(X, deg=20, radius=2):
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
    blue, yellow, orange = cones.to_numpy()

    blue = augment_dataset_circle(blue, deg=deg, radius=radius) 
    yellow = augment_dataset_circle(yellow, deg=deg, radius=radius) 
    orange = augment_dataset_circle(orange, deg=deg, radius=radius) 
    
    return Cones.from_numpy(blue, yellow, orange)

def cones_to_xy(cones: Cones):
    blue_cones, yellow_cones, orange_cones = cones.to_numpy()
    blue_cones[:, 2] = 0
    yellow_cones[:, 2] = 1

    data = np.vstack([blue_cones, yellow_cones]) 
    return data[:, :2], data[:, -1]

def cones_to_midline(cones: Cones):

    blue_cones, yellow_cones, _ = cones.to_numpy()
    if len(blue_cones) == 0 or len(yellow_cones) == 0:
        return []

    # augment dataset to make it better for SVM training  
    
    # TODO: currently no augmentations - use deg=10 and radius=1-2 ish (maybe 1.5)
    cones = augment_cones_circle(cones, deg=10, radius=1.2) 
    supplement_cones(cones)

    X, y = cones_to_xy(cones)

    model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
    model.fit(X, y)

    if DEBUG_SVM:
        debug_svm(cones, X, y, model)

    step = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                        np.arange(y_min, y_max, step))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    boundary_points = []
    for i in range(1, len(xx)):
        for j in range(1, len(xx[0])):
            if Z[i][j] != Z[i-1][j-1]:
                boundary_points.append([xx[i][j], yy[i][j]])

    def norm_func(x): return np.sqrt(x[0]**4 + x[1]**2)
    boundary_points.sort(key=norm_func)
    # print(boundary_points)

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
    
    downsampled = np.array(list(downsampled))
    # print(downsampled)

    return downsampled