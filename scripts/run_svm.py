from perc22a.predictors.stereo.YOLOv5Predictor import YOLOv5Predictor
from perc22a.data.utils.dataloader import DataLoader

import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import torch

def main():

    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    backend = matplotlib.get_backend()
    sp = YOLOv5Predictor()
    matplotlib.use(backend)
    # plt.scatter([1, 2], [1, 2])
    # plt.show()

    for i in range(50, len(dl)):

        cones = sp.predict(dl[i])
        blue, yellow, _ = cones.to_numpy()

        blue[:, 2] = -1
        yellow[:, 2] = 1

        matplotlib.use(backend)
        # plt.scatter([1, 2], [1, 2])
        

        if len(blue) > 0 and len(yellow) > 0:
            data = np.vstack([blue, yellow])
            X = data[:, :2]
            y = data[:, 2]



            model = svm.SVC(kernel='poly', degree=3, C=10, coef0=1.0)
            model.fit(X, y)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
            #                     np.arange(y_min, y_max, 0.01))

            # Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            # Z = Z.reshape(xx.shape)

            # plt.contourf(xx, yy, Z, alpha=0.8)

            # Settings for plotting
            _, ax = plt.subplots(figsize=(4, 3))
            # x_min, x_max, y_min, y_max = -3, 3, -3, 3
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

            # Plot decision boundary and margins
            common_params = {"estimator": model, "X": X, "ax": ax}
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

            ax.scatter(
                model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=250,
                facecolors="none",
                edgecolors="k",
            )


            plt.scatter(yellow[:,0], yellow[:,1], c="orange")
            plt.scatter(blue[:,0], blue[:,1], color="blue")
            # plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

            # model = svm.SVC(kernel='linear', C=100)
            # model.fit(X, y)

            # coef = model.coef_.reshape(-1)
            # coef0 = model.intercept_[0]

            # # import pdb; pdb.set_trace()

            # yint = -coef0 / coef[1]
            # slope = -coef[0] / coef[1]
            # plt.axline((0, yint), slope=slope)

            # b0 + b^Tx = 0
            # b0 + b1 x1 + b2 x2 = 0
            # x2 = (-b1 / b2) x1 + (-b0 / b2)

        plt.show()


if __name__ == "__main__":
    main()