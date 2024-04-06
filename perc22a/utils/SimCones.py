from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.predictors.utils.cones import Cones

import numpy as np
import time

V_START = 2
V_SPACE = 3
N_ROWS = 4
WIDTH = 4.5

PERIOD = 10
MAG = 1
SCALE = 2

class SimCones:

    def __init__(self):

        self.vis = Vis2D()

        vs = ((np.arange(0, N_ROWS) * V_SPACE) + V_START).reshape((N_ROWS, 1))
        orig_blue = np.concatenate([np.full((N_ROWS, 1), -WIDTH), vs, np.full((N_ROWS, 1), 0)], axis=1)
        orig_yellow = np.concatenate([np.full((N_ROWS, 1), WIDTH), vs, np.full((N_ROWS, 1), 0)], axis=1)
        self.s = time.time()

        self.orig_blue = orig_blue.astype(np.float64)
        self.orig_yellow = orig_yellow.astype(np.float64)

        return

    def get_cones(self):

        delta = time.time() - self.s

        blue = np.array(self.orig_blue)
        yellow = np.array(self.orig_yellow)

        mag = MAG * (np.arange(0, N_ROWS) + 1)
        blue[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))
        yellow[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))

        
        cones = Cones.from_numpy(blue, yellow, np.empty((0,3)))
        return cones

def main():

    vis = Vis2D()

    vs = ((np.arange(0, N_ROWS) * V_SPACE) + V_START).reshape((N_ROWS, 1))
    orig_blue = np.concatenate([np.full((N_ROWS, 1), -WIDTH), vs, np.full((N_ROWS, 1), 0)], axis=1)
    orig_yellow = np.concatenate([np.full((N_ROWS, 1), WIDTH), vs, np.full((N_ROWS, 1), 0)], axis=1)
    s = time.time()

    orig_blue = orig_blue.astype(np.float64)
    orig_yellow = orig_yellow.astype(np.float64)

    while True:

        delta = time.time()

        blue = np.array(orig_blue)
        yellow = np.array(orig_yellow)

        mag = MAG * (np.arange(0, SCALE) + 1)
        blue[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))
        yellow[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))

        
        cones = Cones.from_numpy(blue, yellow, np.empty((0,3)))

        vis.set_cones(cones)
        vis.update()


    pass

if __name__ == "__main__":
    main()