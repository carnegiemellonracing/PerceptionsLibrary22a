from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.predictors.utils.cones import Cones

import numpy as np
import time

V_START = 2 # starting y-value of closest row of cones from the car
V_SPACE = 3 # spacing between rows of cones
N_ROWS = 4  # number of rows of cones
WIDTH = 4.5 # spacing betweeen blue and yellow cone in a single row

PERIOD = 10 # seconds it takes for a single cycle of cone motion to happen
MAG = 1     # amount of spacing in meters that cones oscillate
SCALE = 0.5   # add MAG * SCALE to magnitude for row

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

        mag = MAG * (SCALE * np.arange(0, N_ROWS) + 1)
        blue[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))
        yellow[:, 0] += mag * np.cos((delta / PERIOD) * (np.pi * 2))

        
        cones = Cones.from_numpy(blue, yellow, np.empty((0,3)))
        return cones