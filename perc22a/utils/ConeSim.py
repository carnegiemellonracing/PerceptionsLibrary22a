from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.predictors.utils.cones import Cones

import numpy as np
import time


class ConeSim:

    def __init__(self, v_start=2, v_space=3, n_rows=4, width=4.5, period=10, mag=1, scale=0.5):
        '''
        Arguments:
            v_start (float):    starting y-value of closest row of cones from the car
            v_space (float):    spacing between rows of cones
            n_rows (int):       number of rows of cones
            width (float):      spacing betweeen blue and yellow cone in a single row
            period (float):     seconds it takes for a single cycle of cone motion to happen
            mag (float):        amount of spacing in meters that cones oscillate
            scale (float):      add MAG * SCALE to magnitude for row
        '''

        self.v_start = v_start
        self.v_space = v_space
        self.n_rows = n_rows
        self.width = width
        self.period = period
        self.mag = mag
        self.scale = scale

        self.vis = Vis2D()

        vs = ((np.arange(0, self.n_rows) * self.v_space) + self.v_start).reshape((self.n_rows, 1))
        orig_blue = np.concatenate([np.full((self.n_rows, 1), -self.width), vs, np.full((self.n_rows, 1), 0)], axis=1)
        orig_yellow = np.concatenate([np.full((self.n_rows, 1), self.width), vs, np.full((self.n_rows, 1), 0)], axis=1)
        self.s = time.time()

        self.orig_blue = orig_blue.astype(np.float64)
        self.orig_yellow = orig_yellow.astype(np.float64)

        return

    def get_cones(self):
        '''
        Returns: Cones() data type representing cones at that time
        '''

        delta = time.time() - self.s

        blue = np.array(self.orig_blue)
        yellow = np.array(self.orig_yellow)

        mag = self.mag * (self.scale * np.arange(0, self.n_rows) + 1)
        blue[:, 0] += mag * np.cos((delta / self.period) * (np.pi * 2))
        yellow[:, 0] += mag * np.cos((delta / self.period) * (np.pi * 2))

        
        cones = Cones.from_numpy(blue, yellow, np.empty((0,3)))
        return cones