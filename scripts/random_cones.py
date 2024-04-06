from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.predictors.utils.cones import Cones
from perc22a.utils.SimCones import SimCones

import numpy as np
import time

V_START = 2
V_SPACE = 3
N_ROWS = 2
WIDTH = 4.5

PERIOD = 10
MAG = 1
SCALE = 2

def main():

    vis = Vis2D()

    cone_sime = SimCones() 

    while True:

        vis.set_cones(cone_sime.get_cones())
        vis.update()


if __name__ == "__main__":
    main()