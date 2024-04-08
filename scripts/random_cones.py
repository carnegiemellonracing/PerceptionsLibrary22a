from perc22a.predictors.utils.vis.Vis2D import Vis2D
from perc22a.predictors.utils.cones import Cones
from perc22a.utils.ConeSim import ConeSim

import numpy as np
import time

def main():

    vis = Vis2D()

    cone_sime = ConeSim() 

    while True:

        vis.set_cones(cone_sime.get_cones())
        vis.update()


if __name__ == "__main__":
    main()