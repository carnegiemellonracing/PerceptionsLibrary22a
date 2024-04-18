'''SAPColorer.py

SAP stands for Seed And Propogate. This colorer will perform coloring as follows
    1. using a seeding fn, initialize starting seeds for blue and yellow cones
    2. using an propagating fn, iteratively find next blue and yellow cone
'''

from perc22a.predictors.utils.lidar.colorers.ColorerInterface import Colorer
import perc22a.predictors.utils.lidar.colorers.seed as seed

class SAPColorer(Colorer):

    def __init__(self, seed="naive", propogate="naive"):
        pass


    def color(self, cones_pos):
        pass