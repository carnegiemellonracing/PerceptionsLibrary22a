
from perc22a.predictors.utils.cones import Cones

class ICPColorer:

    def __init__(self):
        self.prev_cones = None

        pass


    def recolor(self, cones: Cones):
        if self.prev_cones is None:
            self.prev_cones = cones
            return cones

        return cones