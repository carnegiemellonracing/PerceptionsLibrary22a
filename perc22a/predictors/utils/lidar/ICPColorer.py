
from perc22a.predictors.utils.cones import Cones

class ICPColorer:

    def __init__(self):
        self.prev_cones = None

        pass


    def recolor(self, cones: Cones):
        if self.prev_cones is None:
            self.prev_cones = cones
            return cones

        # save current cones for next iteration as prev_cones 
        curr_cones = cones.copy()
        
        self.prev_cones.plot2d(show=False, label="p")
        cones.plot2d(show=True, label="c")

        # create some new cones
        self.prev_cones = curr_cones

        return cones