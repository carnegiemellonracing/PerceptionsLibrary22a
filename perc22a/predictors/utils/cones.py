"""cones.py

Create Cones class that has a consistent representation of the cones that
Perceptions will be returning.

All Predictor algorithm's .predict(...) method should return this datt type
"""

import numpy as np

# TODO: deprecate add_*_points_cone, instead should just loop and add


class Cones:
    def __init__(self):
        self.blue_cones = []
        self.yellow_cones = []
        self.orange_cones = []

        return

    def _repr_cones(self, cones):
        """cones is a list of 3-tuples"""
        if len(cones) == 0:
            return "\tNo cones\n"
        return (
            "\n".join([f"\tx: {x:6.2f}, y: {y:6.2f}, z: {z:6.2f}" for x, y, z in cones])
            + "\n"
        )

    def __repr__(self):
        blue_str = self._repr_cones(self.blue_cones)
        yellow_str = self._repr_cones(self.yellow_cones)
        orange_str = self._repr_cones(self.orange_cones)

        s = "Cones".center(20, "-") + "\n"
        s += f"Blue ({len(self.blue_cones)} cones)\n"
        s += blue_str
        s += f"Yellow ({len(self.yellow_cones)} cones)\n"
        s += yellow_str
        s += f"Orange ({len(self.orange_cones)} cones)\n"
        s += orange_str
        return s

    def __str__(self):
        return self.__repr__()

    def add_blue_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.blue_cones.append([x, y, z])
        return self.blue_cones
    
    def add_blue_points_cone(self, points):
        for point in points:
            self.blue_cones.append(point)
        return self.blue_cones

    def add_yellow_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.yellow_cones.append([x, y, z])
        return self.yellow_cones
    
    def add_yellow_points_cone(self, points):
        for point in points:
            self.yellow_cones.append(point)
        return self.yellow_cones

    def add_orange_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.orange_cones.append([x, y, z])
        return self.orange_cones
    
    def add_orange_points_cone(self, points):
        for point in points:
            self.orange_cones.append(point)
        return self.orange_cones
    

    def get_cones(self):
        """Returns all cones added to Cones objet

        Columns of all arrays are x, y, and z in that order
        Return:
            blue_cones_arr: (N_b, 3) array
            yellow_cones_arr: (N_y, 3) array
            orange_cones_arr: (N_o, 3) array
        """
        blue_cones_arr = np.array(self.blue_cones)
        yellow_cones_arr = np.array(self.yellow_cones)
        orange_cones_arr = np.array(self.orange_cones)

        return blue_cones_arr, yellow_cones_arr, orange_cones_arr
