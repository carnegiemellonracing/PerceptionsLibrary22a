"""cones.py

Create Cones class that has a consistent representation of the cones that
Perceptions will be returning.

All Predictor algorithm's .predict(...) method should return this datt type
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    def __len__(self):
        return sum([len(cones) for cones in [self.blue_cones, self.yellow_cones, self.orange_cones]])

    def add_blue_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.blue_cones.append([x, y, z])
        return self.blue_cones

    def add_yellow_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.yellow_cones.append([x, y, z])
        return self.yellow_cones

    def add_orange_cone(self, x, y, z):
        """(x, y, z) are cone position in meters"""
        self.orange_cones.append([x, y, z])
        return self.orange_cones
    
    def add_cones(self, cones):
        """adds cones from another Cone object"""
        self.blue_cones += cones.blue_cones
        self.yellow_cones += cones.yellow_cones
        self.orange_cones += cones.orange_cones

        return
        
    def filter(self, function_filter):
        filtered_cone_list_blue = []
        filtered_cone_list_orange = []
        filtered_cone_list_yellow = []

        for cone in self.blue_cones:
            if function_filter(cone): filtered_cone_list_blue.append(cone)
        for cone in self.orange_cones:
            if function_filter(cone): filtered_cone_list_orange.append(cone)
        for cone in self.yellow_cones:
            if function_filter(cone): filtered_cone_list_yellow.append(cone)

        self.blue_cones = filtered_cone_list_blue
        self.orange_cones = filtered_cone_list_orange
        self.yellow_cones = filtered_cone_list_yellow

        return
    
    def map(self, function_map):
        filtered_cone_list_blue = []
        filtered_cone_list_orange = []
        filtered_cone_list_yellow = []

        for cone in self.blue_cones:
            filtered_cone_list_blue.append(function_map(cone))
        for cone in self.orange_cones:
            filtered_cone_list_orange.append(function_map(cone))
        for cone in self.yellow_cones:
            filtered_cone_list_yellow.append(function_map(cone))

        self.blue_cones = filtered_cone_list_blue
        self.orange_cones = filtered_cone_list_orange
        self.yellow_cones = filtered_cone_list_yellow
        

    def to_numpy(self):
        """Returns all cones added to Cones objet

        Columns of all arrays are x, y, and z in that order
        Return:
            blue_cones_arr: (N_b, 3) array
            yellow_cones_arr: (N_y, 3) array
            orange_cones_arr: (N_o, 3) array
        """
        blue_cones_arr = np.array(self.blue_cones).reshape(-1, 3)
        yellow_cones_arr = np.array(self.yellow_cones).reshape(-1, 3)
        orange_cones_arr = np.array(self.orange_cones).reshape(-1, 3)

        return blue_cones_arr, yellow_cones_arr, orange_cones_arr

    @classmethod    
    def from_numpy(cls, blue_cones_arr, yellow_cones_arr, orange_cones_arr):
        """Converts numpy array of points to Cone object

        Class method so call directly from class `Cones.from_numpy(...)
        
        Arguments:
            blue_cones_arr: (N_b, 3) numpy array
            yellow_cones_arr: (N_y, 3) numpy array
            orange_cones_arr: (N_o, 3) numpy array
        Returns:

        """
        cones = cls()

        for i in range(blue_cones_arr.shape[0]):
            cones.add_blue_cone(
                blue_cones_arr[i, 0], blue_cones_arr[i, 1], blue_cones_arr[i, 2]
            )
        for i in range(yellow_cones_arr.shape[0]):
            cones.add_yellow_cone(
                yellow_cones_arr[i, 0], yellow_cones_arr[i, 1], yellow_cones_arr[i, 2]
            )
        for i in range(orange_cones_arr.shape[0]):
            cones.add_orange_cone(
                orange_cones_arr[i, 0], orange_cones_arr[i, 1], orange_cones_arr[i, 2]
            )

        return cones
    
    def plot2d(self, ax=None, show=True, title="", label=""):

        blue_cones, yellow_cones, orange_cones = self.to_numpy()

        if ax is None:
            ax = plt.gca()

        ax.scatter(blue_cones[:, 0], blue_cones[:, 1], c="blue")
        ax.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c="gold")
        ax.scatter(orange_cones[:, 0], orange_cones[:, 1], c="orange")
        ax.scatter([0], [0], c="red")
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)

        cones = np.concatenate([blue_cones, yellow_cones, orange_cones])
        for i in range(cones.shape[0]):
            ax.annotate(label, (cones[i, 0], cones[i, 1]))

        if show:
            plt.show()

        return
    
    def copy(self):
        '''returns deep copy of current cone object as '''
        return Cones.from_numpy(*self.to_numpy())