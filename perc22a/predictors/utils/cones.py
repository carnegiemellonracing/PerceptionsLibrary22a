"""cones.py

Create Cones class that has a consistent representation of the cones that
Perceptions will be returning.

All Predictor algorithm's .predict(...) method should return this datt type
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

# TODO: deprecate add_*_points_cone, instead should just loop and add


class Cones:
    def __init__(self, quat=None, twist=None):
        self.blue_cones = []
        self.yellow_cones = []
        self.orange_cones = []

        self.quat = quat
        self.twist = twist

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
    
    def merge_cones(self, new_cones, max_correspondence_dist=0.5):

        dest_cone_arr = self.to_arr()
        src_cone_arr = new_cones.to_arr()

        # train kNN model to determine nearest neighbors
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dest_cone_arr[:, :2])
        distances, indices = neigh.kneighbors(src_cone_arr[:, :2], return_distance=True)

        indices = np.concatenate([np.arange(indices.size).reshape((-1,1)), indices], axis=1)
        distances = distances.reshape(-1)

        # get correspondences for cones that are only nearby
        correspondences = indices[distances < max_correspondence_dist, :]

        # get cones from source/new cones that don't have correspondence and add them back in
        new_uncorr_mask = np.ones(src_cone_arr.shape[0], dtype=bool)
        new_uncorr_mask[correspondences[:, 0]] = False
        
        uncorr_new_cone_arr = src_cone_arr[new_uncorr_mask, :]

        result_cone_arr = np.concatenate([dest_cone_arr, uncorr_new_cone_arr])
        return Cones.from_arr(result_cone_arr)
        
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

    def to_arr(self):
        """Converts all cones into a single numpy array
        
        Color column maps blue to 0, yellow to 1, and orange to 2
        Return:
            arr: (N_b + N_y + N_o, 4) where the four columns are r, g, b, c
        """

        blue, yellow, orange = self.to_numpy()
        blue = np.concatenate([blue, 0 + np.zeros((blue.shape[0], 1))], axis=1)
        yellow = np.concatenate([yellow, 1 + np.zeros((yellow.shape[0], 1))], axis=1)
        orange = np.concatenate([orange, 2 + np.zeros((orange.shape[0], 1))], axis=1)

        cone_arr = np.concatenate([blue, yellow, orange], axis=0)
        return cone_arr

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
        cones = cls(None, None)

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
    
    @classmethod
    def from_arr(cls, cone_arr):
        blue = cone_arr[cone_arr[:, 3] == 0, :]
        yellow = cone_arr[cone_arr[:, 3] == 1, :]
        orange = cone_arr[cone_arr[:, 3] == 2, :]

        return Cones.from_numpy(blue, yellow, orange)

    def get_sensor_to_global(self):
        q0 = self.quat.quaternion.w
        q1 = self.quat.quaternion.x
        q2 = self.quat.quaternion.y
        q3 = self.quat.quaternion.z
        return np.asarray([
            [2*(q0**2 + q1**2) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2) - 1],
        ])

    def get_global_to_sensor(self):
        return np.linalg.inv(self.get_sensor_to_global())
   
    def get_car_translation(self, future_cones):
        dt = (future_cones.twist.header.stamp.nanosec - self.twist.header.stamp.nanosec) / 1e9
        return np.asarray([
            [(future_cones.twist.twist.linear.x + self.twist.twist.linear.x)/2],
            [(future_cones.twist.twist.linear.y + self.twist.twist.linear.y)/2],
            [(future_cones.twist.twist.linear.z + self.twist.twist.linear.z)/2],
        ]) * dt

    def transform_to_future(self, future_cones):
        # Transformation Algorithm
        # 1) transform curr cones to global frame
        # 2) figure out translation between curr -> future and add to curr
        # 3) multiply by the inverse of future quat to get those into future's frame
        blue_cones_arr, yellow_cones_arr, orange_cones_arr = self.to_numpy()

        def transform(curr_cones):
            if len(curr_cones) == 0:
                return curr_cones
            
            curr_sensor_to_global = self.get_sensor_to_global()

            # (1)
            print(f"matrix: {curr_sensor_to_global}")
            print(f"cones: {curr_cones}")
            # curr_sensor_to_global = 3x3
            # curr_cones = Nx3
            # curr_sensor_to_global @ curr_cones.T = 3xN 
            curr_in_global = (curr_sensor_to_global @ curr_cones.T) # ARE THESE THE CORRECT FRAMES

            #(2)
            # curr_in_global = 3xN
            # translation = 3x1, broadcasted N times in axis=1
            # future_in_global = 3xN
            translation = self.get_car_translation(future_cones)
            future_in_global = curr_in_global - translation # car moves forward, so cones move backward

            #(3)
            # future_global_to_sensor = 3x3
            # future_in_global = 3xN
            # future_global_to_sensor @ future_in_global = 3xN
            # curr_cones_in_future = Nx3
            future_global_to_sensor = future_cones.get_global_to_sensor()
            curr_cones_in_future = (future_global_to_sensor @ future_in_global).T
            return curr_cones_in_future

        blue_cones_transformed = transform(blue_cones_arr)
        yellow_cones_transformed = transform(yellow_cones_arr)
        orange_cones_transformed = transform(orange_cones_arr)
        return Cones.from_numpy(blue_cones_transformed, yellow_cones_transformed, orange_cones_transformed)
        
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
        result = Cones.from_numpy(*self.to_numpy())
        result.quat = self.quat
        result.twist = self.twist
        return result