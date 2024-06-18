'''MotionInfo.py

Class for maintaining the necessary information for transforming from GPS
to world coordinates including the timestamp of the position

Functions:
    - __init__: initializes the MotionInfo class
    - get_sensor_to_global: computes the rotation matrix that converts points 
        positioned in the heading of the car into global heading
    - get_global_to_sensor: computes the rotation matrix that converts points 
        positioned in global heading to the heading of the car
    - get_translation_to: computes the translation between two motion info classes
    - model_motion_to: motion models points in frame of current motion info to 
        frame of other MotionInfo that describes a different timestep
'''

import numpy as np

class MotionInfo:

    def __init__(self, linear_twist, quat, time_ns):
        '''Initialize GPS information necessary for modelling motion of object

        Linear twist, quaternion, and time in nanoseconds of PGS data 
        '''

        assert(linear_twist.shape == (3, 1))
        assert(quat.shape == (4,))

        self.linear_twist = linear_twist
        self.quat = quat
        self.time_ns = time_ns

        return
    
    def get_sensor_to_global(self):
        '''computes the rotation matrix that converts points 
        positioned in the heading of the car into global heading'''

        q0, q1, q2, q3 = self.quat
        return np.asarray([
            [2*(q0**2 + q1**2) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2) - 1, 2*(q2*q3 - q0*q1)],
            [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2) - 1],
        ])
    
    def get_global_to_sensor(self):
        '''computes the rotation matrix that converts points positioned
        in global heading to the heading of the car'''
        return np.linalg.inv(self.get_sensor_to_global())
    
    def get_translation_to(self, other_mi):
        '''computes the translation between two motion info classes'''

        dt = (other_mi.time_ns - self.time_ns) / 1e9
        avg_linear_twist = (other_mi.linear_twist + self.linear_twist) / 2

        return avg_linear_twist * dt
    
    def model_motion_to(self, points, other_mi):
        '''Motion models points in frame of current motion info to
        frame of other MotionInfo that describes a differnet timestep

        Arguments:
            - points: (n x 3) np.ndarray of points in current MotionInfo frame
            - other_mi: MotionInfo to transform point into frame of
        '''
        assert(points.ndim == 2 and points.shape[1] == 3)
        if points.shape[0] == 0:
            return points
        
        # (1) convert heading of current MotionInfo to global frame
        # curr_sensor_to_global = 3x3
        # curr_cones = Nx3
        # curr_sensor_to_global @ curr_cones.T = 3xN 
        curr_sensor_to_global = self.get_sensor_to_global()
        curr_in_global = (curr_sensor_to_global @ points.T)

        # (2) translate the points to the other MotionInfo's global frame
        # curr_in_global = 3xN
        # translation = 3x1, broadcasted N times in axis=1
        # future_in_global = 3xN
        # NOTE: if translation is +, then point positions must decrease
        to_other_translation = self.get_translation_to(other_mi)
        other_in_global = curr_in_global - to_other_translation

        #(3)
        # future_global_to_sensor = 3x3
        # future_in_global = 3xN
        # future_global_to_sensor @ future_in_global = 3xN
        # curr_cones_in_future = Nx3
        other_global_to_sensor = other_mi.get_global_to_sensor()
        points_in_other = (other_global_to_sensor @ other_in_global).T
        return points_in_other