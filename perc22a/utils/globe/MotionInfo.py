'''MotionInfo.py

Class for maintaining the necessary information for transforming from GPS
to world coordinates including the timestamp of the position
'''


class MotionInfo:

    def __init__(self, linear_twist, quat, time_ns):

        assert(linear_twist.shape == (3, 1))
        assert(quat.shape == (4,))

        self.linear_twist = linear_twist
        self.quat = quat
        self.time_ns = time_ns

        return
