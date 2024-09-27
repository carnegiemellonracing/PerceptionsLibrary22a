
# general imports
import numpy as np
# ROS2 msg to python datatype conversions
import perceptions.ros.utils.conversions as conv

# perceptions Library visualization functions (for 3D data)


def transform_left_lidar(points):

    tf_mat_left = np.array([[   0.76604444,    -0.64278764,    0.        ,  -0.18901      ],
                            [  0.64278764,    0.76604444,    0.        , 0.15407      ],
                            [   0.        ,    0.        ,    1.        ,    0.        ],
                            [   0.        ,    0.        ,    0.        ,    1.        ]])
    column = np.array([[1.0] for _ in range(len(points))])
    points = np.hstack((points, column))
    points = np.matmul(points, tf_mat_left.T)

    # # multiply transformation matrix with points
    # points = points[:, [1, 0, 2]]
    # points[:, 0] *= -1
    # self.points = points[:, :3]
        
    return points

def transform_right_lidar(points):

    tf_mat_right = np.array([[  0.76604444,  0.64278764,   0.        , -0.16541      ],
                                [  -0.64278764,   0.76604444,   0.        , -0.12595      ],
                                [  0.        ,   0.        ,   1.        ,   0.        ],
                                [  0.        ,   0.        ,   0.        ,   1.        ]])
    column = np.array([[1.0] for _ in range(len(points))])
    points = np.hstack((points, column))
    points = np.matmul(points, tf_mat_right.T)
    # points = np.transpose(np.matmul(tf_mat_right, np.transpose(points)))[:, :3]

    # points = points[:, [1, 0, 2]]
    # points[:, 0] *= -1
    # self.points = points[:, :3]
    
    return points