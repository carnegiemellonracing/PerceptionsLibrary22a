LiDAR
=====

What is LiDAR?
--------------

LiDAR is a remote sensing technology that measures distance by illuminating 
a target with a laser and analyzing the reflected light. LiDAR is used to 
create high-resolution maps. We use LiDAR as the primary sensor to detect
obstacles in the environment. The LiDAR sensor we use is the `HESAI AT128 Solid State LiDAR. <"https://www.hesaitech.com/product/at128/"/>`_
The LiDAR is one of the best sensors for obstacle detection and is widely used in autonomous vehicles.

LiDAR Point Cloud conversion
----------------------------

The LiDAR sensor provides a 3D point cloud of the environment. The point
cloud is a set of data points in space. Each point represents the distance
from the sensor to an object in the environment. The point cloud is a 3D
representation of the environment, where each point has an x, y, and z
coordinate.

Our LiDAR prediction stack:
---------------------------

1. The LiDAR sensor provides a 3D point cloud of the environment. We take in the point cloud data and then transform it to an np array.

2. Then we take in the information and transform it to the car frame.

3. Then we take the information and filter out the points that are on the ground. Additionally, we filter out points that are too far away.

4. We then take the filtered points and cluster them to find the obstacles in the environment.

.. image :: ../../../../../lidar/pointcloud.png
    :align: center