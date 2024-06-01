Utils
=====

What is Utils?
--------------

Utils is a collection of classes and functions used across the predictors. 
There is a lot of code that is common to all predictors, and it is stored here.

Cones.py
--------

This file contains the class `Cones`, which is used to store the cones of a predictor.
The cones are stored in a dictionary, where the keys are the names of the cones and the values are the cones themselves.
The cones are stored as a list of tuples, where each tuple contains the name of the cone and the value of the cone.
The cones are stored in the order they were added to the dictionary.
The cones have a color attribute that is used to color the cones for side recognition.

ConeState.py
------------

This file defines the ConeState class, which is used to
maintain and update the state of cones detected by a 
LiDAR sensor. The class integrates prior color estimates 
of cones with new incoming data to refine the color
predictions of cones over time. The implementation 
uses the Iterative Closest Point (ICP) algorithm to 
determine correspondences between previously seen cones
and new cones.

stereo.py
---------

Stereo.py contains certain functions that are useful during the stereo pipeline.

- calc_box_center: calcualte the center of a bounding box, 

- get_object_depth: getting the object depth from the stereo depth map

- get_world_coords: getting the world coordinates of the object from the stereo depth map

- get_cone_color: simple function to get the color of the cone by color averaging and thresholding



icp.py
------

This file contains the implementation of the Iterative Closest Point (ICP) algorithm.
The ICP algorithm is used to determine correspondences between previously seen cones and new cones.
We use the ICP algorithm to determine the transformation that aligns the previously seen cones with the new cones.

.. raw:: html

    <h1 align="center">
      <div>
        <div style="position: relative; padding-bottom: 0%; overflow: hidden; max-width: 100%; height: auto;">
          <iframe width="450" height="300" src="https://www.youtube.com/watch?v=uzOCS_gdZuM" frameborder="1" allowfullscreen></iframe>
        </div>
      </div>
    </h1>