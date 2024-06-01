=================
Merging pipelines
=================

What is merging pipeliens?
----------------------------

Merging pipelines is a process of combining two or more pipelines into a single
pipeline. This is useful when you have multiple pipelines that you want to run 
together. In our case, we want to merge the cone data we get from our LiDAR and
camera pipelines. 

Merging Logic
--------------

The merging logic is simple. We will take the cone data from the LiDAR pipeline
and use that as the ground truth for our position data. We will then take the
cone data from the camera and use the color information as the ground truth about the color.
Any cones that are detected by only the LiDAR or only the camera will be taken into 
account as well, but the cones that are detected by both will have a more accurate
position and color. We will then merge the two sets of cones into a single set of cones and that
is what we will use for our final cone data.