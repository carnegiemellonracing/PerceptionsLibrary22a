=========
SVM
=========

What is SVM?
------------

Support Vector Machines (SVMs) are a powerful supervised learning algorithm used for classification or for regression. 
SVMs are a discriminative classifier: that is, they draw a boundary between clusters of data. We use SVMs to find the
midline that best divides the blue and yellow cones from the Perceptions pipeline. This midline is the SVM decision boundary.
We use this midline as the reference line for the lane center, and the primary input to the Control pipeline.

The SVM algorithm is part of the the path planning pipeline, which is responsible for controlling the car. The SVM algorithm, however, is
written in Python and for the purposes of documentation, is included in the Perceptions documentation.

How does it work?
-----------------

The SVM algorithm takes in the cones class, which contains the x and y coordinates of 2 types of cones, the blue and yellow cones (it also takes in
orange cones, but for the purposes of the current SVM, they are not a primary concern). The SVM algorithm then fits a line to the data, which is the
SVM decision boundary. This line is then used as the reference line for the lane center. 

Our Perceptions pipeline outputs the x and y coordinates of the blue and yellow cones. These x and y coordinates are given in global frame
every 0.1 seconds. The SVM algorithm takes in these x and y coordinates and fits a line to the data. It does this by taking the mesh frame from the
cones and passing a line through the center of the mesh frame. This line is the SVM decision boundary. After we get a rough estimate for 
what the decision boundary is between the 2 colors of cones, we then downsample the points on the line to provide an equidistant line through
all of the cones in the mesh frame as a better center line for the car.

Additionally, we also provide 2 base cones behind the cars original location to provide the SVM with an understanding of color and distance.
After we get the decision boundaries from the SVM, we then use the decision boundaries to improve coloring for our LiDAR algorithm.