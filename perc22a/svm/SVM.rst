=========
SVM
=========

What is SVM?
------------

Support Vector Machines (SVMs) are a powerful supervised learning algorithm used for classification or for regression. 
SVMs are a discriminative classifier: that is, they draw a boundary between clusters of data. We use SVMs to find the
midline that best divides the blue and yellow cones from the Perceptions pipeline. This midline is the SVM decision boundary.
We use this midline as the reference line for the lane center, and the primary input to the Control pipeline.

How does it work?
-----------------

The SVM algorithm takes in the cones class, which contains the x and y coordinates of 2 types of cones, the blue and yellow cones (it also takes in
orange cones, but for the purposes of the current SVM, they are not a primary concern). The SVM algorithm then fits a line to the data, which is the
SVM decision boundary. This line is then used as the reference line for the lane center. 

@DAN - I think this is a good place to start. I think we should start by explaining the SVM algorithm, and then move on to the SVM implementation in the files
Also maybe pictures or diagrams? Idk just a thought