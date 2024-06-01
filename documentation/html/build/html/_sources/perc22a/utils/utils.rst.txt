======
Utils
======


What is Utils?
---------------

Utils are gerenal pieces of code we use throughout the project. They are 
not directly related to the project's main functionality, but they 
are used to make the code more readable, maintainable and testable. 

There are 3 main parts to our utils.

1. Timer
======== 
This file is used to time the execution of the code. The main functionality of it
is to time the execution of the code and either return it or print the time taken to the console.
To use this file, create a new instance of the timer object, call start with the name
of the timer and then call end with the same name. This will return the time taken in seconds.
If you use the ret flag on the end function, it will print the time taken in seconds onto the console.

2. ConeSim
==========

The second file is the ConeSim file. This file is used to simulate the cones on a track.
Designed for testing purposes, it allows you to publish fake cone information that changes with time
from the cone node allowing for further testing of the pipelnine. To use this file, create a new instance
of the ConeSim object with all the parameters you need for the cones to simulate. Then call the
get cones function to get the cones at the current time. This will return a cones object of the cones which
can be used in the pipeline.

3. MotionInfo
=============

Lastly, there is the motion info file. This file is used to take in informaiton about the cars movement
and predict its future position. This file is primarily used in the motion modeling code we have
for the LiDAR pipelien. The motion modeling code allows us to accuratly color cones and update the cones positions we have 
seen in between cycles of the LiDAR. To use this file, create a new instance of the MotionInfo object with the