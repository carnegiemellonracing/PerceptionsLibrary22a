Stereo Camera
=============

What is Stereo Camera?
-----------------------

Stereo Camera is a combination of camera and a depth sensor. It is used 
to capture the depth information of the scene. It allows you to run vision algorthms
for object detection, tracking, and recognition in combination with accurate distance
information. The stereo camera we use is a ZED camera from Stereolabs. 

How does it work?
------------------

The ZED camera uses two sensors to capture the scene. The two sensors 
are placed at a distance from each other. The distance between the two sensors
is called the baseline. The ZED camera uses the disparity between the two images
to calculate the depth of the scene. The disparity is the difference between the
two images. The disparity is inversely proportional to the depth of the scene. 
The ZED camera uses the disparity to calculate the depth of the scene.

Stereo Camera Pipeline
-----------------------

The stereo camera pipeline consists of the following steps:

1. Take in the information from the ZED sensor.

2. Using YOLOV5, detect the cones in the scene.

3. With the detected cones, calculate the depth of the cones.

4. Using the depth information, calculate the distance of the cones from the camera.

5. Transform the distance of the cones from the camera to the distance of the cones from the car.

.. raw:: html

    <h1 align="center">
      <div>
        <div style="position: relative; padding-bottom: 0%; overflow: hidden; max-width: 100%; height: auto;">
          <iframe width="450" height="300" src=../../../documentation/stereo/yolov8.mp4 frameborder="1" allowfullscreen></iframe>
        </div>
      </div>
    </h1>
