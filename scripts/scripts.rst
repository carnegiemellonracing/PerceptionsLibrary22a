===========================================
Runnable Scripts for PerceptionsLibrary22a
===========================================

What are scripts?
-----------------

Scripts are small programs that can be run from the command line. 
We have developed a large variety of scripts to help us test out 
the entirty of our library.

How do I run a script?
-----------------------

Before running a script, make sure that the current version of python in the
virtual environment is 3.8, otherwise the script may not run properly. Additionally,
make sure all the requirements are installed properly.

To run a script:
================

1. Open a terminal window. Navigate to the PerceptionsLibrary22a directory.
2. Activate the python virtual environment by typing 
    .. code:: bash 
        
        source venv/bin/activate
3. Run the script by typing
    .. code:: bash 
        
        python3 <script_name>.py

What scripts are available?
----------------------------

This is a list of all the scripts available in the PerceptionsLibrary22a directory:

1. **load_ecg_data.py**
    This script loads the data from the data folder, and runs our LiDAR and Stereo Predictors
    on the data, then displays it to see. 

2. **run_aggregate_predictor.py**
    This script runs the aggregate predictor on the data in the data folder, and displays the results.

3. **run_both_predictors.py**
    This script runs both the LiDAR and Stereo predictors on the data in the data folder, and displays the results.

4. **run_cone_merger.py**
    This script tests the merging of information from the LiDAR and Stereo predictors. This 
    is largely a test script, and is not used in the final product. Merging is done in the
    complete predictor.

5. **run_lidar_predictor.py**
    This script runs the LiDAR predictor on the data in the data folder, and displays the results.

6. **run_stereo_predictor.py**
    This script runs the Stereo predictor on the data in the data folder, and displays the results.

7. **run_threshold_predictor.py**
    This script runs the threshold predictor on the data in the data folder, and displays the results.
    This is largely a test script, and is not used in the final product. Thresholding was a method of extracting
    color from image data, however, this was not used in the final product. We use the YOLO model instead.

8. **run_yolov5_predictor.py**
    This script runs the YOLOv5 predictor on the data in the data folder, and displays the results.

9. **run_endtoend.py**
    This script runs the complete predictor on the data in the data folder, and displays the results.
    This was the file used to generate the results for the final product. It will gather data across the 
    entire perceptions suite and display the results.

10. **run_svm.py**
    This script runs the SVM predictor on the data in the data folder, and displays the results.
    This is largely a test script, and is not used in the final product. SVM was a method of getting the 
    midline of the road. This was used in the final product, and was implemented in the complete predictor.

11. **sim_cones.py**
    This script simulates the placement of cones in the data folder, and displays the results.
    This is largely a test script, and is not used in the final product. This was used to test the 
    complete predictor.

12. **test_logreg_idea.py**
    This script tests the idea of using logistic regression for merging the LiDAR and Stereo predictors.
    This is largely a test script, and is not used in the final product. Merging is done in the complete predictor.

13. **test_setup.py**
    This script tests the setup of the library. It makes sure the library is set up properly.

14. **test_vis2d.py**
    This script tests the 2D visualization of the data in the data folder.
    This is largely a test script, and is not used in the final product. This was a method of 
    creating a 2D visualization of the data. This was not used in the final product.

15. **test_vis3d.py**
    This script tests the 3D visualization of the data in the data folder.
    This is largely a test script, and is not used in the final product. This was used to test the 
    complete predictor.

16. **visualize_cones.py**
    This script visualizes the cones in a 2d plot. This is largely a test script, and provided a 
    view of the cones from the perception stack during the complete predictor testing.

17. **load_data.py**
    This script loads the data from the data folder.