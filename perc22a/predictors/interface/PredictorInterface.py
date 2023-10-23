""" interface.py

This file contains the interface definining a [Predictor] object for 
perceptions. 

CMR Perception's primary goal is taking sensor data and estimating the
3D positions of cones with ...
    - high accuracy = estimated positions of cones are close to true positions
    - high recall = more cones are have estimated positions than fewer ones
    - low latency = functions return quickly and are efficient

Subclasses of the [Predictor] interface will implement various computer-vision
based algorithms to take sensor data and estimate 3D cone position.

Usage of Predictor classes will be as follows

>>> dl = DataLoader(<path>)
>>> predictor = Predictor()
>>> for i in range(len(dl)):
>>>     data = dl[i]
>>>     cones = predictor.predict(data) # note that cones is a 3-tuple
>>>     predictor.display()


"""

class Predictor:
    """interface for a [Predictor] that takes time-series data from various
    sensors and performs predictions using it
    
    the current list of sensor data (in numpy arrays) that we have used are...
        - point clouds
            - (N, 6) where N is the number of points in the point cloud
            - np.float32 datatype
                - see Wikipedia article on "Single-precision floating-format"
            - Columns: [x, y, z, intensity, ring, timestamp]
                - x, y, and z are in meters
                - intesity is a value from 0-255 representing point reflectivity
                - ring is the index of laser that point was created from
                - timestamp is the time at which laser was shot or received
        - left stereocamera images
            - (H, W, 4) where H and W are the height and width of the image
            - np.uint8 datatype
            - 4 channels are [R, G, B, Alpha]
        - depth stereocamera images
            - (H, W) where H and W are th height and width of the image
            - np.float32 datatype
            - elements are either nan or the depth of the object at pixel
                - see Wikipedia article on "NaN" for what a nan variable is
        - xyz stereocamea images
            - (H, W, 4) where H and W are the height and widht of the image
            - np.float32 datatype
            - Columns: [x, y, z, color]
                - color is a 32-bit datatype with 8-bit color values packed
                inside (R, G, B, Alpha)
        - TODO: incorporate gps collected data
    
    the goal of [Predictor] sub-classes is to take the sensor data and create
    the "best" predictor of 3D cone positions
    """

    def __init__(self):
        """initializes the predictor object, will include adding any stateful
        information such as opening files, loading PyTorch model parameters,
        and more
        """
        pass
    
    def predict(data):
        """will take the data and predict the 3D positions of cones

        Arguments:
            data - dictionary containing various sensor data used for prediction

        Return: (orange_cones, blue_cones, yellow_cones)
            - orange_cones = None if no orange cones, else (N, 3) array where
            N is the number of orange cones and columns are [x, y, z] coords
            in meters
            - blue_cones = similar to orange_cones, but for blue cones
            - yellow_cones = similar to blue_cones, but for yellow cones
        """

        orange_cones = None
        blue_cones = None
        yellow_cones = None

        return orange_cones, blue_cones, yellow_cones
    
    def display():
        """this functions is purely for displaying visual information used in
        the pipeline and primarily for debugging purposes.

        Should be implemented to visualize images, point clouds, bounding boxes,
        and more.

        No arguments and no return values.
        """