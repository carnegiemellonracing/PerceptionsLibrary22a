.. Perceptions Library 22a documentation master file, created by
   sphinx-quickstart on Sat Jun  1 18:42:05 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================================
Carnegie Mellon Racing Perceptions Library for 22a
==================================================

This library is a collection of perception algorithms for the 22a racecar for Carnegie Mellon Racing.
Carnegie Mellon Racing is a premier student organization at Carnegie Mellon University that designs 
and builds autonomous vehicles to compete in the annual Formula Student Driverless competition. This workspace
works in combination with the driverless workspace that can be found on the `Carnegie Mellon Racing GitHub <https://github.com/carnegiemellonracing/>`_ .

.. image :: ../../index_images/car.jpg
    :align: center

What is Perceptions?
====================
Perceptions is the process of interpreting sensor data to understand the environment around the car. 
This is a crucial part of the autonomous driving stack as it provides the car with the information it 
needs to make decisions. The perception stack on this car consists of a variety of sensors:

- `HESAI AT128 Solid State LiDAR <"https://www.hesaitech.com/product/at128/"/>`_
- `Dual ZED2 Stereo Cameras <"https://www.stereolabs.com/products/zed-2"/>`_
- `MTi-680G RTK GNSS/INS GPS <"https://www.movella.com/products/sensor-modules/xsens-mti-680g-rtk-gnss-ins"/>`_

Together, these sensors provide the car with a complete view of the track and allow for an accurate understanding
of the cones. From this information, we can run a midline algorithm to determine the best path through the cones.

There are also a variety of other algorithms that are used to interpret the data from these sensors. These include
object detection, lane detection, and cone detection. These algorithms are used to provide the car with a complete
understanding of the environment around it. To understand the specifics of these algorithms, please refer to the
documentation.

Getting Started
===============

Setup
-----

1. **Clone the Repository:**

   .. code:: bash

      git clone [repository-link]
      cd PerceptionsLibrary22a

2. **Setup Virtual Environment:** Ensure you have Python 3.8 installed,
   then create a virtual environment:

   .. code:: bash

      python3.8 -m venv env
      source env/bin/activate

3. **Install Dependencies:**

   .. code:: bash

      pip install -r requirements.txt

4. **Set PYTHONPATH:** To ensure ``import perc22a`` works in any script,
   add the absolute path of the ``PerceptionsLibrary22a`` to your
   ``PYTHONPATH``:

   .. code:: bash

      echo "export PYTHONPATH=\"$(pwd):$PYTHONPATH\"" >> ~/.zshrc # or ~/.bashrc
      source ~/.zshrc # or ~/.bashrc

5. **Verify Setup:** Confirm the path was correctly added by echoing the
   ``$PYTHONPATH``:

   .. code:: bash

      echo $PYTHONPATH

   Test the setup:

   .. code:: bash

      python scripts/test_setup.py

   Successful output: ``"Running 'import perc22a' successful"``.

Loading Data
------------

1. **Download Data:** Fetch the data from `this Google Drive
   Link <https://drive.google.com/drive/folders/12l2DpvS4oEfl7_Noc7oUX4AcIDCfB8Zc?usp=drive_link>`__
   and place the ``<name>.tar.gz`` files in the ``data/raw/`` directory.
   Note: The files are large and can expand to more than 10GB when
   extracted.

2. **Extract Data:**

   .. code:: bash

      tar -zxvf data/raw/<name>.tar.gz

   This creates a ``data/raw/<name>`` directory containing numerous
   ``instance-<n>.npz`` files, which represent snapshots of sensor data
   during track testing.

3. **Use DataLoader:** The ``DataLoader`` class, found in
   ``data/utils/dataloader.py``, provides a convenient method for data
   access.

   To demonstrate its use:

   .. code:: bash

      python3 scripts/load_data.py

   This displays a ``cv2`` window. Click on the image and press any key
   to navigate through the data. To exit, either hit ``<Ctrl-C>`` in the
   terminal and press a key in the ``cv2`` window or continue pressing
   keys until all images are cycled through.


Sponsors
========

None of this would be possible without our absolutely amazing sponsors!

.. image :: ../../index_images/sponsors.png
    :align: center

.. toctree::
   :maxdepth: 5
   :caption: Contents:

   perc22a/data/load_data
   perc22a/mergers/merger
   perc22a/predictors/aggregate/AggregatePredictor
   perc22a/predictors/lidar/lidar
   perc22a/predictors/stereo/stereo
   perc22a/predictors/utils/utils
   perc22a/predictors/utils/vis/vis
   perc22a/predictors/utils/lidar/lidar
   perc22a/svm/SVM
   perc22a/utils/utils
   scripts/scripts
