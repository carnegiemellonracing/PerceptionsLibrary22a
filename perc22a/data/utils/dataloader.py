'''DataLoader.py

Defines class that represents a loadable stream of data that can be loaded
from a folder of numpy arrays created from the `FileNode` class defined in
the `driverless` repository.

Functions:
    __init__: initializes data loader pointing to directory
    __len__: returns number of instances available from DataLoader
    __getitem__: returns index'th instance of data from directory
'''

from perc22a.data.utils.DataInstance import DataInstance

import numpy as np
import os
import cv2
import time

class DataLoader():
    def __init__(self, data_dir):
        '''initializes data loader pointing to directory'''
        self.data_dir = data_dir
        self.instances = os.listdir(data_dir)
    
    def __len__(self):
        '''returns number of instances available from DataLoader'''
        return len(self.instances)
    
    def __getitem__(self, index):
        '''returns index'th instance of data from directory'''
        filename = f"instance-{index}.npz"
        filepath = os.path.join(self.data_dir, filename)

        di = DataInstance()
        di.load(filepath)
        return di