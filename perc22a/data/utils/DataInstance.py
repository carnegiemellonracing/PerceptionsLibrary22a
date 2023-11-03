'''DataInstance.py

Contains class-representation of an instance of data that perceptions algos
can use to perform prediction on 
'''

from perc22a.data.utils.config import DataType

import numpy as np


class DataInstance:

    def __init__(self, required_types=None):
        # define dictionary for storing data
        self.data = {}

        # define list of required types necessary for DataInstance to save
        self.required_types = []
        if required_types is not None:
            self.required_types = required_types
        else:
            self.required_types = [t for t in DataType]


    def have_all_data(self):
        '''returns true if all required_types are set in DataInstance'''
        for t in self.required_types:
            if t not in self.data:
                return False
        return True

    def __getitem__(self, key: DataType):
        '''get a specific portion of the data'''
        return self.data[key]

    def __setitem__(self, key: DataType, value: np.ndarray):
        '''set a specific data type'''
        self.data[key] = value

    def save(self, path):
        '''save the data as a numpy file at the appropriate path'''
        np.savez(path, **self.data)

    def load(self, path):
        '''load the data from a numpy file created with DataInstance.save'''
        self.data = np.load(path)