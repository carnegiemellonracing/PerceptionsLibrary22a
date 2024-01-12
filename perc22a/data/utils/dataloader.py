'''DataLoader.py

Defines class that represents a loadable stream of data that can be loaded
from a folder of numpy arrays created from the `FileNode` class defined in
the `driverless` repository.
'''

from perc22a.data.utils.DataInstance import DataInstance

import numpy as np
import os
import cv2
import time


class DataLoader:
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
        return np.load(filepath)


def main():
    data_loader = DataLoader("./data/np-data/")
    print(len(data_loader))
    print(data_loader[5])

    while True:
        for i in range(len(data_loader)):
            left_img = data_loader[i]["left"]  # H x W x 3 (np.uint8) image
            cv2.imshow("Test", left_img)
            cv2.waitKey(1)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
