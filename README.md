# 22a's Perceptions Library

## Loading Data

Go to [this Google Drive Link](https://drive.google.com/drive/folders/12l2DpvS4oEfl7_Noc7oUX4AcIDCfB8Zc?usp=drive_link) download any of the `<name>.tar.gz` files and move it to the `data/raw/` folder. Note that this file will be quite large and will be even larger when untar-ed (up to and more than 10GB). Inside of it, extract the contents by running
```
tar -zxvf <name>.tar.gz
```
This will create a folder labeled as `data/raw/<name>`. Inside of the folder, you will find multiple files labeled `instance-<n>.npz`. Each of these files are a snapshot of the sensor data while the car was driving during track testing and areordered by number.

We've implemented a Python class for loading the data in a nice way called the `DataLoader` loacted in `data/utils/dataloader.py`. 

A sample script using the data loader can be found at `scripts/load_data.py`. To run it, go to the root of the repository, and run
```
python3 scripts/load_data.py
```
This program will display a `cv2` window (`cv2` is a great library for displaying images and doing computer vision operations on them too). Click on the image and press any key some number of times. This will tab through the data in the track testing. 

If you would like to quit the program, go to the terminal containing the running code, press `<Ctrl-C>`, and then, click on the `cv2` window and press a key. Then, the process should exit and you can run commands on your terminal. If that doesn't work, you can keep pressing keys until all images are looped through.