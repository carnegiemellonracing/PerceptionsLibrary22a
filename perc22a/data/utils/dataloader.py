import numpy as np
import os
import cv2
import time


class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.instances = os.listdir(data_dir)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
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


# d = {"left": left, "right": right, "point": point}
# np.savez("instance-{i}.npz", d)
