# perceptions specific imports
from perc22a.data.utils.dataloader import DataLoader

# general python imports
import time
import cv2


def main():
    dl = DataLoader("perc22a/data/raw/track-testing-09-29")

    for i in range(len(dl)):
        # load the i-th image from track testing run
        print(dl)
        data = dl[i]
        img = data["left_color"]

        # display the image
        cv2.imshow(f"left image (idx: {i})", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
