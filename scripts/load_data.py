# perceptions specific imports
from utils.dataloader import DataLoader

# general python imports
import time
import cv2


def main():
    dl = DataLoader("data/raw/track-testing-09-29")
    for i in range(len(dl)):
        # load the i-th image from track testing run
        
        data = dl[i]
        img2 = data["depth_image"]
        img2 = cv2.resize(img2, (750, 400))
        img3 = data["xyz_image"]
        img3 = cv2.resize(img3, (750, 400))
        imgL = data["left_color"]
        imgL = cv2.resize(imgL, (750, 400))
        imgR = data["right_color"]
        imgR = cv2.resize(imgR, (750, 400))
        print(data["points"])
        print(data["points"].shape)

        cv2.imshow("Depth Image???", img2)
        cv2.imshow("XYZ Image???", img3)
        cv2.imshow("Left Image???", imgL)
        cv2.imshow("Right Image???", imgR)
        if cv2.waitKey(0) & 0xff == 'w':
            continue
        if cv2.waitKey(1) & 0xff == 'q':
            break

        # display the image
        # cv2.imshow(f"left image (idx: {i})", img)
        # cv2.waitKey(0)

if __name__ == "__main__":
    main()