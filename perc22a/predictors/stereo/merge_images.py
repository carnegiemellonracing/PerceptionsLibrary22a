#! do not use

import cv2
import numpy as np


def combine_stereo_images(image1, image2, baseline_distance, camera_angle):
    """
    Combines two stereo images into a full image.

    :param image1: First stereo image (left-right stitched).
    :param image2: Second stereo image (left-right stitched).
    :param baseline_distance: Distance between the stereo camera pairs.
    :param camera_angle: Angle between the cameras.
    :return: Combined full image.
    """

    # Combine images side-by-side
    combined_image = np.hstack((image1, image2))

    # Additional processing can be done here if needed,
    # especially to handle the camera angle and baseline distance.

    return combined_image


# Example usage
# Load your stereo images (image1 and image2) here
image1 = cv2.imread("path_to_first_stereo_image.jpg")
image2 = cv2.imread("path_to_second_stereo_image.jpg")

baseline_distance = 10
camera_angle = 60

combined_image = combine_stereo_images(image1, image2, baseline_distance, camera_angle)

# Optionally, save or display the combined image
cv2.imwrite("combined_stereo_image.jpg", combined_image)
cv2.imshow("Combined Image", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
