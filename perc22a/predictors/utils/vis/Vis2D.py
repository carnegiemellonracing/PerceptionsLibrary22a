
from enum import Enum
import numpy as np
import cv2

from perc22a.predictors.utils.cones import Cones


# conversion from meters to pixels
PIXELS_PER_M = 30

# image space
DIMS = np.array([990, 540])
ORIGIN = np.array([720, 270])

# sizes of drawable objects (in meters)
AXIS_LONG_M = 1
AXIS_SHORT_M = 0.1
CONE_LENGTH_M = 0.1
SPLINE_LENGTH_M = 0.075

# sizes of drawable objects (in pixels)
AXIS_LONG_PIXELS = int(AXIS_LONG_M * PIXELS_PER_M)
AXIS_SHORT_PIXELS = int(AXIS_SHORT_M * PIXELS_PER_M)
CONE_LENGTH_PIXELS = int(CONE_LENGTH_M * PIXELS_PER_M)
SPLINE_LENGTH_PIXELS = int(SPLINE_LENGTH_M * PIXELS_PER_M)

ORANGE = np.array([32, 131, 250])

class CFG_COLORS(Enum):
    BLUE = 1
    YELLOW = 2
    ORANGE = 3
    UNKNOWN = 4

CV2_COLORS = {
    CFG_COLORS.BLUE: [255, 191, 0],
    CFG_COLORS.YELLOW: [7, 238, 255],
    CFG_COLORS.ORANGE: [0, 150, 255]
}


class Vis2D:
    def __init__(self, name='vis2d') -> None:       
        # initialize display-able objects
        self.points = None
        self.cones = None
        self.image = np.ones(())
        self.name = name

    def start(self):
        cv2.imshow(self.name, self.image)
        cv2.waitKey(1)

    def _draw_grid(self):
        '''draw meter lines in an image to make a frame of reference'''
        first_horiz_bars = ORIGIN[0] - np.arange(0, ORIGIN[0], PIXELS_PER_M)
        second_horiz_bars = np.arange(ORIGIN[0], DIMS[0], PIXELS_PER_M)
        first_vert_bars = ORIGIN[1] - np.arange(0, ORIGIN[1], PIXELS_PER_M)
        second_vert_bars = np.arange(ORIGIN[1], DIMS[1], PIXELS_PER_M)

        # get indices where bars are supposed to happen
        horiz_bars = np.concatenate([first_horiz_bars, second_horiz_bars])
        vert_bars = np.concatenate([first_vert_bars, second_vert_bars])

        # draw the horizontal meter-lines with a black bar
        self.image[horiz_bars, :, :] = 0
        self.image[:, vert_bars, :] = 0

        return

    def _draw_axes(self):
        '''draw an axis on the image from the origin'''
        # draw the x-axis (red)
        rs = ORIGIN[0] - AXIS_SHORT_PIXELS // 2
        re = ORIGIN[0] + AXIS_SHORT_PIXELS // 2
        cs = ORIGIN[1]
        ce = ORIGIN[1] + AXIS_LONG_PIXELS
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [255, 0, 0]

        # draw the y-axis (blue)
        rs = ORIGIN[0] - AXIS_LONG_PIXELS
        re = ORIGIN[0]
        cs = ORIGIN[1] - AXIS_SHORT_PIXELS // 2
        ce = ORIGIN[1] + AXIS_SHORT_PIXELS // 2
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [0, 255, 0]

        # draw the center of the axis (black)
        rs = ORIGIN[0] - AXIS_SHORT_PIXELS // 2
        re = ORIGIN[0] + AXIS_SHORT_PIXELS // 2
        cs = ORIGIN[1] - AXIS_SHORT_PIXELS // 2
        ce = ORIGIN[1] + AXIS_SHORT_PIXELS // 2
        rs, re, cs, ce = int(rs), int(re), int(cs), int(ce)
        self.image[rs:re, cs:ce, :] = [0, 0, 0]

        return

    def _draw_squares(self, centers, color, length=20):
        '''draws squares in the image for each point'''
        for i in range(centers.shape[0]):
            r, c = centers[i, :]
            rs, re = r - length, r + length
            cs, ce = c - length, c + length

            c = color if color is not None else ORANGE
            self.image[rs:re, cs:ce, :] = c

        return
    
    def _setup_image(self):
        self.image = np.ones((DIMS[0], DIMS[1], 3)) * 255
        self._draw_grid()
        self._draw_axes()
        return

    def _points_to_pixels(self, points):
        '''converts points in x and y dimensions to a central position in the 
        image where the pixels should be 

        NOTE: function will remove points that are not in the image
        '''
        points = np.rint(points * PIXELS_PER_M)
        points[:, 1] *= -1
        pixel_deltas = points[:, [1, 0]]
        pixels = pixel_deltas + ORIGIN
        pixels = pixels.astype(np.int64)

        in_height = np.logical_and(pixels[:, 0] >= 0, pixels[:, 0] <= DIMS[0])
        in_width = np.logical_and(pixels[:, 1] >= 0, pixels[:, 1] <= DIMS[1])
        in_image = np.logical_and(in_height, in_width)

        pixels = pixels[in_image]
        return pixels
    
    def set_points(self, points: np.ndarray):
        '''sets the point cloud to visualize on next .display call'''
        self.points = points

    def set_cones(self, cones: Cones):
        '''sets the cones to visualize on next .display call'''
        self.cones = cones

    def _update_points(self):
        '''updates 2D visualization with latest points'''
        if self.points is None:
            return
        
        # remove any all zero points in the pointcloud
        self.points = self.points[np.any(self.points != 0, axis=1)][:,:3]

        # modify the pointcloud geometry        
        self.points_vis.points.clear()
        self.points_vis.points.extend(self.points)

        # update geometry in visualization
        self.vis.update_geometry(self.points_vis)

        self.points = None

    def update(self):
        self._setup_image()
        cones = self.cones.to_numpy()
        blue_cones_arr, yellow_cones_arr, orange_cones_arr = cones
        if len(blue_cones_arr.shape) != 2 or blue_cones_arr.shape[0] == 0:
            cones = np.zeros((0, 3))
            print("CVVis Warning: not given any cones or in improper format")
        
        #make color arrs
        yellow_color = CV2_COLORS[CFG_COLORS.YELLOW]
        blue_color = CV2_COLORS[CFG_COLORS.BLUE]
        orange_color = CV2_COLORS[CFG_COLORS.ORANGE]

        # draw the points
        yellow_cone_pixels = self._points_to_pixels(yellow_cones_arr)
        blue_cone_pixels = self._points_to_pixels(blue_cones_arr)
        orange_cone_pixels = self._points_to_pixels(orange_cones_arr)


        self._draw_squares(yellow_cone_pixels, yellow_color, length=CONE_LENGTH_PIXELS)
        self._draw_squares(blue_cone_pixels, blue_color, length=CONE_LENGTH_PIXELS)
        self._draw_squares(orange_cone_pixels, orange_color, length=CONE_LENGTH_PIXELS)

        cv2.imshow(self.name, self.image.astype(np.uint8))
        cv2.waitKey(1)
        pass

    def close(self):
        cv2.destroyWindow(self.name)
        pass
