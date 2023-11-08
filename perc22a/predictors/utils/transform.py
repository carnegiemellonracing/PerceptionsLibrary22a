"""
    this file contains classes for performing transformations of objects
    from one perspective to another perspective. This includes 
        - transforming from one 3d coordinate axis to another (AxisTransformer)
        - transforming from world coords to image coords (WorldImageTransformer)
"""

import numpy as np
import open3d as o3d
from skspatial.objects import Plane
import yaml
import time

DEG_TO_RAD = np.pi / 180


def c(x):
    return np.cos(x)


def s(x):
    return np.sin(x)


def make_RX(rad):
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c(rad), -s(rad), 0],
            [0, s(rad), c(rad), 0],
            [0, 0, 0, 1],
        ]
    )


def make_RY(rad):
    return np.array(
        [
            [c(rad), 0, s(rad), 0],
            [0, 1, 0, 0],
            [-s(rad), 0, c(rad), 0],
            [0, 0, 0, 1],
        ]
    )


def make_RZ(rad):
    return np.array(
        [
            [c(rad), -s(rad), 0, 0],
            [s(rad), c(rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def make_T(dx, dy, dz):
    return np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])


class PoseTransformations:
    # stereo_cam = config['stereo_camera']
    # lidar = config['lidar']
    # gps = config['gps']

    # create transformation matrices for each sensor
    def __init__(self, path):
        with open(path, "r") as file:
            self.config = yaml.safe_load(file)
            for sensor in self.config["sensors"]:
                self.TRdict[sensor["name"]] = self.create_sensor_t(sensor)

                # inverse?

    def create_sensor_t(sensor):
        # construct homogeneous rotation matrices
        # all angles and positions are given relative to origin from yaml
        rx = sensor["pose"]["orientation"]["theta"] * DEG_TO_RAD
        ry = sensor["pose"]["orientation"]["phi"] * DEG_TO_RAD
        rz = sensor["pose"]["orientation"]["psi"] * DEG_TO_RAD
        RX = make_RX(rx)
        RY = make_RY(ry)
        RZ = make_RZ(rz)

        dx = sensor["pose"]["position"]["x"]
        dy = sensor["pose"]["position"]["y"]
        dz = sensor["pose"]["position"]["z"]

        # make transformation matrix
        T = make_T(dx, dy, dz)
        result = T @ RZ @ RY @ RX

        return result

    # ripped from other classes
    def _homogenize(self, points):
        return np.hstack([points, np.ones((points.shape[0], 1))])

    def _inhomogenize(self, points):
        if np.any(np.abs(points[:, 3]) < 0.00005):
            raise Exception(
                "AxisTransformer: converting to inhomogenous coordinate with small divisor"
            )

        points[:, :3] /= points[:, 3].reshape((-1, 1))
        return points[:, :3]

    def to_origin(self, sensor_name, points, inverse=False):
        # homogenize points and get transformation matrix depending on sensor
        points_homogenous = self._homogenize(points)
        M = self.TRdict[sensor_name]  # if not inverse else self.TRdict[sensor_name]

        # transform points and inhomegenize
        points_transformed = (M @ points_homogenous.T).T
        result = self._inhomogenize(points_transformed)

        return result

    def from_origin(self, sensor_name, points):
        points_homogenous = self._homogenize(points)
        start = time.time()
        M = np.linalg.inv(self.TRdict[sensor_name])
        end = time.time()
        print(end - start)

        # transform points and inhomegenize
        points_transformed = (M @ points_homogenous.T).T
        result = self._inhomogenize(points_transformed)

        return result


class AxisTransformer:
    """
    TODO: consider if you want to change to do a custom sequence of the 4 operations
        - e.g. could do [translate, rotate x, rotate z, translate, rotate x]

    does the following transformations in order
        1. rotate w.r.t. x-axis by degx (pitch)
            - positive points upwards (i think?)
        2. rotate w.r.t. y-axis by degy (roll)
            - positive rolls right (i think?)
        3. rotate w.r.t. z-axis by degz (yaw)
            - positive turns left (i think?)
        4. translate according to (dx, dy, dz)
    """

    def __init__(self, degx=0, degy=0, degz=0, dx=0, dy=0, dz=0):
        rx, ry, rz = np.array([degx, degy, degz]) * DEG_TO_RAD

        # construct homogenous rotation matrices
        RX = make_RX(rx)
        RY = make_RY(ry)
        RZ = make_RZ(rz)

        # construct homgenous translation matrix
        T = make_T(dx, dy, dz)

        # combine matrices in documented order of operations
        # NOTE: transformation matrix is left-multiplied onto points to right-most matrices building TR do operations first
        self.TR = T @ RZ @ RY @ RX
        self.TRinv = np.linalg.inv(self.TR)

    def _homogenize(self, points):
        return np.hstack([points, np.ones((points.shape[0], 1))])

    def _inhomogenize(self, points):
        if np.any(np.abs(points[:, 3]) < 0.00005):
            raise Exception(
                "AxisTransformer: converting to inhomogenous coordinate with small divisor"
            )

        points[:, :3] /= points[:, 3].reshape((-1, 1))
        return points[:, :3]

    def transform(self, points, inverse=False):
        # convert points to homogenous coordinates and get transform matrix
        points_homogenous = self._homogenize(points)
        M = self.TR if not inverse else self.TRinv

        # transform and convert back to inhomogenous coordinates
        points_transformed = (M @ points_homogenous.T).T
        result = self._inhomogenize(points_transformed)

        return result


class CustomAxisTransformer:
    """
    Can perform the following operations with key and value
    1. "degx": rotate w.r.t. x-axis by degx (pitch)
        - positive points upwards (i think?)
    2. "degy": rotate w.r.t. y-axis by degy (roll)
        - positive rolls right (i think?)
    3. "degz": rotate w.r.t. z-axis by degz (yaw)
        - positive turns left (i think?)
    4. "t": translate according to [dx, dy, dz]
    """

    def __init__(self, commands):
        self.TR = np.eye(4)

        for c, v in commands:
            if c == "degx":
                self.TR = make_RX(v * DEG_TO_RAD) @ self.TR
            elif c == "degy":
                self.TR = make_RY(v * DEG_TO_RAD) @ self.TR
            elif c == "degz":
                self.TR = make_RZ(v * DEG_TO_RAD) @ self.TR
            elif c == "t":
                self.TR = make_T(v[0], v[1], v[2]) @ self.TR

        self.TRinv = np.linalg.inv(self.TR)

    def _homogenize(self, points):
        return np.hstack([points, np.ones((points.shape[0], 1))])

    def _inhomogenize(self, points):
        if np.any(np.abs(points[:, 3]) < 0.00005):
            raise Exception(
                "AxisTransformer: converting to inhomogenous coordinate with small divisor"
            )

        points[:, :3] /= points[:, 3].reshape((-1, 1))
        return points[:, :3]

    def transform(self, points, inverse=False):
        # convert points to homogenous coordinates and get transform matrix
        points_homogenous = self._homogenize(points)
        M = self.TR if not inverse else self.TRinv

        # transform and convert back to inhomogenous coordinates
        points_transformed = (M @ points_homogenous.T).T
        result = self._inhomogenize(points_transformed)

        return result


class WorldImageTransformer:
    def __init__(self, fx, fy, cx, cy):
        self.params = (fx, fy, cx, cy)

    def world_to_image(self, points):
        """
        converts 3D world points in image frame "(Z forward, X right, Y down)"
        to pixel coordinates in 2D image

        input:
            - points: (N,3) np.ndarray where ith row is [x,y,z] coordinate of ith point
                    where Z is forward, X right, and Y down
        output:
            - coords: (N,2) np.ndarray of floating point values representing position
                    where ith point in points would be in coords
        """
        # copied from ZED.py ZEDSDK.world_to_image
        # following the procedure described here in the below articles
        # converting between 3D and 2D: https://support.stereolabs.com/hc/en-us/articles/4554115218711-How-can-I-convert-3D-world-coordinates-to-2D-image-coordinates-and-viceversa-
        # where pinhole parameters are stored: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CameraParameters.html#a94a642b5f1c8511df7584aa6a187f559
        # how to access pinhole parameters: https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#aabca561d73610707323bcec1a4c1d6b0

        fx, fy, cx, cy = self.params
        N = points.shape[0]

        coords = np.zeros((N, 2))
        coords[:, 0] = (points[:, 0] / points[:, 2]) * fx + cx
        coords[:, 1] = (points[:, 1] / points[:, 2]) * fy + cy

        return coords

    def image_to_world(self, coords, depth):
        """
        converts 2D pixel coordinates in images with their depth map
        to 3D world points in the image frame with (Z forward, X right, Y down)

        matches the results of the point cloud object that is collected with
        the same depth map

        input:
            - coords: (N,2) np.ndarray where ith row is [u,v] coordinate of ith pixel
            - depth: (W,H) np.ndarray storing the depth values of pixels in image
        output:
            - points: (N,3) np.ndarray storing 3D world points corresonding to the pixel coordinates in coords
        """
        fx, fy, cx, cy = self.params
        N = coords.shape[0]

        u = coords[:, 0]
        v = coords[:, 1]
        Z = depth[u, v]

        # NOTE: had to modify formula to get it to match the structure of point cloud object
        # (X-values: use v instead of u, Y-vals: use u instead of v and negate, Z-vals: negate)
        points = np.zeros((N, 3))
        points[:, 0] = ((v - cx) * Z) / fx
        points[:, 1] = -((u - cy) * Z) / fy
        points[:, 2] = -Z

        return points


if __name__ == "__main__":
    plane = Plane([0, 0, 0], [0, 0, 1])
    xs = np.linspace(-1, 1, 30)
    ys = np.linspace(-0.5, 0.5, 30)
    points = plane.to_points(lims_x=xs, lims_y=ys)

    # T = AxisTransformer(degx=45, dx=1, dy=0.7, dz=-0.5)
    T = CustomAxisTransformer([("degx", 90), ("degy", 45), ("degz", 45)])
    result = T.transform(points)

    transforms = PoseTransformations("./run_config.yaml")
    # transformed_points = transforms.to_origin('sensor_name', points), where sensor_name is name in yaml

    axis_vis = vis.create_axis_vis()
    points_vis = vis.create_point_vis(points, colors=np.array([0, 0, 1]))
    result_vis = vis.create_point_vis(result, colors=np.array([1, 0.6, 0]))

    o3d.visualization.draw_geometries([axis_vis, points_vis, result_vis])
