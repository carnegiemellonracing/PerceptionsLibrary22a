import numpy as np
import open3d as o3d

# these matrices setup the parameters of the Open3D camera
# to get various views of the car in the visualizer
EXTRINSIC_BEHIND = np.array(
    [
        [1, 0.03, -0.04, 0.19],
        [-0.03, -0.24, -0.97, 1.24],
        [-0.04, 0.97, -0.24, 3.52],
        [0, 0, 0, 1],
    ]
)

EXTRINSIC_BIRD = np.array(
    [
        [1.0, -0.04, -0.07, 2.43],
        [-0.04, -1.0, -0.01, 12.8],
        [-0.07, 0.01, -1.0, 35.52],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def create_axis_vis(size=1, npoints=100):
    """
    Creates an Open3D point cloud object that contains points that
    will visualize 3D axis with the intersecting point being the origin
    of the point cloud. This will be useful for having a simple frame of
    reference for the point cloud
    Inputs:
        - size: floating point value describing how long the axes will be
        - npoints: the number of points that each axis will be made of
    Output:
        - axis_vis: an Open3D Point Cloud Object setup ready for vis
    """
    dim = np.linspace(0, size, npoints).reshape((-1, 1))
    zeros = np.zeros(dim.shape).reshape((-1, 1))
    xpoints = np.hstack([dim, zeros, zeros])
    ypoints = np.hstack([zeros, dim, zeros])
    zpoints = np.hstack([zeros, zeros, dim])
    axis_points = np.vstack([xpoints, ypoints, zpoints])
    axis_colors = np.where(axis_points > 0, 1, 0)
    axis_vis = create_point_vis(axis_points, axis_colors)
    return axis_vis


def create_plane_vis(plane, xmin=-3, xmax=3, ymin=0.5, ymax=3, npoints=100):
    """Creates Open3D PointCloud object that uniformly samples points on the
    given input plane and returns them for visualizing a plane as a point cloud

    Args:
        plane (skspatial.objects.Plane): plane object containing fitted plane parameters
        xmin (int, optional): minimum x-coordinate to get points from. Defaults to -3.
        xmax (int, optional): maximum x-coordinate to get points from. Defaults to 3.
        ymin (float, optional): minimum y-coordinate to get points from. Defaults to 0.5.
        ymax (int, optional): maximum y-coordinate to get points from. Defaults to 3.
        npoints (int, optional): number of points along each dimension to sample points from. Defaults to 100.

    Returns:
        o3d.geometry.PointCloud(): point cloud visualization objecting containing points sampled on the plane
    """
    # get points on the plane
    xs = np.linspace(xmin, xmax, npoints)
    ys = np.linspace(ymin, ymax, npoints)
    points = plane.to_points(lims_x=xs, lims_y=ys)

    # compute colors for the plane
    colors = np.zeros((points.shape[0], 3))
    colors[:, 0] = 105
    colors[:, 1] = 105
    colors[:, 2] = 179
    colors = colors / 255

    # return visualization as points on the plane
    return create_point_vis(points, colors=colors)


def create_point_vis(points, colors=None):
    """
    Creates an Open3D point cloud object used in visualization
    and returns it

    Inputs:
        - points: (N,3) np.array of real values representing N points
        - colors: (N,3) or (3,) np.array of real values [0,1] representing colors
                  (optional)
    Output: pcd - Open3D Point Cloud Object setup ready for visualization
                  with the o3d.visualization.draw_geometries function
    """
    N = points.shape[0]
    points = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        if len(colors.shape) == 1:
            colors = np.vstack([colors.reshape((-1, 3))] * N)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def create_cylinder_vis(
    cylinder_centers, colors=[0, 1, 0], radius=0.2, height=0.4, resolution=10
):
    """
    Creates a list of Open3D LineSet meshes of cylinders whose centers
    are listed by the list cylinder_centers

    Inputs:
        - cylinder_centers: (N,3) np.array of cylinder center positions
        - color: list of length 3 of color to assign cylinders (optional)
        - radius: radius (units???) to give cylinders (optional)
        - height: height (units???) to give cylinders (optional)
        - resolution: how circular to make the cylinders

    Output:
        - cylinders: list of Open3D LineSet meshes of cylinders ready for
                     visualization with o3d.visualization.draw_geometries
    """

    n_centers = len(cylinder_centers)

    # bring colors to a fixed input size
    if n_centers > 0:
        colors = np.array(colors)
        if len(colors.shape) == 1:
            colors = np.array([colors] * n_centers)

    cylinders = []
    # print(cylinder_centers)
    # print(colors)
    for i in range(n_centers):
        centroid = cylinder_centers[i]
        color = colors[i, :]

        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=height, resolution=resolution, split=1
        )
        cylinder = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder)
        cylinder = cylinder.translate(tuple(centroid))
        #print(color)
        cylinder.paint_uniform_color(color)
        cylinders.append(cylinder)

    return cylinders


def display_point_cloud(pc, colors=None, cones=None):
    pcd = create_point_vis(pc, colors)
    if cones is not None:
        cylinders = create_cylinder_vis(cones, colors=[0, 1, 0])
    else:
        cylinders = []
    o3d.visualization.draw_geometries([pcd] + cylinders)


def init_visualizer_window(name="Point Cloud Visualization"):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=960, height=540, left=1000)
    return vis


def update_visualizer_perspective(vis, extrinsic):
    ctr = vis.get_view_control()
    params = ctr.convert_to_pinhole_camera_parameters()
    params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(params)
    return


def update_visualizer_window(
    window, points, pred_cones=[], colors=None, colors_cones=[0, 1, 0], plane=None
):
    # takes about 60-100ms to update the visualizer
    # calculate the new geometries
    objects = []

    objects.append(create_axis_vis())
    objects.append(create_point_vis(points, colors))
    if plane is not None:
        objects.append(create_plane_vis(plane, ymin=0))

    if pred_cones is not None:
        pred_cylinders = create_cylinder_vis(pred_cones, colors=colors_cones)
        [objects.append(cyl) for cyl in pred_cylinders]
    if window is not None:
        # reset the geometries
        window.clear_geometries()
        for o in objects:
            window.add_geometry(o)

        # update the camera position and orientation
        update_visualizer_perspective(window, EXTRINSIC_BEHIND)

        # update the actual view
        window.poll_events()
        window.update_renderer()
    else:
        o3d.visualization.draw_geometries(objects, window_name=f"{points.shape[0]} points")


def color_matrix(fns=None, pcs=None):
    """
    Written By: Meher Mankikar (mmankika) 10/22/2022
    point cloud visualization using Open3D will write the first points at
    a specific position and will not write any future points at that exact
    same position. As a result, display the later stages of the point cloud
    and then earlier ones with various colors so that pipeline stages can
    be visualized
    """
    C = (
        np.array(
            [
                [91, 192, 235],
                [216, 157, 106],
                [219, 83, 117],
                [29, 220, 32],
                [127, 29, 220],
            ]
        )
        / 255
    )
    ncolors = C.shape[0]

    pointsList = []
    colorsList = []

    # add points and colors in order of stage
    for f in range(len(fns if fns is not None else pcs)):
        # get the colors for the current stage and add
        tempPoints = fns[f](tempPoints) if fns is not None else pcs[f]
        tempColors = np.ones(tempPoints.shape)[:, :3] * np.array(C[f % ncolors, :])

        pointsList.append(tempPoints)
        colorsList.append(tempColors)

    # reverse the stages so that later stages are able to be visualized separately
    pointsList = pointsList[::-1]
    colorsList = colorsList[::-1]

    P = np.vstack(pointsList)
    C = np.vstack(colorsList)
    return P, C
