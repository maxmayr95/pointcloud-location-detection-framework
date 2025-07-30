import laspy
import numpy as np
import open3d as o3d

def read_las(filepath):
    """
    Reads a LAS file and converts it to an Open3D PointCloud with a temporary downsampled size.

    Args:
    filepath (str): Path to the LAS file.
    target_size (int): Target number of points to sample (default is 100000).

    Returns:
    o3d.geometry.PointCloud: The loaded point cloud.
    """
    filepath = filepath
    with laspy.open(filepath) as f:
        las = f.read()
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Temporarily downsampling the points (if needed)
        # if len(points) > target_size:
        #     indices = np.random.choice(len(points), target_size, replace=False)
        #     points = points[indices]

        # Create the point cloud
        point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    return point_cloud

