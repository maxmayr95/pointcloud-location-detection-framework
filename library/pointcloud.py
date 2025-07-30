import os
from dotenv import load_dotenv

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def downsample_point_cloud(point_cloud,voxel_size=0.001, force_downsample=False):
    load_dotenv()
    should_downsample = os.getenv("DOWNSAMPLE_POINTCLOUD", "False").strip().lower() == "true"

    """
    Downsample the input point cloud using voxel sampling to reduce the number of points.
    """
    #voxel_size = 0.001
    logging.info(f"Downsampling point cloud with voxel size: {voxel_size}")
    if not should_downsample and not force_downsample:
        return point_cloud

    return point_cloud.voxel_down_sample(voxel_size)

