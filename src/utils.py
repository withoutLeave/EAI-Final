import os
import random
from typing import Optional
import numpy as np

from transforms3d.quaternions import quat2mat

from .type import Grasp
from .constants import PC_MAX, PC_MIN
import torch
import torch.nn.functional as F
import numpy as np
from . import constants  # Make sure constants.py is accessible


def get_workspace_mask_height(
    pc_world: np.ndarray,
    z_percentile_threshold: float = 85,
    height_above_table: float = 0.02,
    max_height_above_table: float = 0.10,
    min_object_points: int = 50
) -> np.ndarray:
    """
    仅基于高度统计分割物体点，不用DBSCAN
    """
    if pc_world.shape[0] == 0:
        return np.array([], dtype=bool)
    # 1. 估计桌面高度
    print(f"len(pc_world): {len(pc_world)}")
    table_height = np.percentile(pc_world[:, 2], z_percentile_threshold) 
    print(f"Estimated table height: {table_height:.3f} m")
    # 2. 基于高度过滤
    height_diff = pc_world[:, 2] - table_height
    object_mask = (height_diff >= height_above_table) & (height_diff <= max_height_above_table)
    # 3. 可选：XY范围过滤
    if hasattr(constants, 'PC_MIN') and hasattr(constants, 'PC_MAX'):
        xy_mask = (
            (pc_world[:, 0] >= constants.PC_MIN[0]) & 
            (pc_world[:, 0] <= constants.PC_MAX[0]) &
            (pc_world[:, 1] >= constants.PC_MIN[1]) & 
            (pc_world[:, 1] <= constants.PC_MAX[1])
        )
        object_mask = object_mask & xy_mask
    # 4. 确保有足够的点
    if np.sum(object_mask) < min_object_points:
        return np.zeros(len(pc_world), dtype=bool)
    return object_mask

def get_workspace_mask_pose(
    pc_world: np.ndarray,
    table_pose: np.ndarray
) -> np.ndarray:
    """
    Filters points based on X, Y AABB (from constants) and Z distance to the table plane.

    Args:
        pc_world (np.ndarray): Point cloud in world coordinates, shape (N, 3).
        table_pose (np.ndarray): Pose of the table (4x4 matrix) in world coordinates.
                                 Assumes table surface is its local XY plane (Z=0) and
                                 its local Z-axis points "up" from the surface.

    Returns:
        np.ndarray: Boolean mask of shape (N,) indicating points within the workspace.
    """
    if pc_world.shape[0] == 0:
        return np.array([], dtype=bool)

    # 1. X and Y AABB filtering (using world coordinates from constants)
    # These constants (PC_MIN, PC_MAX) are defined based on OBJ_INIT_TRANS and OBJ_RAND_RANGE.
    # This defines a general area of interest in the XY plane.
    mask_x = (pc_world[:, 0] >= constants.PC_MIN[0]) & (pc_world[:, 0] <= constants.PC_MAX[0])
    mask_y = (pc_world[:, 1] >= constants.PC_MIN[1]) & (pc_world[:, 1] <= constants.PC_MAX[1])

    # 2. Z filtering based on distance to the table plane
    table_origin_world = table_pose[:3, 3]  # Origin of the table's frame in world
    table_z_axis_world = table_pose[:3, 2]  # Z-axis of the table's frame in world (normal to surface)

    # Calculate signed distance of each point to the table plane
    # points_relative_to_table_origin = pc_world - table_origin_world  # Broadcasting if pc_world is (N,3)
    # distances_to_plane = np.dot(points_relative_to_table_origin, table_z_axis_world)
    
    # More explicit broadcasting for clarity:
    points_vec_from_table_origin = pc_world - table_origin_world.reshape(1, 3)
    distances_to_plane = np.sum(points_vec_from_table_origin * table_z_axis_world.reshape(1, 3), axis=1)


    # Define lower and upper bounds for the distance from the table plane
    # constants.OBJ_RAND_SCALE is the clearance above the table surface (dynamically set in generate_pose.py)
    lower_distance_bound = constants.OBJ_RAND_SCALE
    
    # Upper bound is the clearance + object's approximate height
    upper_distance_bound = constants.OBJ_RAND_SCALE + constants.APPROX_OBJECT_MAX_HEIGHT

    mask_z_plane = (distances_to_plane >= lower_distance_bound) & \
                   (distances_to_plane <= upper_distance_bound)
                   
    final_mask = mask_x & mask_y & mask_z_plane
    
    return final_mask
def to_pose(
    trans: Optional[np.ndarray] = None, rot: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert translation and rotation into a 4x4 pose matrix.

    Parameters
    ----------
    trans: Optional[np.ndarray]
        Translation vector, shape (3,).
    rot: Optional[np.ndarray]
        Rotation matrix, shape (3, 3).

    Returns
    -------
    np.ndarray
        4x4 pose matrix.
    """
    ret = np.eye(4)
    if trans is not None:
        ret[:3, 3] = trans
    if rot is not None:
        ret[:3, :3] = rot
    return ret

def transform_grasp_pose(
    grasp: Grasp,
    est_trans: np.ndarray,
    est_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_rot: np.ndarray,
) -> Grasp:
    """
    Transform grasp from the object frame into the robot frame

    Parameters
    ----------
    grasp: Grasp
        The grasp to be transformed.
    est_trans: np.ndarray
        Estimated translation vector in the camera frame.
    est_rot: np.ndarray
        Estimated rotation matrix in the camera frame.
    cam_trans: np.ndarray
        Camera's translation vector in the robot frame.
    cam_rot: np.ndarray
        Camera's rotation matrix in the robot frame.

    Returns
    -------
    Grasp
        The transformed grasp in the robot frame.
    """
    # raise NotImplementedError
    return Grasp(
        trans=cam_rot @ (est_rot @ grasp.trans + est_trans) + cam_trans,
        rot=cam_rot @ est_rot @ grasp.rot,
        width=grasp.width,
    )

def rand_rot_mat() -> np.ndarray:
    """
    Generate a random rotation matrix with shape (3, 3) uniformly.
    """
    while True:
        quat = np.random.randn(4)
        if np.linalg.norm(quat) > 1e-6:
            break
    quat /= np.linalg.norm(quat)
    return quat2mat(quat)


def theta_to_2d_rot(theta: float) -> np.ndarray:
    """
    Convert a 2D rotation angle into a rotation matrix.

    Parameters
    ----------
    theta : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The resulting 2D rotation matrix (2, 2).
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_dist(r1: np.ndarray, r2: np.ndarray) -> float:
    """
    The relative rotation angle between two rotation matrices.

    Parameters
    ----------
    r1 : np.ndarray
        The first rotation matrix (3, 3).
    r2 : np.ndarray
        The second rotation matrix (3, 3).

    Returns
    -------
    float
        The relative rotation angle in radians.
    """
    return np.arccos(np.clip((np.trace(r1 @ r2.T) - 1) / 2, -1, 1))

def get_pc(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth image into point cloud using intrinsics

    All points with depth=0 are filtered out

    Parameters
    ----------
    depth: np.ndarray
        Depth image, shape (H, W)
    intrinsics: np.ndarray
        Intrinsics matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        Point cloud with shape (N, 3)
    """
    # Get image dimensions
    height, width = depth.shape
    # Create meshgrid for pixel coordinates
    v, u = np.meshgrid(range(height), range(width), indexing="ij")
    # Flatten the arrays
    u = u.flatten()
    v = v.flatten()
    depth_flat = depth.flatten()
    # Filter out invalid depth values
    valid = depth_flat > 0
    u = u[valid]
    v = v[valid]
    depth_flat = depth_flat[valid]
    # Create homogeneous pixel coordinates
    pixels = np.stack([u, v, np.ones_like(u)], axis=0)
    # Convert pixel coordinates to camera coordinates
    rays = np.linalg.inv(intrinsics) @ pixels
    # Scale rays by depth
    points = rays * depth_flat
    return points.T
def get_pc_from_rgbd(rgb, depth, intrinsics, depth_scale=1000.0):
    """
    Generate point cloud from RGB-D image
    
    Parameters:
    -----------
    rgb : np.ndarray
        RGB image (H, W, 3)
    depth : np.ndarray  
        Depth image (H, W)
    intrinsics : np.ndarray
        Camera intrinsic matrix (3, 3)
    depth_scale : float
        Depth scale factor
        
    Returns:
    --------
    pc : np.ndarray
        Point cloud (N, 3) in camera frame
    colors : np.ndarray
        Point colors (N, 3)
    """
    height, width = depth.shape
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    
    # Convert depth to meters
    depth_flat = depth.flatten() / depth_scale
    
    # Filter out invalid depth values
    valid_mask = (depth_flat > 0.01) & (depth_flat < 10.0)
    # valid_mask = depth_flat > 0
    u = u[valid_mask]
    v = v[valid_mask]
    depth_flat = depth_flat[valid_mask]
    
    # Back-project to 3D
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x = (u - cx) * depth_flat / fx
    y = (v - cy) * depth_flat / fy
    z = depth_flat
    
    # Stack to point cloud
    pc = np.stack([x, y, z], axis=1)
    
    # Get colors
    if rgb is not None:
        colors = rgb[v, u] / 255.0
    else:
        colors = np.ones((len(pc), 3))
    
    return pc, colors

def filter_workspace_pc(pc, workspace_bounds=None):
    """
    Filter point cloud to workspace region
    """
    if workspace_bounds is None:
        # Default workspace bounds for table-top manipulation
        workspace_bounds = {
            'x': [-0.5, 0.5],
            'y': [-0.5, 0.5], 
            'z': [0.0, 1.0]
        }
    
    mask = (
        (pc[:, 0] >= workspace_bounds['x'][0]) & (pc[:, 0] <= workspace_bounds['x'][1]) &
        (pc[:, 1] >= workspace_bounds['y'][0]) & (pc[:, 1] <= workspace_bounds['y'][1]) &
        (pc[:, 2] >= workspace_bounds['z'][0]) & (pc[:, 2] <= workspace_bounds['z'][1])
    )
    
    return pc[mask]


def get_pc_mask(pc, workspace_bounds=None):
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > PC_MIN[2])
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask    

def get_workspace_mask(pc: np.ndarray) -> np.ndarray:
    """Get the mask of the point cloud in the workspace."""
    pc_mask = (
        (pc[:, 0] > PC_MIN[0])
        & (pc[:, 0] < PC_MAX[0])
        & (pc[:, 1] > PC_MIN[1])
        & (pc[:, 1] < PC_MAX[1])
        & (pc[:, 2] > PC_MIN[2])
        & (pc[:, 2] < PC_MAX[2])
    )
    return pc_mask

def downsample_pc(pc, num_points=1024):
    """
    Downsample point cloud to fixed number of points
    """
    if len(pc) >= num_points:
        indices = np.random.choice(len(pc), num_points, replace=False)
    else:
        indices = np.random.choice(len(pc), num_points, replace=True)
    
    return pc[indices]

def preprocess_pc_for_model(pc, num_points=1024, normalize=True):
    """
    Preprocess point cloud for pose estimation model
    """
    # Filter to workspace
    # pc_filtered = filter_workspace_pc(pc)
    pc_filtered = pc
    
    if len(pc_filtered) < 100:  # Too few points
        return None
        
    # Downsample
    pc_sampled = downsample_pc(pc_filtered, num_points)
    
    return pc_sampled
    # Normalize (optional - depends on training)
    # pc_sampled = pc_sampled - np.mean(pc_sampled, axis=0)
    
    centroid = np.mean(pc_sampled, axis=0)
    pc_centered = pc_sampled - centroid

    # Normalization (scale to unit sphere)
    if normalize:
        scale = np.max(np.linalg.norm(pc_centered, axis=1))
        if scale > 0:
            pc_normalized = pc_centered / scale
        else:
            pc_normalized = pc_centered
        return pc_normalized
    else:
        return pc_centered

def compute_rotation_matrix_from_ortho6d(ortho6d: torch.Tensor) -> torch.Tensor:
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    x = F.normalize(x_raw, dim=1)
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, dim=1)
    y = torch.cross(z, x, dim=1)

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), dim=2)
    return matrix

# Helper function to convert rotation matrix to 6D representation
def compute_ortho6d_from_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6) # (B, 6)