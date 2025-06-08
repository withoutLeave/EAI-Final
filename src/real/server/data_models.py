from dataclasses import dataclass
import numpy as np

@dataclass
class CameraData:
    color_image: np.ndarray
    depth_image: np.ndarray
    intrinsics: list
    pose: list  # [x, y, z, qx, qy, qz, qw]

@dataclass
class HumanoidData:
    head_camera: CameraData
    wrist_camera: CameraData
    joint_angles: dict  # {'arm_left': [q1,q2,...], ...}
    timestamp: float