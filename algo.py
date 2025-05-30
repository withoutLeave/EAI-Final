import argparse
import numpy as np
import cv2
from pyapriltags import Detector
import os
from src.utils import to_pose, rot_dist
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv

from scipy.spatial.transform import Rotation as R

def from_dog_frame_to_world(dog_pose, obj_local):
    """
    将物体在狗坐标系下的7D位姿转换到世界坐标系。
    dog_pose和obj_local均为7D格式：[x,y,z, qx,qy,qz,qw]
    """
    p_dog = dog_pose[:3]
    q_dog = dog_pose[3:]  # [x,y,z,w]
    p_obj_local = obj_local[:3]
    q_obj_local = obj_local[3:]  # [x,y,z,w]
    r_dog = R.from_quat(q_dog)
    r_obj_local = R.from_quat(q_obj_local)

    # 位置变换：将obj_local的位置旋转到世界坐标系，再加上狗的位置
    p_world = p_dog + r_dog.apply(p_obj_local)

    # 姿态变换：世界坐标系下的旋转 = 狗的旋转 * 物体在狗坐标系的旋转
    r_world = r_dog * r_obj_local
    q_world = r_world.as_quat()  # [x,y,z,w]
    return np.concatenate([p_world, q_world])



def pose7d_to_T(pose7d):
    t = np.array(pose7d[:3])
    q = np.array(pose7d[3:7])  # 输入四元数 [x,y,z,w]

    rot = R.from_quat(q).as_matrix()  # scipy默认四元数格式是 [x,y,z,w]
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T

def T_to_pose7d(T):
    t = T[:3, 3]
    rot_mat = T[:3, :3]
    q = R.from_matrix(rot_mat).as_quat()  # 输出 [x,y,z,w]
    return np.concatenate([t, q])

def pred_dog_pose_in_world(tag_pose_world, tag_pose_dog):
    """
    计算狗在世界坐标系下的位姿:
    输入tag_pose_world和tag_pose_dog均为7D，格式：
    [x,y,z, qx, qy, qz, qw]
    """
    tag_pose_world_T = pose7d_to_T(tag_pose_world)
    tag_pose_dog_T = pose7d_to_T(tag_pose_dog)
    tag_pose_dog_inv = np.linalg.inv(tag_pose_dog_T)
    dog_pose_world = np.dot(tag_pose_world_T, tag_pose_dog_inv)
    return dog_pose_world

