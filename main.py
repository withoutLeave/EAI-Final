import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data

# For pose detection
from src.utils import get_pc_from_rgbd, preprocess_pc_for_model
from src.model.est_pose import EstPoseNet
from src.model.est_coord import EstCoordNet
from src.config import Config
from transforms3d.quaternions import mat2quat, quat2mat
from src.utils import to_pose,rot_dist,get_pc,get_workspace_mask,get_workspace_mask_pose,get_workspace_mask_height
from src import constants

import torch,cv2
from src.constants import DEPTH_IMG_SCALE
import traceback
from src.vis import Vis

COORD_MODEL_DIR = "./models/est_coord/checkpoint_21500.pth"
COORD_MODEL_DIR =""
POSE_MODEL_DIR = "./models/est_pose/checkpoint_24000.pth"
POSE_MODEL = None
COORD_MODEL = None
DEVICE = None

def load_models():
    """
    Load pre-trained pose estimation models
    """
    global POSE_MODEL, COORD_MODEL, DEVICE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load EstPoseNet model
    try:
        POSE_MODEL = EstPoseNet(config=Config())
        # pose_checkpoint = torch.load('/home/zhujl/Assignment2/exps/est_pose/checkpoint/checkpoint_10000.pth', 
        #                            map_location=DEVICE)
        pose_checkpoint = torch.load(POSE_MODEL_DIR, map_location=DEVICE)
        POSE_MODEL.load_state_dict(pose_checkpoint['model'])
        POSE_MODEL.to(DEVICE)
        POSE_MODEL.eval()
        print("Loaded EstPoseNet model")
    except Exception as e:
        print(f"Failed to load EstPoseNet: {e}")
        POSE_MODEL = None
    
    # Load EstCoordNet model  
    try:
        COORD_MODEL = EstCoordNet(config=Config())
        coord_checkpoint = torch.load(COORD_MODEL_DIR, map_location=DEVICE)
        COORD_MODEL.load_state_dict(coord_checkpoint['model'])
        COORD_MODEL.to(DEVICE)
        COORD_MODEL.eval()
        print("Loaded EstCoordNet model")
    except Exception as e:
        print(f"Failed to load EstCoordNet: {e}")
        COORD_MODEL = None


def detect_driller_pose(img, depth, camera_matrix, camera_pose, table_pose,*args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    # 
    global POSE_MODEL, COORD_MODEL, DEVICE
    
    # Load models if not already loaded
    if POSE_MODEL is None and COORD_MODEL is None:
        load_models()
    try:
        # Generate point cloud from RGB-D
        pc_camera, colors = get_pc_from_rgbd(img, depth, camera_matrix,depth_scale=1.0) # meter
        # pc_camera = get_pc(depth,camera_matrix) 
       
        # # Transform to world coordinates for workspace filtering
        pc_world = (camera_pose[:3, :3] @ pc_camera.T + camera_pose[:3, 3:]).T
        
        # # Preprocess point cloud
        # pc_processed = preprocess_pc_for_model(pc_camera, num_points=1024).astype(np.float32)
        # # print(f"Processed point cloud shape: {pc_processed.shape}")
        # if pc_processed is None:
        #     print("Warning: Failed to preprocess point cloud")
        #     print(f"Point cloud shape: {pc_camera.shape}")
        #     print(f"depth min: {depth.min()}, max: {depth.max()}, mean: {depth.mean()}")
        #     return np.eye(4)
        
        # # Convert to torch tensor
        # pc_tensor = torch.from_numpy(pc_processed).float().unsqueeze(0).to(DEVICE)

        # show_point_cloud(pc_camera)
        # show_point_cloud(pc_tensor.cpu().numpy()[0])
        # cv2.imshow("rgb", img)
        cv2.imwrite('rgb.png', img)
        # full_pc_camera = get_pc(
        #     depth,camera_matrix
        # ) * np.array([-1, -1, 1])
        # # show_point_cloud(full_pc_camera)
        # full_pc_world = (
        #     np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
        #     + camera_pose[:3, 3]
        # )
        full_pc_camera = pc_camera
        # full_pc_world = pc_world
        full_pc_world = (
            np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
            + camera_pose[:3, 3]
        )
        # full_coord = np.einsum(
        #     "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
        # )

        # pc_mask = get_workspace_mask(full_pc_world)
        # pc_mask = get_workspace_mask_pose(full_pc_world, table_pose)
        pc_mask = get_workspace_mask_height(full_pc_world)
        # pc_mask = get_workspace_mask_histogram(full_pc_world)
        # pc_mask = np.ones(full_pc_world.shape[0], dtype=bool)
        
        sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)
        # plotly_list = []
        # plotly_list += Vis.pc(
        #     full_pc_world[pc_mask],
        #     size=3,
        #     color='blue',
        # )
        # plotly_list += Vis.pose(camera_pose[:3, 3], camera_pose[:3, :3], length=0.1)
        # Vis.show(plotly_list, path="output/vis_sample_0.html") 
        # Vis.show(plotly_list)
        pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
        pc_tensor = torch.from_numpy(pc_camera).float().unsqueeze(0).to(DEVICE)
        print(f"pc_tensor shape: {pc_tensor.shape}")
        # show_point_cloud(pc_camera)
        # pc_mask = np.ones(full_pc_world.shape[0], dtype=bool)
        
        # Try EstCoordNet first (usually better performance)
        if COORD_MODEL is not None:
            try:
                est_trans, est_rot = COORD_MODEL.est(pc_tensor)
                est_trans = est_trans.cpu().numpy().squeeze()
                est_rot = est_rot.cpu().numpy().squeeze()
                
                # Convert from camera frame to world frame
                print(f"camera_pose: {camera_pose}")
                print(f"est_rot: {est_rot}")
                world_trans = camera_pose[:3, :3] @ est_trans + camera_pose[:3, 3]
                world_rot = camera_pose[:3, :3] @ est_rot
                
                # Construct pose matrix
                pose = np.eye(4)
                pose[:3, :3] = world_rot
                pose[:3, 3] = world_trans
                
                print("Used EstCoordNet for pose estimation")
                print(f"Pose: {pose}")
                return pose
            except Exception as e:
                traceback.print_exc()
                print(f"EstCoordNet failed: {e}")
        
        # Fallback to EstPoseNet
        if POSE_MODEL is not None:
            try:
                est_trans, est_rot = POSE_MODEL.est(pc_tensor)
                est_trans = est_trans[0].cpu().numpy().squeeze()
                est_rot = est_rot[0].cpu().numpy().squeeze()
                
                # Convert from camera frame to world frame
                world_trans = camera_pose[:3, :3] @ est_trans + camera_pose[:3, 3]
                world_rot = camera_pose[:3, :3] @ est_rot
                
                # Construct pose matrix
                pose = np.eye(4)
                pose[:3, :3] = world_rot
                pose[:3, 3] = world_trans
                print(f"camera_pose: {camera_pose}")
                print(f"est_rot: {est_rot}")
                print(f"est_trans: {est_trans}")
                print("Used EstPoseNet for pose estimation")
                print(f"Pose: {pose}")
                return pose
                
            except Exception as e:
                traceback.print_exc()
                print(f"EstPoseNet failed: {e}")
        
        print("Warning: All models failed, returning identity pose")
        return np.eye(4)
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error in detect_pose: {e}")
        return np.eye(4)
    return pose

def detect_marker_pose(
        detector: Detector, 
        img: np.ndarray, 
        camera_params: tuple, 
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # implement
    trans_marker_world = None
    rot_marker_world = None
    return trans_marker_world, rot_marker_world

def forward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped to position where you drop the driller """
    # implement
    action = np.array([0,0,0])
    return action

def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    action = np.array([0,0,0])
    return action

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']

    traj_reach = []
    traj_lift = []
    succ = False
    if not succ: return None

    return [np.array(traj_reach), np.array(traj_lift)]

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, *args, **kwargs):
    """Plan a trajectory moving the driller from table to dropping position"""
    # implement
    traj = []

    succ = False
    if not succ: return None
    return traj

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)
def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)
def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = True

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)
    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    
    head_init_qpos = np.array([0.,0.12]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)

    # print("table_pose:", env.config.table_pose, type(env.config.table_pose))
    TABLE_HEIGHT = float(env.config.table_pose[2][3])
    # constants.TABLE_HEIGHT = TABLE_HEIGHT
    # constants.PC_MIN[2] = TABLE_HEIGHT + constants.OBJ_RAND_SCALE
    print(f"TABLE_HEIGHT: {constants.TABLE_HEIGHT}")
    print(f"{constants.PC_MIN=}, {constants.PC_MAX=}")


    observing_qpos = humanoid_init_qpos + np.array([0.01, 0., 0.18, 0., 0., 0. , 0.15]) # you can customize observing qpos to get wrist obs
    # observing_qpos = humanoid_init_qpos + np.array([0.01, 0., 0.,0., 0.35, 0., 0.]) # you can customize observing qpos to get wrist obs
    # observing_qpos = humanoid_init_qpos + np.array([0.01,0,0,0,0.,0.,0]) # you can customize observing qpos to get wrist obs
    # target_qpos = move_wrist_camera_higher(env, delta_z=0.001) # move wrist camera higher to get better view
    # print(f"target_qpos: {target_qpos}")
    # print(env.sim.humanoid_robot_cfg.joint_init_qpos)
    # print(env.sim.humanoid_robot_cfg.joint_names)
    # return
    # init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos, steps = 20) 
    # original code seems wrong
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos, steps=20)
    execute_plan(env, init_plan)

    for _ in range(5):
        env.step_env()


    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1000 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        target_container_pose = [] 
        def is_close(pose1, pose2, threshold): return True
        for step in range(forward_steps):
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                if trans_marker_world is not None:
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)

            quad_command = forward_quad_policy(pose_container_world, target_container_pose)
            move_head = False
            if move_head:
                # if you need moving head to track the marker, implement this
                head_qpos = [0,0]
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                env.step_env(quad_command=quad_command)
            if is_close(pose_container_world, target_container_pose, threshold=0.05): break


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:      
        table_pose = env.config.table_pose
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        print(f"wrist camera pose after: {obs_wrist.camera_pose}")
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        print(f'camera_pose: {camera_pose}')
        # return
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        print(f"wrist_camera_matrix: {wrist_camera_matrix}")
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose,table_pose)
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=0,
            lift_steps=0,
            delta_dist=0, 
        ) # the grasping design in assignment 2, you can choose to use it or design yours

        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans
                + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan

        pregrasp_plan = plan_move_qpos(env, observing_qpos, reach_plan[0], steps=50) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        move_plan = plan_move(
            env=env,
        ) 
        execute_plan(env, move_plan)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 1000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            quad_command = backward_quad_policy()
            env.step_env(
                quad_command=quad_command
            )
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()