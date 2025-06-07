import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2,torch
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R

from src.vis import Vis
from src.type import Grasp
from src.utils import to_pose, rot_dist
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
from src.robot.cfg import get_robot_cfg

# predict mask
from src.utils import get_workspace_mask_height
from src.utils import get_pc_from_rgbd
from src.model.est_pose import EstPoseNet
from src.model.est_coord import EstCoordNet
from src.config import Config

import traceback





def T_to_pose7d(T):
    t = T[:3, 3]
    rot_mat = T[:3, :3]
    q = R.from_matrix(rot_mat).as_quat()  # 输出 [x,y,z,w]
    return np.concatenate([t, q])

np.set_printoptions(precision=4, suppress=True)

'''#################################################################
以下是控制狗的位置的变量
'''
THR1 = 1.05 # if dog x < 0.8: turn; container
FLAG_COME_TURN = False # [0,0,0 ,1] to [-0.7071,0.,0.,0.7071], 到达阈值后将 flag_come_trun 设置为 1
dog_ready = False
FLAG_GO_TURN = False 
TURN_OVER = False
turn_quat = 0.,0.,0.707,0.707
quad_move_traj = [] # store all the quad_command to reverse and roll out
'''Pose detection'''
# For pose detection


COORD_MODEL_DIR = "./models/est_coord/checkpoint_21500.pth"
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

        pose_checkpoint = torch.load(POSE_MODEL_DIR, map_location=DEVICE,weights_only=True)
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
        coord_checkpoint = torch.load(COORD_MODEL_DIR, map_location=DEVICE,weights_only=True)
        COORD_MODEL.load_state_dict(coord_checkpoint['model'])
        COORD_MODEL.to(DEVICE)
        COORD_MODEL.eval()
        print("Loaded EstCoordNet model")
    except Exception as e:
        print(f"Failed to load EstCoordNet: {e}")
        COORD_MODEL = None


def detect_driller_pose(img, depth, camera_matrix, camera_pose, *args, **kwargs):
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
        cv2.imwrite('rgb.png', img)
        full_pc_camera = pc_camera
        # full_pc_world = pc_world
        full_pc_world = (
            np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
            + camera_pose[:3, 3]
        )

        pc_mask = get_workspace_mask_height(full_pc_world)
        # pc_mask = np.ones(full_pc_world.shape[0], dtype=bool)
        
        sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)
        plotly_list = []
        plotly_list += Vis.pc(
            full_pc_world[pc_mask],
            size=3,
            color='blue',
        )
        plotly_list += Vis.pose(camera_pose[:3, 3], camera_pose[:3, :3], length=0.1)
        Vis.show(plotly_list, path="output/vis_sample_0.html") 
        # Vis.show(plotly_list)
        pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
        pc_tensor = torch.from_numpy(pc_camera).float().unsqueeze(0).to(DEVICE)
        print(f"pc_tensor shape: {pc_tensor.shape}")

        
        # Try EstCoordNet first (usually better performance)
        if COORD_MODEL is not None:
            try:
                est_trans, est_rot = COORD_MODEL.est(pc_tensor)
                est_trans = est_trans.cpu().numpy().squeeze()
                est_rot = est_rot.cpu().numpy().squeeze()
                
                # Convert from camera frame to world frame
                world_trans = camera_pose[:3, :3] @ est_trans + camera_pose[:3, 3]
                world_rot = camera_pose[:3, :3] @ est_rot
                
                # Construct pose matrix
                pose = np.eye(4)
                pose[:3, :3] = world_rot
                pose[:3, 3] = world_trans
                
                print("Used EstCoordNet for pose estimation")
                # print(f"Pose: {pose}")
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
                # print(f"camera_pose: {camera_pose}")
                # print(f"est_rot: {est_rot}")
                # print(f"est_trans: {est_trans}")
                print("Used EstPoseNet for pose estimation")
                # print(f"Pose: {pose}")
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
'''Pose detection'''

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
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detections_list = detector.detect(gray_image, 
                                    estimate_tag_pose=True, 
                                    camera_params=camera_params, 
                                    tag_size=tag_size) # 使用之前定义的 tag_physical_size

    if len(detections_list) == 0:
        return None,None
    else:
        tag_detection=detections_list[0]        
        T_cam_tag = np.eye(4)
        T_cam_tag[0:3, 0:3] = tag_detection.pose_R
        # print("tag_detection.pose_R",tag_detection.pose_R)
        T_cam_tag[0:3, 3] = tag_detection.pose_t[:, 0]
        M = np.array([
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, -1]
        ])
        A = np.eye(4) # 转成4x4
        A[:3, :3] = M
        trans_marker_world =  np.dot(camera_pose, T_cam_tag)[:3,3]
        T_cam_tag = T_cam_tag @ A
        T_world_tag = np.dot(camera_pose, T_cam_tag)         
        rot_marker_world = T_world_tag[:3,:3]

    return trans_marker_world, rot_marker_world


def is_desired_qpos_for_post_grasp(qpos_arr: np.ndarray) -> bool:
    """
    检查给定的 7 自由度 qpos 是否满足抓取后期望的条件。
    qpos_arr: 机械臂的 7 自由度关节角度数组。
    """
    if qpos_arr.shape[0] != 7:
        print(f"警告: qpos 数组维度不为 7, 实际为 {qpos_arr.shape[0]}")
        return False # 如果维度不对，直接返回 False

  
    cond1 = (qpos_arr[0] >= -0.7) and (qpos_arr[0] <= -0.4)
    cond2 = (qpos_arr[1] >= -1.2) and (qpos_arr[2] <= 0.8)
    cond3 = (qpos_arr[2] >= -0.25) and (qpos_arr[2] <= 0.2)
    cond6 = (qpos_arr[5] >= -0.5) and (qpos_arr[5] <= 0.5)
    cond7 = qpos_arr[6] < 0

    return cond1 and cond2 and cond3 and cond6 and cond7

def is_desired_qpos_for_post_grasp1(qpos_arr: np.ndarray) -> bool:
    # ... (维度检查不变)

    
    cond1 = (qpos_arr[0] >= -1.0) and (qpos_arr[0] <= -0.2)
    cond3 = (qpos_arr[2] >= -0.5) and (qpos_arr[2] <= 0.5)
    cond2 = (qpos_arr[1] >= -1.2) and (qpos_arr[2] <= 0.8) # cond2 保持不变
    cond6 = (qpos_arr[5] >= -0.5) and (qpos_arr[5] <= 0.5) # cond6 保持不变
    cond7 = qpos_arr[6] < 0                               # cond7 保持不变

    return cond1 and cond2 and cond3 and cond6 and cond7
def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    robot_cfg = get_robot_cfg("galbot")
    robot_model = env.humanoid_robot_model
    current_qpos = env.get_state()[:7]
    # get necessary info

    _, depth = robot_cfg.gripper_width_to_angle_depth(grasp.width)
    grasp_trans = grasp.trans - depth * grasp.rot[:, 0]

    succ, grasp_arm_qpos = robot_model.ik(trans= grasp_trans,rot= grasp.rot,init_qpos = current_qpos)
    if not succ:
        return None
    traj = [grasp_arm_qpos]

    cur_trans, cur_rot, cur_qpos = (
        grasp_trans.copy(),
        grasp.rot.copy(),
        grasp_arm_qpos.copy(), # 7
    )
    for _ in range(grasp_config["reach_steps"]):
        cur_trans = cur_trans - grasp_config["delta_dist"] * cur_rot[:, 0]# 将目标靠近现有位置
        succ, cur_qpos = robot_model.ik(
            cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
        )
        if not succ: return None
        traj = [cur_qpos] + traj  


    '''对轨迹的加强'''
    final_grasp_qpos = traj[-1] # check final qpos
    if not is_desired_qpos_for_post_grasp1(final_grasp_qpos):
        print(f"The grasp plan is not good: because a loop is included!")
        return None # 返回 None，指示这个轨迹不合格 """
    
    return traj


def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)

def open_gripper(env: WrapperEnv, steps = 30):
    for _ in range(steps):
        env.step_env(gripper_open=1)

def close_gripper(env: WrapperEnv, steps = 30):
    for _ in range(steps):
        env.step_env(gripper_open=0)

def execute_plan(env, gra_plan: np.ndarray,obj_pose,plan_type:int) -> bool:
    """Execute the planned trajectory in the simulation and check if the grasp was successful."""
    if plan_type == 1: 
        '''特定针对抓取的执行方式，包括返回抓取的起点'''
        open_gripper(env)
        # succ_height_thresh = 0.18
        # grasp
        initial_ok = False
        dist = 0
        for _ in range(200):
            env.step_env(
                humanoid_action=gra_plan[0][:7], 
            )
            p_xyz=env.humanoid_robot_model.fk_link(gra_plan[0],env.humanoid_robot_cfg.link_eef)[0]
            c_xyz = env.humanoid_robot_model.fk_link(env.get_state(),env.humanoid_robot_cfg.link_eef)[0]
            dist = np.linalg.norm(p_xyz - c_xyz)
            if abs(p_xyz[0]-c_xyz[0])<0.01 and abs(p_xyz[1]-c_xyz[1])<0.01 and abs(p_xyz[2]-c_xyz[2])<0.01:
                initial_ok = True
                break
        print(f"In the prepare stage success {initial_ok}, distance is {dist}.")
        for qpos in gra_plan:
            for _ in range(3):
                env.step_env(
                    humanoid_action=qpos[:7], 
                )
        # gripper close
        close_gripper(env)
        qpos = env.get_state()
        print(f" grasp arm qpos is {qpos}.")
        # lift
        gra_plan.reverse()
        for qpos in gra_plan:
            for _ in range(3):
                env.step_env(
                    humanoid_action=qpos[:7], 
                )
        for _ in range(5):
            env.step_env(
                    humanoid_action=gra_plan[-1][:7], 
                )
            return True
    elif plan_type == 2:# move and drop, grasp_plan is plan for move, lift_plan is None
        '''使用较少的步数'''
        for qpos in gra_plan: 
            for _ in range(8):
                env.step_env(
                    humanoid_action=qpos[:7], 

                )
            qpos = env.get_state()
        open_gripper(env)
    elif plan_type ==3: # here gra_plan is [,7]
        '''特定针对只有没有中间过程的规划'''
        for i in range(200):
            env.step_env(
                    humanoid_action=gra_plan, 
                )
    elif plan_type ==4: # here gra_plan is [,7]
        '''移动一小段'''
        for i in range(5):
            env.step_env(
                    humanoid_action=gra_plan, 
                )

def execute_plan_for_pose(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )

TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False # if false, Dog can come near to the robot
DISABLE_RETURN = True
def main():

    global quad_move_traj,DISABLE_MOVE # used to record all quad commands

    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0) # 暂时不显式
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=2)
    parser.add_argument("--try_plan_num", type=int, default=3) # for each grasp, find ik

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
    
    pred_wrist_obs =None
    head_init_qpos = np.array([0.,0.3]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    

    # observing_qpos = humanoid_init_qpos + np.array([0.01,0,0.40,0,0,0,0.15])
    # 移动手腕并拍照
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0.20,0,0,0,0.15]) # you can customize observing qpos to get wrist obs

    grasp_init_qpos = humanoid_init_qpos + np.array([0.01,0,0.15,0,0,0,0.15])
    init_plan = plan_move_qpos(humanoid_init_qpos, observing_qpos)
    execute_plan_for_pose(env, init_plan)
    obs_head = env.get_obs(camera_id=0) # head camera
    pred_wrist_obs = env.get_obs(camera_id=1) # wrist camera

    env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
    env.debug_save_obs(pred_wrist_obs, 'data/obs_wrist')

    '''Back the gripper pose'''
    backplan = plan_move_qpos(observing_qpos,grasp_init_qpos)
    execute_plan_for_pose(env, backplan)

    '''以下是过程中两个非常重要的变量'''
    pose_container_world = None # Used when move and drop object
    quad_initial_pose = None # return to here finally
    get_ini_quad_pose = False
    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        print("*"*80,"\nStage 1: move dog near the robot.")
        global dog_ready,FLAG_COME_TURN,TURN_OVER,THR1


  
             
        align_y = False
        align_over = False
        count_y = 0
        y_setting = -0.17  # 规范化 y 到 -0.2
        head_qpos = head_init_qpos
        forward_steps = 700 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])


        for step in range(forward_steps):
            move_head = False
            if dog_ready: 
                print(f"After {step} steps, The dog is ready, with container xyz {pose_container_world[:3,3]}")
                break # OK

            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                # env.debug_save_obs(obs_head, f'data/obs_head_OK')
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                
                if trans_marker_world is not None: # in the world 
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0.31,0.,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    if not get_ini_quad_pose: # haven't get quad init pose, only save once
                        get_ini_quad_pose = True
                        quad_initial_pose = pose_container_world
                    '''暂时设置为真实情况'''
                    # pose_container_world = algo.pose7d_to_T(real_container_pose)
                    container_7dim = T_to_pose7d(pose_container_world)
                else: # No AprilTag detected: Try to change human head pose
                    move_head = True 
                    
            y_tol = 0.03
            if np.linalg.norm(container_7dim[0]-1.3)<=0.03 and np.linalg.norm(container_7dim[1]-y_setting)>y_tol and not align_y:
                '''需要在这个特定时刻对齐 y; 对齐基准x = 1.70, y = -0.15'''
                align_y = True
                print(f"Need to align y! Now y is {container_7dim[1]} .")
            elif np.linalg.norm(container_7dim[1]-y_setting)<= y_tol and align_y and not align_over:
                align_over = True
                '''对齐结束，恢复前进过程'''
                print(f"After {count_y} times alignment steps. Align over! Now y is {container_7dim[1]} .")
            elif align_y and not align_over:
                '''对齐的过程中,y 和 -0.15 对齐'''
                count_y +=1
                y_dist = np.linalg.norm(container_7dim[1]-y_setting)
                if container_7dim[1]-y_setting > 0: 
                    quad_command = [0,y_dist*3,0] # 机器人偏左
                else:
                    quad_command = [0,-y_dist*3,0]
                quad_move_traj.append(quad_command)
                env.step_env(
                    quad_command=quad_command
                )
                # if step%3==0:
                #     print(container_7dim)
                continue
           

            if move_head:
                ''' Heap qpos: horizontal:[-1.57,1.57], negative -> turn right; vertical: [-0.366,0.366], negative -> up'''
                if FLAG_COME_TURN and head_qpos[1] >= 0.366:
                    head_qpos[0]-=0.05
                    print("Turn the head to most down, But no Tag detected! Turn head to right!")
                head_qpos[1] = min(head_qpos[1]+0.1,0.366)
                quad_command = [0,0,0]                    
                quad_move_traj.append(quad_command)
                print(f"Move head to {head_qpos}!")
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
                continue
            if not FLAG_COME_TURN: # come colser
                if pose_container_world[0,3]<THR1: # x near
                    FLAG_COME_TURN = True
                    print("Come to the place, next step is turn over!")
                else:
                    quad_command = [0.18,0,0] # move forward
            elif FLAG_COME_TURN and not TURN_OVER: # turn 90 degree
                turn_rot = R.from_quat(turn_quat).as_matrix()
                if rot_dist(turn_rot,pose_container_world[:3,:3]) > 0.15:
                    # print(rot_dist(turn_rot,pose_container_world[:3,:3]))
                    quad_command = [0,0,0.5]
                else:
                    TURN_OVER = True
                    quad_command = [0,0,0]
                    print("Turn over!")
            elif TURN_OVER:  # 侧身走过来
                if pose_container_world[0,3] < 0.68:
                    dog_ready = True
                    print("The first stage is completed by approaching sideways!")
                    quad_command = [0,0,0]
                else:
                    quad_command = [0,-0.08,0]
            # is_close(pose_container_world, target_container_pose, threshold=0.05)
            quad_move_traj.append(quad_command)
            env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
    '''盲走一段路，让箱子更加靠近'''

    if not DISABLE_MOVE:  
        def fine_get_closer():
            for i in range(45):
                quad_command=[-0.2,0,0]
                quad_move_traj.append(quad_command)
                env.step_env(quad_command=quad_command)
            env.step_env(quad_command=[0,0,0])  
        
        fine_get_closer()
        assert(pose_container_world is not None)


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        print("*"*80)
        print("Stage 2: Predict the driller pose in the world frame by PC.")

        rgb, depth, camera_pose = pred_wrist_obs.rgb, pred_wrist_obs.depth, pred_wrist_obs.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        # table_pose = env.config.table_pose


        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose)
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)

    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        print("*"*80,"\nStage 3: grasp and lift task begin")
        # 预先设置：
        print("The driller pose is ", T_to_pose7d(driller_pose))
        obj_pose = driller_pose.copy()
        # obj_pose1 = env.get_driller_pose()

        print(f"Object pose is {T_to_pose7d(obj_pose)}!")
        grasps = get_grasps(args.obj)  # 得到了 8 个 grasp

        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n]
        # grasps[0] 应该是可以成功的
        est_trans = obj_pose[:3,3]
        est_rot = obj_pose[:3,:3]
        est_trans[2] -= 0.05 # 让它更低一点，夹得更稳
        grasp_config = dict( 
            reach_steps=20,
            delta_dist=0.01
        ) 
        solve_grasp_ik = 0
        for obj_frame_grasp in valid_grasps:
            for _ in range(10): 
                robot_frame_grasp = Grasp(
                    trans = est_rot @ obj_frame_grasp.trans + est_trans, # 这里不加 perturb_trans，因为在循环外面已经定义好了
                    rot   = est_rot @ obj_frame_grasp.rot,
                    width = obj_frame_grasp.width,
                )
                solve_grasp_ik += 1
                # 调用 plan_grasp。如果它返回 None，会继续下一个 try_plan_num 循环
                gra_plan = plan_grasp(env=env, grasp=robot_frame_grasp, grasp_config=grasp_config)
                if gra_plan is not None: # 如果找到了符合条件的计划
                    print(f"After {solve_grasp_ik} time ik solve, find a good grasp candidate.")
                    break # 跳出 inner loop (try_plan_num)
                   
            if gra_plan is not None: # 如果找到了符合条件的计划
                break # 跳出 outer loop (obj_frame_grasp_candidate)
        
        if gra_plan is not None:
            # 这里不确定是否抓取成功
            execute_plan(env,gra_plan=gra_plan,obj_pose=obj_pose,plan_type= 1)
            DISABLE_MOVE = False

    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        print("*"*80)
        print("Stage 4: Move the object to a specific pose!")
        # implement your moving plan
        # current_gripper_trans, current_gripper_rot = env.humanoid_robot_model.fk_link(env.sim.mj_data.qpos[env.sim.qpos_humanoid_begin:env.sim.qpos_humanoid_begin+7], env.humanoid_robot_cfg.link_eef) # 正向运动学获取末端执行器位姿

        qpos = env.get_state()
        print(f" humanoid arm qpos is {qpos}.")

        # move_plan=np.array([ 0.9371,-1.5984 ,-1.4545, -0.5024, -0.0948, -0.2711,  0.8]) # 这是一个可行的 qpos，但是路径最后夹爪的角度不合适
        move_traj = np.array([
            [-0.7541, -1.5842,  0.1683, -1.5229, -0.9842, -0.2657, -0.5892],
            [-0.7, -1.5842,  0, -1.5229, -0.9842, -0.2657, -0.5892],
            [-0.7, -1.5842,  -0.2, -1.4229, -0.9842, -0.2657, -0.5892],
            [ -0.6, -1.5842,  -0.4, -1.4229, -0.9842, -0.2657, -0.5892],
            [ -0.6, -1.5842,  -0.7, -1.4229, -0.9842, -0.2657, -0.5892],
        ])

        def interpolate_joint_trajectory(traj, num_interp=10):
            """
            对机械臂的关节角度序列进行线性插值，每对相邻帧之间插入若干中间帧。
            """
            traj = np.asarray(traj)
            interpolated = []

            for i in range(len(traj) - 1):
                start = traj[i]
                end = traj[i + 1]

                for j in range(num_interp + 1):  # 包含起始点，但不包含终点
                    t = j / (num_interp + 1)
                    interp = (1 - t) * start + t * end
                    interpolated.append(interp)

            interpolated.append(traj[-1])  # 最后一个关键帧
            return np.array(interpolated)
        
        move_traj = interpolate_joint_trajectory(move_traj)
        execute_plan(env, gra_plan=move_traj, obj_pose=None, plan_type = 2)
        open_gripper(env)
        # print(f"Open gripper at: {env.humanoid_robot_model.fk_link(env.sim.mj_data.qpos[env.sim.qpos_humanoid_begin:env.sim.qpos_humanoid_begin+7], env.humanoid_robot_cfg.link_eef)[0]} \
        #     should open at : {env.humanoid_robot_model.fk_link(move_plan, env.humanoid_robot_cfg.link_eef)[0]}")
        '''倒放轨迹，移开手臂'''
        move_traj = np.flip(move_traj)
        execute_plan(env, gra_plan=move_traj, obj_pose=None, plan_type = 2)

    DISABLE_RETURN = False # let the dog return 
    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_RETURN:
        # implement
        print("*"*80)
        print("Stage 5: Make the dog return to the initial position.")
        '''把轨迹倒放一遍，之后结合视觉微调'''
        backward_steps = 400 # customize by yourselves
        num_come_steps = len(quad_move_traj)
        quad_move_traj.reverse()
        print(f"when come to the target pose, use {num_come_steps}, now reverse all the command to roll out.")
        for i in range(num_come_steps):
            quad_command = quad_move_traj[i]
            quad_command = [ -x for x in quad_move_traj[i] ] 
            # print(np.shape(quad_move_traj),quad_command)
            env.step_env(
                quad_command=quad_command
            )

        reset_head_qpos = [0,0.1]
        print(f"Change head pose to {reset_head_qpos}.")
        env.step_env(
                quad_command=[0,0,0],
                humanoid_head_qpos= reset_head_qpos
            )

        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])
        obs_head = env.get_obs(camera_id=0)
        env.debug_save_obs(obs_head, 'data/obs_final') # obs has rgb, depth, and camera pose
        trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
        if trans_marker_world is not None:
            current_tag_pose = to_pose(trans_marker_world, rot_marker_world)
            print(f"After roll out, fine tune the final position to the task initial pose. Current xyz is {current_tag_pose[:3,3]}, goal xyz is {quad_initial_pose[:3,3]}.")
        else: 
            print("Can't find the dog!")
        final_rot = [0.,0.,0.,1.]
        final_rot = R.from_quat(final_rot).as_matrix()


        for step in range(backward_steps): # 这里需要保证精确性
            obs_head = env.get_obs(camera_id=0) # head camera
            trans_marker_world, rot_marker_world = detect_marker_pose(
                detector, 
                obs_head.rgb, 
                head_camera_params,
                obs_head.camera_pose,
                tag_size=0.12
            )
            
            if trans_marker_world is not None: # in the world 
                trans_container_world = rot_marker_world @ np.array([0.31,0.,0.02]) + trans_marker_world
                rot_container_world = rot_marker_world
                pose_container_world = to_pose(trans_container_world, rot_container_world)
                container_7dim = T_to_pose7d(pose_container_world)


                dist_x= np.linalg.norm(container_7dim[0]-quad_initial_pose[0,3])
                dist_y= np.linalg.norm(container_7dim[1]-quad_initial_pose[1,3])
                con_x = np.where(container_7dim[0]-quad_initial_pose[0,3]> 0, dist_x, - dist_x)
                con_y = np.where(container_7dim[1]-quad_initial_pose[1,3]> 0, dist_y, - dist_y)
                quad_command = [con_x,con_y,0]
                # print(dist_x,dist_y)
                env.step_env(
                    quad_command=quad_command
                )
                if (dist_x<0.01) and dist_y<0.01:
                    print("Dog has come to the initial position. Task finished!!!")
                    break
            else: 
                
                head_qpos[1] = min (0, head_qpos[1]-0.05)
                print(f"Can't see the dog! See further: {head_qpos}.")
                env.step_env(
                    humanoid_head_qpos= head_qpos
                )

        
    print("*"*80)
    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()