import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose, rot_dist
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
import algo
from scipy.spatial.transform import Rotation as R



'''
目前还有一个问题，头低到最低，在狗接近时也看不到 Tag，我们不放让狗先转 90度，然后侧着靠近。


如果需要考虑 y 轴的泛化能力，需要先移动 y 轴
'''



np.set_printoptions(precision=4, suppress=True)

'''#################################################################
以下是控制狗的位置的变量
'''
initial_pose = None # return to here finally
THR1 = 1.05 # if dog x < 0.8: turn; container
FLAG_COME_TURN = False # [0,0,0 ,1] to [-0.7071,0.,0.,0.7071], 到达阈值后将 flag_come_trun 设置为 1
dog_ready = False
FLAG_GO_TURN = False 
TURN_OVER = False
turn_rot = np.array([ # 绕 z 轴顺时针旋转 90 度
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
        ])
turn_quat = -0.7071,0.,0.,0.7071
'''##################################################################'''

# For pose detection
'''Begin of pose detection function'''
from src.utils import get_pc_from_rgbd, preprocess_pc_for_model
from src.model.est_pose import EstPoseNet
from src.model.est_coord import EstCoordNet
from src.config import Config
from transforms3d.quaternions import mat2quat, quat2mat
from src.utils import to_pose,rot_dist,get_pc,get_workspace_mask
import torch,cv2
from src.constants import DEPTH_IMG_SCALE
import traceback

COORD_MODEL_DIR = "./models/est_coord/checkpoint_8000.pth"
COORD_MODEL_DIR =""
POSE_MODEL_DIR = "./models/est_pose/checkpoint_10000.pth"
POSE_MODEL = None
COORD_MODEL = None
DEVICE = None
# For pose detection
import open3d as o3d
import numpy as np

def show_point_cloud(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])

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
        # pc_camera, colors = get_pc_from_rgbd(img, depth, camera_matrix,depth_scale=1.0) # meter
        # pc_camera = get_pc(depth,camera_matrix) 
       
        # # Transform to world coordinates for workspace filtering
        # pc_world = (camera_pose[:3, :3] @ pc_camera.T + camera_pose[:3, 3:]).T
        
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
        cv2.imshow("rgb", img)
        cv2.imwrite('rgb.png', img)
        full_pc_camera = get_pc(
            depth,camera_matrix
        ) * np.array([-1, -1, 1])
        # show_point_cloud(full_pc_camera)
        full_pc_world = (
            np.einsum("ab,nb->na", camera_pose[:3, :3], full_pc_camera)
            + camera_pose[:3, 3]
        )
        # full_coord = np.einsum(
        #     "ba,nb->na", obj_pose[:3, :3], full_pc_world - obj_pose[:3, 3]
        # )

        pc_mask = get_workspace_mask(full_pc_world)
        # pc_mask = np.ones(full_pc_world.shape[0], dtype=bool)
        
        sel_pc_idx = np.random.randint(0, np.sum(pc_mask), 1024)

        pc_camera = full_pc_camera[pc_mask][sel_pc_idx]
        pc_tensor = torch.from_numpy(pc_camera).float().unsqueeze(0).to(DEVICE)
        print(f"pc_tensor shape: {pc_tensor.shape}")
        show_point_cloud(pc_camera)
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
'''End of pose detection function'''
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
        # 变换后的新坐标系
        trans_marker_world =  np.dot(camera_pose, T_cam_tag)[:3,3]
        T_cam_tag = T_cam_tag @ A
        T_world_tag = np.dot(camera_pose, T_cam_tag) 
        
        rot_marker_world = T_world_tag[:3,:3]

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
    """
    规划抓取和提升轨迹
    """
    reach_steps = grasp_config["reach_steps"]
    lift_steps = grasp_config["lift_steps"]
    delta_dist = grasp_config["delta_dist"]

    pre_grasp_trans = grasp.trans - grasp.rot @ np.array([delta_dist, 0, 0])
    pre_grasp_rot = grasp.rot

    current_8d_qpos = env.get_state()
    
    success_reach_ik, reach_qpos_full_ik_solution = env.humanoid_robot_model.ik(
        trans=pre_grasp_trans,
        rot=pre_grasp_rot, 
        init_qpos=current_8d_qpos[:7],
        
    )

    if not success_reach_ik or reach_qpos_full_ik_solution is None: # 添加 None 检查
        print("警告: plan_grasp 无法找到到达姿态的 IK 解。")
        return None

    if reach_qpos_full_ik_solution.shape[0] >= 8:
        reach_qpos = reach_qpos_full_ik_solution[:8]
    else:
        print(f"警告: inverse_kinematics 返回的 QPOS 维数 ({reach_qpos_full_ik_solution.shape[0]}) 小于 8。手动添加夹爪关节 (默认 0.0)。")
        reach_qpos = np.concatenate([reach_qpos_full_ik_solution, np.array([0.0])])

    lift_trans = grasp.trans + grasp.rot @ np.array([delta_dist, 0, 0.2])
    lift_rot = grasp.rot

    
    success_lift_ik, lift_qpos_full_ik_solution = env.humanoid_robot_model.ik(
        trans=lift_trans,
        rot=lift_rot,
        init_qpos=current_8d_qpos[:7],
     
    )

    if not success_lift_ik or lift_qpos_full_ik_solution is None: # 添加 None 检查
        print("警告: plan_grasp 无法找到提升姿态的 IK 解。")
        return None

    if lift_qpos_full_ik_solution.shape[0] >= 8:
        lift_qpos = lift_qpos_full_ik_solution[:8]
    else:
        print(f"警告: inverse_kinematics 返回的 QPOS 维数 ({lift_qpos_full_ik_solution.shape[0]}) 小于 8。手动添加夹爪关节 (默认 0.0)。")
        lift_qpos = np.concatenate([lift_qpos_full_ik_solution, np.array([0.0])])
    
    print(f"DEBUG: plan_grasp calling plan_move_qpos with reach_steps={reach_steps}")
    reach_plan = plan_move_qpos(env, env.get_state(), reach_qpos, steps=reach_steps)

    print(f"DEBUG: plan_grasp calling plan_move_qpos with lift_steps={lift_steps}")
    lift_plan = plan_move_qpos(env, reach_qpos, lift_qpos, steps=lift_steps)

    return [reach_plan, lift_plan]


def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 300, *args, **kwargs):
    """
    规划从当前位姿到目标位姿的移动轨迹
    """
    current_8d_qpos = begin_qpos


    success_move_ik, target_qpos_full_ik_solution = env.humanoid_robot_model.ik(
        trans=end_trans,
        rot=end_rot, 
        init_qpos=current_8d_qpos[:7],
    
    )
    if not success_move_ik or target_qpos_full_ik_solution is None: # 添加 None 检查
        print("警告: plan_move 无法计算 IK 目标 QPOS。")
        return None

    if target_qpos_full_ik_solution.shape[0] >= 8:
        target_qpos_8d = target_qpos_full_ik_solution[:8]
    else:
        print(f"警告: plan_move 的 inverse_kinematics 返回的 QPOS 维数 ({target_qpos_full_ik_solution.shape[0]}) 小于 8。手动添加夹爪关节 (默认 0.0)。")
        target_qpos_8d = np.concatenate([target_qpos_full_ik_solution, np.array([0.0])])

    delta_qpos = (target_qpos_8d - current_8d_qpos) / steps
    traj = []
    current_qpos_iter = current_8d_qpos.copy()
    for _ in range(steps):
        current_qpos_iter += delta_qpos
        traj.append(current_qpos_iter.copy())
    
    return np.array(traj)

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)

def plan_move_qpos(env: WrapperEnv, begin_qpos, end_qpos, steps=200) -> np.ndarray:
    

    if begin_qpos.shape != end_qpos.shape:
        print(f"警告: plan_move_qpos 输入 QPOS 形状不匹配：begin_qpos 形状 {begin_qpos.shape} 与 end_qpos 形状 {end_qpos.shape}。")
        raise ValueError(f"QPOS 形状不匹配，无法在 plan_move_qpos 中进行操作：{begin_qpos.shape} vs {end_qpos.shape}")
    
    if steps <= 0: 
        print(f"警告: plan_move_qpos 接收到非正的步数 (steps={steps})。返回空轨迹。")
        return np.array([]) 
    
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    return np.array(traj)

def execute_plan(env: WrapperEnv, plan):
    for step in range(len(plan)):
        if plan[step].shape[0] != 8:
            print(f"警告: execute_plan 接收到非 8 维的计划步骤 ({plan[step].shape})。只尝试使用前 7 维。")     
        env.step_env(
            humanoid_action=plan[step][:7], 
        )

def head_move_policy(tag_world,current_head_pose):
    current_head_pose[1]+=0.3
    return current_head_pose

TESTING = True
DISABLE_GRASP = True
DISABLE_MOVE = False

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0) # 暂时不显式
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
    
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0,0,0,0,0]) # you can customize observing qpos to get wrist obs
    # init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos)
    # execute_plan(env, init_plan)

    '''############################################################################################################
    Priviledge infomation in the environment: You can't use them directly, only used to debug and evaluation
    '''
    container_dog = np.array([0.09, 0, 0.115, 0, 0, 0, 1])
    # tag_pose_dog = np.array([-0.22, 0, 0.09, 0.7071, 0.7071, 0, 0])
    tag_pose_dog = np.array([-0.22, 0, 0.09, 0, 0, 0, 1])
    dog_world = env.sim.mj_data.qpos[:7]
    
    real_tag_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=tag_pose_dog)
    # real_tag_pose = algo.pose7d_to_T(real_tag_pose)
    real_container_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=container_dog)
    # real_container_pose = algo.pose7d_to_T(real_container_pose)
    print("Dog pose in the world is : ",dog_world)
    '''################################################################################################################'''



    obs_head = env.get_obs(camera_id=0) # head camera
    obs_wrist = env.get_obs(camera_id=1) # wrist camera

    env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
    env.debug_save_obs(obs_wrist, 'data/obs_wrist')

    pose_container_world = None

    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        global dog_ready,FLAG_COME_TURN,TURN_OVER,turn_rot,THR1

        align_y = False
        count_y = 0
        y_setting = -0.20 
        head_qpos = head_init_qpos
        forward_steps = 600 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        target_container_pose = np.array([0.4, -0.085,   0.2745, -0.7071,  0.,  0.,  0.7071]) 

        for step in range(forward_steps):
            move_head = False
            if dog_ready: 
                print(f"After {step} steps, The dog is ready, with container xyz {pose_container_world[:3,3]}")
                break # OK

            dog_pose = env.sim.mj_data.qpos[:7]
            real_container_pose = algo.from_dog_frame_to_world(dog_pose=dog_pose,obj_local=container_dog)
            real_tag_pose = algo.from_dog_frame_to_world(dog_pose=dog_pose,obj_local=tag_pose_dog)

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
                    trans_container_world = rot_marker_world @ np.array([0.31,0,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)


                    '''暂时设置为真实情况'''
                    # pose_container_world = algo.pose7d_to_T(real_container_pose)
                    '''Check if detect mark in the world is true'''
                    tag_world = to_pose(trans_marker_world,rot_marker_world)
                    if step < 30 and step%5==0:
                        print(f"   pred Tag in the world {algo.T_to_pose7d(tag_world)}, groudtruth Tag in the world: {real_tag_pose}")
                        print(f"   pred Container in the world {algo.T_to_pose7d(pose_container_world)}, groudtruth container in the world: {real_container_pose}")
                else: # No AprilTag detected: Try to change human head pose
                    print("No AprilTag detected: Try to change human head pose")
                    move_head = True

            
            '''DEBUG'''
            # dog_rot = R.from_quat(env.sim.mj_data.qpos[3:7]).as_matrix()
            # if rot_dist(dog_rot,turn_rot)<2:
            #     print(rot_dist(dog_rot,turn_rot))
            
            '''DEBUG'''

            '''Set y to a goal: -0.2; align for different initial state'''

            # dog_begin_y = tag_world[1,3]
            # y_dist = np.sqrt(dog_begin_y-y_setting)

            # if y_dist > 0.2: # move along y axis
            #     count_y +=1
            #     align_y = True
            #     # change y 
            #     if dog_begin_y> y_setting:
            #         quad_command = [0,-y_dist,0]
            #     else:
            #         quad_command = [0,y_dist,0]
            #     env.step_env(
            #         humanoid_head_qpos=head_qpos,
            #         quad_command=quad_command
            #     )
            #     continue
            if align_y:
                print(f"Use {count_y} steps to align y to {y_setting}!")

            if move_head:
                ''' Heap qpos
                default: np.array([-0.05, 0.35])
                horizontal:[-1.57,1.57], negative -> turn right
                vertical: [-0.366,0.366], negative -> up
                '''
                '''也许需要动态的调整'''
                if FLAG_COME_TURN and head_qpos[1] > 0.36:
                    head_qpos[0]-=0.1
                head_qpos[1] = min(head_qpos[1]+0.1,0.366)
                quad_command = [0,0,0]
                if head_qpos[1]>=0.366:
                    print("turn the head to most down, But no Tag detected!")
                else:
                    print(f"Move head to {head_qpos}!")
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
                continue
            if not FLAG_COME_TURN: # come colser
                if pose_container_world[0,3]<THR1: # x near
                    FLAG_COME_TURN = True
                else:
                    quad_command = [0.5,0,0] # move forward
            elif FLAG_COME_TURN and not TURN_OVER: # turn 90 degree
                if rot_dist(turn_rot,pose_container_world[:3,:3]) > 0.15:
                    # print(rot_dist(turn_rot,pose_container_world[:3,:3]))
                    quad_command = [0,0,0.5]
                else:
                    TURN_OVER = True
                    quad_command = [0,0,0]
                    print("Turn over!")
            elif TURN_OVER:  # 侧身走过来
                if pose_container_world[0,3] < 0.65:
                    dog_ready = True
                    quad_command = [0,0,0]
                else:
                    print(pose_container_world[0,3])
                    quad_command = [0,-0.1,0]
            # is_close(pose_container_world, target_container_pose, threshold=0.05)
            env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose[:3, 3])
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
      
        obj_pose = driller_pose.copy()
        obj_pose = env.get_driller_pose()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=300,
            lift_steps=300,
            delta_dist=0.01, 
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

        print(f"DEBUG: Shape of observing_qpos before plan_move_qpos call: {observing_qpos.shape}")
        print(f"DEBUG: Content of observing_qpos before plan_move_qpos call: {observing_qpos}")
    
        if len(reach_plan) == 0:
            print("警告: reach_plan 为空，无法继续执行。")
            env.close()
            return


        print(f"DEBUG: Shape of reach_plan[0] before plan_move_qpos call: {reach_plan[0].shape}")
        print(f"DEBUG: Content of reach_plan[0] before plan_move_qpos call: {reach_plan[0]}")
        pregrasp_plan = plan_move_qpos(env, observing_qpos, reach_plan[0], steps=100) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        print("Pregrasp plan executed.")
        open_gripper(env)

        execute_plan(env, reach_plan)
        print("Reach plan executed.")
        close_gripper(env)
        execute_plan(env, lift_plan)
        print("Lift plan executed.")


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        current_gripper_trans, current_gripper_rot = env.humanoid_robot_model.fk_link(env.get_state()[:7], env.humanoid_robot_cfg.link_eef) # 正向运动学获取末端执行器位姿
    
        # 假设 target_container_pose 在 step1 中已计算并可用
        # fixed_container_trans 和 fixed_container_rot 应该在此处定义
        # 例如：
        fixed_container_trans = np.array([0.6, -0.06, 0.6])
        fixed_container_rot = np.eye(3)
        
        target_container_pose = to_pose(fixed_container_trans, fixed_container_rot)
        
     
        target_container_pose = np.asarray(target_container_pose)    
        trans_container_world = target_container_pose[:3, 3]
        rot_container_world = target_container_pose[:3, :3]
     
        drop_trans = trans_container_world + np.array([0, 0, 0.1]) # 调整上方位置
        drop_rot = rot_container_world 
        
    
        move_plan = plan_move(
            env=env,
            begin_qpos=env.get_state(), 
            begin_trans=current_gripper_trans,
            begin_rot=current_gripper_rot,
            end_trans=drop_trans,
            end_rot=drop_rot,
            steps=300
        ) 
        
        if move_plan is None:
            print("No valid move plan found for dropping.")
            env.close()
            return

        execute_plan(env, move_plan) 
        open_gripper(env) 

    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        # if flag_go_turn is
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