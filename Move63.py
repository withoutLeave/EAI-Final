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
from scipy.spatial.transform import Rotation as R, Slerp
from src.robot.cfg import get_robot_cfg

'''
临死前面临的问题及可能的补救措施：

0. 训练出一个误差较小的位姿估计网络。

1. AprilTag Detection，不能依照 groundtruth 调整，因为误差较大，而且误差不是恒定的。（不知道其他组有没有遇到），也有可能是我们的代码中的 BUG。如果仍然无法解决需要对 狗 进行更加复杂的控制。

2. 目前 grasp 的成功率相对较高，但是仍然有不成功的时候，可以在最近的一个轨迹之间做插值，强制不要绕大圈。（由于 initial qpos是确定的，所以估计插值 1~2 个值），这个解决的策略可能相对容易。

3. move and drop 做的仍然不好,目前最优的方案是根据确定的 initial pose, goal drop pose 来确定一个固定的路径。这样的话，需要解出来一个较好的目的地 IK 的 qpos，
所谓较好，指与初始位置距离较近。目前考虑的是：仍然利用现有的解 IK 的方式，解出来一个末端最终位置对应的 qpos，然后人工插值 DEBUG，确保找到一条合理的路径。之后可以根据情况自动差值。
实现时参考 GXC 同学的实现方案.
'''



np.set_printoptions(precision=4, suppress=True)

'''#################################################################
以下是控制狗的位置的变量
'''
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
quad_move_traj = [] # store all the quad_command to reverse and roll out
'''##################################################################
以下是控制机器人的变量，目前考虑从随机的初始位置到 “一个固定的位置”，然后抓取，回到这个位置，从这个位置前往放置的位置
'''
general_pose = 0






'''###################################################################'''

def detect_driller_pose(img, depth, camera_matrix, camera_pose, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    # 
    
    pose = np.eye(4)
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

def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    action = np.array([0,0,0])
    return action

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
    for i in range(grasp_config["reach_steps"]):
        cur_trans = cur_trans - grasp_config["delta_dist"] * cur_rot[:, 0]# 将目标靠近现有位置
        succ, cur_qpos = robot_model.ik(
            cur_trans, cur_rot, cur_qpos, delta_thresh=0.5
        )
        if not succ: return None
        traj = [cur_qpos] + traj    
    return traj

def plan_move(env: WrapperEnv,
              begin_qpos,       # 当前 7 维关节角
              begin_pose,       # 4×4 齐次矩阵，表示 begin_qpos 对应的末端位姿
              end_pose,         # 4×4 齐次矩阵，期望到达的目标末端位姿
             ):
    
    '''所以 为了保证成功率，我们确定好某个点，这个点一定能到释放位置'''
    end_trans = end_pose[:3, 3]
    # end_trans[0]=0.68
    # end_trans[1] = -0.05
    # end_trans[2] = 0. # 暂时这样设置 z
    success_move_ik = False
    target_qpos = None

    # '''方式1：使用严格的 R，原始的 IK，但是不一定能解出来'''
    # way1_rot = end_pose[:3, :3]
    # succ, qpose = env.humanoid_robot_model.ik(trans= end_trans,rot= way1_rot,init_qpos = begin_qpos)
    # if succ:
    #     target_qpos = qpose
    #     success_move_ik = True
    #     target_xyz = end_trans

    '''方式2：更复杂的枚举，同时利用自己定义的 IK'''
    r_base = R.from_matrix(begin_pose[:3, :3])
    # 2. 在不同的绕局部 z 轴旋转 + x/y 微扰动的姿态中尝试 IK
    N = 50
    rot_matrices = []
    for i in range(N):
        # 绕 base 的局部 z 轴做 i*(360/N) 度旋转
        angle = 2 * np.pi * i / N
        r_z = R.from_rotvec([0, 0, angle])

        # 再给 x/y 加一点随机扰动（这里示范 ±20°）
        delta = np.deg2rad(np.random.uniform(-5, 5, size=2))
        r_xy = R.from_rotvec([delta[0], delta[1], 0])

        r_new = r_base * r_z * r_xy
        rot_matrices.append(r_new.as_matrix())

    for height in np.arange(9.0, 0.6, -0.1): # 
        end_trans[2]=height
        for rot in rot_matrices:
            success, qpos_ik = ik_4move(
                robot=env.humanoid_robot_model,
                trans=end_trans,
                rot=rot,
                init_qpos=begin_qpos,
                xy_tol = 0.01,
                z_tol= 0.02,
                rot_tol=0.5,
                tol=1e-3,
                delta_thresh=0.5
            )
            if success:
                target_qpos = qpos_ik
                success_move_ik = True
                
                break
    if not success_move_ik:
        print(f"Failed, can't solve ik to xyz {end_trans}.")
        return None
    else:
        return target_qpos


def open_gripper(env: WrapperEnv, steps = 30):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 30):
    for _ in range(steps):
        env.step_env(gripper_open=0)

def ik_4move(
        robot,
        trans: np.ndarray,
        rot: np.ndarray,
        init_qpos: Optional[np.ndarray] = None,
        retry_times: int = 10,
        xy_tol=1e-2,
        z_tol=0.1,
        tol = 1e-5,
        rot_tol=1e-1,
        delta_thresh: float = None,
    ) -> Tuple[bool, np.ndarray]:
        '''自己定义的 IK，为释放 driller 定制了更宽松的约束'''
        pose = to_pose(trans, rot)
        for _ in range(retry_times):
            ik_result, success, _, _, _ = robot.robot.ik_lm_chan(
                pose, end=robot.cfg.link_eef, q0=init_qpos,tol=tol
            )
            if success:
                if delta_thresh is not None:
                    if np.linalg.norm(ik_result - init_qpos) > delta_thresh:
                        continue
                t, r = robot.fk_link(ik_result, robot.cfg.link_eef)
                if np.linalg.norm(t[:2] - trans[:2]) < xy_tol and np.linalg.norm(t[2] - trans[2]) < z_tol and rot_dist(r, rot) < rot_tol:
                    return True, ik_result
        return False, ik_result

def execute_plan(env, gra_plan: np.ndarray,obj_pose,plan_type:int) -> bool:
    """Execute the planned trajectory in the simulation and check if the grasp was successful."""
    if plan_type == 1: 
        '''特定针对抓取的执行方式，包括返回抓取的起点'''
        open_gripper(env)
        succ_height_thresh = 0.1
        obj_init_z = obj_pose[2,3]
        # grasp
        initial_ok = False
        dist = 0
        for _ in range(200):
            env.step_env(
                humanoid_action=gra_plan[0][:7], 
            )
            p_xyz=env.humanoid_robot_model.fk_link(gra_plan[0],env.humanoid_robot_cfg.link_eef)[0]
            c_xyz = env.humanoid_robot_model.fk_link(env.sim.mj_data.qpos[env.sim.qpos_humanoid_begin:env.sim.qpos_humanoid_begin+7],env.humanoid_robot_cfg.link_eef)[0]
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

        obj_final_z = env.get_driller_pose()[2,3]
        if obj_final_z - obj_init_z > succ_height_thresh:
            return True
        else:
            print(f"Execute false, because the height is smaller than {succ_height_thresh} m!")
            return False
    elif plan_type == 2:# move and drop, grasp_plan is plan for move, lift_plan is None
        '''使用较少的步数'''
        for qpos in gra_plan: 
            for _ in range(8):
                env.step_env(
                    humanoid_action=qpos[:7], 

                )
            current_gripper_trans, current_gripper_rot = env.humanoid_robot_model.fk_link(qpos, env.humanoid_robot_cfg.link_eef) # 正向运动学获取末端执行器位姿

            print("Current end effector is :",current_gripper_trans)
        open_gripper(env)
    elif plan_type ==3: # here gra_plan is [,7]
        '''特定针对只有没有中间过程的规划'''
        for i in range(200):
            env.step_env(
                    humanoid_action=gra_plan, 
                )

def safe_drop(
    env,
    begin_pose: np.ndarray,
    drop_xyz: np.ndarray,
    safe_z: float = 1.0,
    seg_steps: int = 10
) -> Optional[np.ndarray]:
    """
    三段式避障轨迹：
      1) 把末端从 begin_pose 的 z 抬升到 safe_z（保持 x,y 不变）
      2) 在 safe_z 高度下，将 x,y 从 (begin_x, begin_y) 移到 (drop_x, drop_y)
      3) 从 (drop_x, drop_y, safe_z) 下降到 (drop_x, drop_y, drop_z)

    参数：
      env         - WrapperEnv 对象
      begin_pose  - 当前末端的 4×4 位姿矩阵
      drop_xyz    - 目标投放位置的 [x, y, z]（z≈0.2732）
      safe_z      - 在桌面之上的安全高度（必须 > 桌面高度，一般取 0.9~1.0m）
      seg_steps   - 每一段插值用的关键帧数

    返回：
      若成功，返回形状 (3*seg_steps, 7) 的关节角轨迹数组；  
      若任一段 IK 失败，则返回 None。
    """
    def _make_pose(xyz, rot):
        # 将 xyz (3,) 和 rot (3×3) 拼成一个 4×4 的齐次矩阵
        return to_pose(np.array(xyz), rot)

    # 1. 从 begin_pose 提取当前 x,y,z 和当前朝向（3×3）
    curr_x, curr_y, curr_z = begin_pose[0,3], begin_pose[1,3], begin_pose[2,3]
    curr_rot = begin_pose[:3,:3]

    # 2. 构造 via1：抬到 safe_z（x,y 不变，转角保持不变）
    via1 = _make_pose([curr_x, curr_y, safe_z], curr_rot)

    # 3. 构造 via2：先水平移动到 drop_x,drop_y，同时保持 safe_z（旋转矩阵保持不变）
    drop_x, drop_y, drop_z = drop_xyz
    via2 = _make_pose([drop_x, drop_y, safe_z], curr_rot)

    # 4. 构造 target：最终下降到 drop_z
    target = _make_pose([drop_x, drop_y, drop_z], curr_rot)

    traj_segments = []
    q0 = env.get_state()[:7]  # 当前关节角，作为第一段 IK 的初始值

    # 三段依次做 IK 规划
    for (start_pose, end_pose) in [(begin_pose, via1), (via1, via2), (via2, target)]:
        seg = plan_move(
            env,
            begin_qpos=q0,
            begin_pose=start_pose,
            end_pose=end_pose,
            move_steps=seg_steps
        )
        if seg is None:
            # 任一段 IK 失败，就直接报错返回 None
            print("safe_drop: IK 失败，无法从\n", start_pose, "\n移动到\n", end_pose)
            return None
        traj_segments.append(seg)
        # 下一段的起始 q0 应该等于当前段最后一个关节角
        q0 = seg[-1].copy()

    # 将三段关节轨迹拼接成一个 (3*seg_steps, 7) 的大数组
    return np.vstack(traj_segments)


TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = True # if false, Dog can come near to the robot
DISABLE_RETURN = True
def main():

    global quad_move_traj,DISABLE_MOVE # used to record all quad commands

    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0) # 暂时不显式
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)
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
    
    head_init_qpos = np.array([0.,0.3]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0.15,0,0,0,0.15]) # you can customize observing qpos to get wrist obs
    init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos) # plan move to observing position
    execute_plan(env, init_plan) # execute the planned movement

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

    '''以下是过程中两个非常重要的变量'''
    pose_container_world = None # Used when move and drop object
    quad_initial_pose = None # return to here finally
    get_ini_quad_pose = False
    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        print("*"*30,"\nStage 2: move dog near the robot.")
        global dog_ready,FLAG_COME_TURN,TURN_OVER,turn_rot,THR1


        def fine_get_closer():
            for i in range(45):
                quad_command=[-0.2,0,0]
                quad_move_traj.append(quad_command)
                env.step_env(
                        quad_command=quad_command
                    )
            real_container_pose = algo.from_dog_frame_to_world(dog_pose=env.sim.mj_data.qpos[:7],obj_local=container_dog)
            # print(container_dog)
            # print(f"IIIIIImportant （这里的 x 可能有问题，我觉得 [0.68,-0.05,z] 应该就可以）    Now dog pos is: {env.sim.mj_data.qpos[:7]},container xyz is {real_container_pose}." )
            # print(container_dog)
            env.step_env(
                            quad_command=[0,0,0]
                        )     
        align_y = False
        count_y = 0
        y_setting = -0.20  # 规范化 y 到 -0.2
        head_qpos = head_init_qpos
        forward_steps = 700 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        # target_container_pose = np.array([0.4, -0.085,   0.2745, -0.7071,  0.,  0.,  0.7071]) 

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
                    if not get_ini_quad_pose: # haven't get quad init pose, only save once
                        get_ini_quad_pose = True
                        quad_initial_pose = to_pose(trans_marker_world,rot_marker_world)
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0.31,0.,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    '''暂时设置为真实情况'''
                    # pose_container_world = algo.pose7d_to_T(real_container_pose)
                    '''Check if detect mark in the world is true'''
                    # tag_world = to_pose(trans_marker_world,rot_marker_world)
                    # if step%20==0:
                        # print(f"   pred Tag in the world {algo.T_to_pose7d(tag_world)} , groudtruth Tag in the world: {real_tag_pose}")
                        # print(f"   pred Container in the world {algo.T_to_pose7d(pose_container_world)}, groudtruth container in the world: {real_container_pose}")
                else: # No AprilTag detected: Try to change human head pose
                    move_head = True            
            '''DEBUG'''
            # dog_rot = R.from_quat(env.sim.mj_data.qpos[3:7]).as_matrix()
            # if rot_dist(dog_rot,turn_rot)<2:
            #     print(rot_dist(dog_rot,turn_rot))
            '''DEBUG'''
            '''Set y to a goal: -0.2; align for different initial state'''
            '''这里有一些问题，需要调整'''
            # dog_begin_y = trans_marker_world[2] # dog's y 
            # print(dog_begin_y)
            # y_dist = np.linalg.norm(dog_begin_y-y_setting)

            # if y_dist > 0.08: # move along y axis
            #     count_y +=1
            #     # change y 
            #     if not align_y:
            #         align_y = True
            #         print(f"The dog's initial y need to be aligned from {dog_begin_y} to {y_setting}!")
            #     if dog_begin_y > y_setting:
            #         quad_command = [0,-y_dist,0]
            #     else:
            #         quad_command = [0,y_dist,0]
            #     quad_move_traj.append(quad_command)
            #     env.step_env(
            #         humanoid_head_qpos=head_qpos,
            #         quad_command=quad_command
            #     )
            #     continue

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
                if pose_container_world[0,3] < 0.75:
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
        fine_get_closer()
        assert(pose_container_world is not None)


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP and False:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose[:3, 3])
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)

    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:

        print("*"*30,"\nStage 3: grasp and lift task begin")
        # 预先设置：
        # obj_pose = driller_pose.copy()
        obj_pose = env.get_driller_pose()
        print(f"Object pose is {algo.T_to_pose7d(obj_pose)}!")
        grasps = get_grasps(args.obj)  # 得到了 8 个 grasp

        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n]
        # grasps[0] 应该是可以成功的
        est_trans = obj_pose[:3,3]
        est_rot = obj_pose[:3,:3]

        grasp_config = dict( 
            reach_steps=20,
            delta_dist=0.01
        ) 
        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans = est_rot @ obj_frame_grasp.trans + est_trans,
                rot   = est_rot @ obj_frame_grasp.rot,
                width = obj_frame_grasp.width,
            )
            for _ in range(args.try_plan_num):
                gra_plan = plan_grasp(env=env,grasp=robot_frame_grasp,grasp_config=grasp_config)
                if gra_plan is not None:
                    break
            if gra_plan is not None:
                break
        
        if gra_plan is not None:
            # print(f"choose grasp in the world is {choosed_grasp.trans}.")
            succ = execute_plan(env,gra_plan=gra_plan,obj_pose=obj_pose,plan_type= 1)
            print(f"Grasp driller from the desktop and lift task: {'succeeded' if succ else 'failed'}\n","*"*30)
            if succ:
                DISABLE_MOVE = False
        else:
            succ = False
            env.close()
            print("No plan found")
            return 
    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    
    if not DISABLE_GRASP and not DISABLE_MOVE:
        print("Stage 4: Move the object to a specific pose!")
        # implement your moving plan

        '''#################################
        尝试 move 到不同的位置,z在找一个可行的最终的 qpos 时可能需要用到这些内容
        使用 FK ， IK 和可视化验证得到的解是一个可行的解
        '''
        # be_qpos = humanoid_init_qpos
        # ini_pose = env.humanoid_robot_model.fk_link(humanoid_init_qpos,env.humanoid_robot_cfg.link_eef)
        # be_pose = to_pose(ini_pose[0],ini_pose[1])
        # # print("original ini pose of end effector",quat)    
        # mid_pose = to_pose([0.68,0.00,0.9],ini_pose[1])
        # move_p = plan_move(env,begin_qpos=be_qpos,begin_pose=be_pose,end_pose=mid_pose)
        # if move_p is not None:
        #     print("get traj")
        #     execute_plan(env,gra_plan=move_p,obj_pose=None,plan_type=3)
        #     print("成功解出测试的最终 pose. 最终的 qpos 是：",move_p)
        # else: 
        #     print("暂时无法解出测试的最终 pose.")

        # '''测试和打印 FK'''
        # test_qpos = [ 0.8583, -1.674 ,  1.9545,  0.5242 , 2.6412, -0.469,   0.7482]
        # test_pose = env.humanoid_robot_model.fk_link(test_qpos,env.humanoid_robot_cfg.link_eef)
        # print(f"Test pose after FK is {test_pose}")
        '''#################################'''


        # env.step_env(
        #     humanoid_action=humanoid_init_qpos, # reset humanoid arm to initial position
        # )
        current_gripper_trans, current_gripper_rot = env.humanoid_robot_model.fk_link(env.sim.mj_data.qpos[env.sim.qpos_humanoid_begin:env.sim.qpos_humanoid_begin+7], env.humanoid_robot_cfg.link_eef) # 正向运动学获取末端执行器位姿
        
        #drop_trans = pose_container_world[:3,3]
        

        '''Debug for container pose''' 

        drop_rot = current_gripper_rot
        current_gripper_pose = to_pose(current_gripper_trans,current_gripper_rot)
        #gripper_open_pose = to_pose(drop_trans, drop_rot) # 在世界坐标系下的位置
        gripper_open_pose = np.eye(4) # 末端执行器打开的位姿
        gripper_open_pose[:3,3] =  np.array([0.68,-0.05 ,None])# 这里的 z 是桌面高度
         
        print(f"Current xyz is {current_gripper_trans}.")
   
        print(f"Gripper open pose is {gripper_open_pose[:3,3]}.")

        '''##############################################################
        原本的实现方式，有的时候找不到 IK 的解
        '''
        # move_plan = plan_move(
        #     env=env,
        #     begin_qpos = env.get_state()[:7],
        #     begin_pose=current_gripper_pose, 
        #     end_pose=gripper_open_pose,
        # ) 
        # if move_plan is None:
        #     print("No valid move plan found for dropping.")
        #     env.close()
        #     return
        # print(move_plan)
        '''##############################################################'''

        # move_plan=np.array([ 0.9371,-1.5984 ,-1.4545, -0.5024, -0.0948, -0.2711,  0.8]) # 这是一个可行的 qpos，但是路径最后夹爪的角度不合适
        move_traj = np.array([
            [-0.7541, -1.5842,  0.1683, -1.5229, -0.9842, -0.2657, -0.5892],

            [-0.397,  -1.596,   0.0547, -1.4073, -0.1278, -0.271,   0.2212],
            [-0.1843, -1.5982, -0.023,  -1.3433, -0.0929, -0.2674,  0.7111],

            [ 0.1823, -1.5986, -0.1519, -1.2273, -0.0942, -0.2694,  0.7949],
            [ 0.2899, -1.5984, -0.187,  -1.1901, -0.0945, -0.2697,  0.7947],
            # [ 0.3879, -1.5982, -0.2192, -1.155,  -0.0948, -0.27,    0.7945],
            # [ 0.5085, -1.598,  -0.2589, -1.1099, -0.0951, -0.2703,  0.7944],
            # [ 0.6304, -1.5978, -0.2998, -1.0621, -0.0954, -0.2706,  0.7943],
            # [ 0.7609, -1.5967, -0.8024, -0.9955, -0.0966, -0.2718,  0.7937],
            # [ 0.8718, -1.5983, -0.3981, -0.9526, -0.0935, -0.2677,  0.7917],
            # [ 0.9371, -1.5984, -1.4545, -0.5024, -0.0948, -0.2711,  0.8]
        ])


        def interpolate_joint_trajectory(traj, num_interp=5):
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
 
   
    DISABLE_RETURN = True # let the dog return 
    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_RETURN:
        # implement
        '''把轨迹倒放一遍，之后结合视觉微调'''
        backward_steps = 1000 # customize by yourselves
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

        reset_head_qpos = [0,0]
        print(f"After roll out, fine tune the final position to the task initial pose. First change head pose to {reset_head_qpos}.")
        env.step_env(
                quad_command=[0,0,0],
                humanoid_head_qpos= reset_head_qpos
            )

        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])
        obs_head = env.get_obs(camera_id=0)
        trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
        current_tag_pose = to_pose(trans_marker_world, rot_marker_world)

        print(f"After roll out, fine tune the final position to the task initial pose. Current xyz is {current_tag_pose[:3,3]}, goal xyz is {quad_initial_pose[:3,3]}.")
        # step +=  num_come_steps+1
        # for step in range(backward_steps):
        #     # same as before, please implement this
        #     #  TODO:
        #     # 先矫正方向，然后严格拟合 x 与 y
        #     quad_command = backward_quad_policy()
        #     env.step_env(
        #         quad_command=quad_command
        #     )
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()