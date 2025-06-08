import asyncio
import websockets
import json
import time
import cv2
import base64
from src.real.client.realsense_camera_ros import RealSenseCameraROS as rscr
from galbot_control_interface import GalbotControlInterface

import threading
from queue import Queue
import signal
import tf2_ros
import rospy
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv("/home/galbot_orin_2024/Projects/Assignment4/hw4.env")

# Initialize once at startup
gci = GalbotControlInterface()

# Constants for camera types
CAMERA_TYPES = {
    "head": {
        "rgb_topic": "/cam/head_f/color/image_raw",
        "depth_topic": "/cam/head_f/aligned_depth_to_color/image_raw",
        "info_topic": "/cam/head_f/color/camera_info",
        "camera_frame": "front_head_camera_color_optical_frame",
    },
    "wrist": {
        "rgb_topic": "/cam/left_arm/wrist/color/image_raw",
        "depth_topic": "/cam/left_arm/wrist/aligned_depth_to_color/image_raw",
        "info_topic": "/cam/left_arm/wrist/color/camera_info",
        "camera_frame": "left_arm_camera_color_optical_frame",
    },
}

# Base frame constant
BASE_FRAME = "base_link"

tf_buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tf_buffer)
def get_camera_pose(base_frame: str, camera_frame: str):
    """Get the camera pose from tf tree.
    Args:
        base_frame (str): the target frame, e.g., robot_base_link
        camera_frame (str): the source frame, e.g., left_arm_camera_link
    Returns:
        np.ndarray: camera pose as [x, y, z, qx, qy, qz, qw] or None if not found
    """
    timeout = rospy.Duration(5)
    start_time = time.time()
    while not rospy.is_shutdown():
        try:
            tf_res = tf_buffer.lookup_transform(
                base_frame, camera_frame, rospy.Time(0), timeout
            )
            trans = tf_res.transform.translation
            rot = tf_res.transform.rotation
            camera_pose = np.array([trans.x, trans.y, trans.z, rot.x, rot.y, rot.z, rot.w])
            return camera_pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            if time.time() - start_time > 5:
                rospy.logwarn(f"TF lookup timed out after {5} seconds: {e}")
                return None
            rospy.logwarn(f"TF lookup failed: {e}, retrying...")
            rospy.sleep(0.1)

class HumanoidClient:
    def __init__(self, address):
        self.address = address
        self.message_queue = Queue()
        self.running = True
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def start_receive_thread(self):
        def run_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect_and_receive())

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        return thread
    
    async def connect_and_receive(self):
        while self.running:
            try:
                async with websockets.connect(self.address, max_size=10*1024*1024) as websocket:
                    await self.receive_messages(websocket)
            except (websockets.ConnectionClosed, OSError, Exception) as e:
                delay = 1
                print(f"{delay:.1f} seconds before reconnecting") 
                await asyncio.sleep(delay)
        # async with websockets.connect(self.address, max_size=10*1024*1024) as websocket:
        #     print("Connected to WebSocket server")
        #     receive_task = asyncio.create_task(self.receive_messages(websocket))
        #     while self.running:
        #         data = self._humanoid_data()
        #         await websocket.send(json.dumps(data))
        #         # if len(data) > 1000:
        #         #     print(f"Sent: {data[:1000]}")
        #         # else:
        #         #     print(f"Sent: {data}")
        #         await asyncio.sleep(0.1) 

        #     receive_task.cancel()
        #     try:
        #         await receive_task
        #     except asyncio.CancelledError:
        #         print("Receive task cancelled")

    async def receive_messages(self, websocket):
        while self.running:
            try:
                message = await websocket.recv()  # 接收消息
                if len(message) > 1000:
                    print(f"Received: {message[:1000]}")
                else:
                    print(f"Received: {message}")
                
                msg = json.loads(message)
                await self._execute_control(websocket, msg)
                if 'type' in msg:
                    done = {
                        "type": msg['type']+'.done',
                        "status": "success",
                    }
                    await websocket.send(json.dumps(done))
                    print(f"Sent done: {done}")
            except websockets.ConnectionClosed:
                print("Connection closed by server.")
                # self.running = False
                break

    # async def connect(self, uri):
    #     async with websockets.connect(uri, max_size=10*1024*1024) as ws:
    #         while True:
    #             # send sensor data
    #             data = self._humanoid_data()
    #             await ws.send(json.dumps(data))

    #             # # receive control command 
    #             response = await ws.recv()
    #             self._execute_control(json.loads(response))

    def _humanoid_data(self):
        """Generate mock sensor data"""
        return {
            "type": "humanoid_data",
            "timestamp": time.time(),
            "head_camera": self._get_camera_data("head"),
            "wrist_camera": self._get_camera_data("wrist"),
            "joint_angles": self._get_state(),
        }
        
    def _encode_image(self, img, is_depth=False):
        if img is None:
            return None
        if is_depth:
            # Depth: uint16 single channel
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            _, buffer = cv2.imencode(".png", img, params)
        else:
            # RGB: uint8 3 channel
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            _, buffer = cv2.imencode(".jpg", img, params)
        return base64.b64encode(buffer).decode("utf-8")

    def _get_camera_data(self, part):
        """Capture camera data"""

        if part not in CAMERA_TYPES:
            raise ValueError(f"Invalid camera part '{part}'. Choose 'head' or 'wrist'.")

        config = CAMERA_TYPES[part]
        try:
            color = rscr.get_rgb_image(config["rgb_topic"])
            depth = rscr.get_depth_image(config["depth_topic"])
            intrinsic = rscr.get_camera_info(config["info_topic"])
            pose_data = get_camera_pose(BASE_FRAME, config["camera_frame"])

            # pose = {
            #     "position": list(pose_data["position"]) if pose_data is not None else [0, 0, 0],
            #     "orientation": list(pose_data["orientation"]) if pose_data is not None else [0, 0, 0, 1],
            # }
            
            pose = {
                "position": list(pose_data[:3]) if pose_data is not None else [0, 0, 0],
                "orientation": list(pose_data[3:]) if pose_data is not None else [0, 0, 0, 1],
            }

            return {
                "color": self._encode_image(color),
                "depth": self._encode_image(depth, is_depth=True),
                "intrinsic": intrinsic,
                "pose": pose,
            }
        except Exception as e:
            print(f"Error capturing {part} camera data: {str(e)}")
            return None
        
    def _get_state(self):
        """Get the current state of the robot."""
        try:
            leg_joint_angles = gci.get_leg_joint_angles()
            head_joint_angles = gci.get_head_joint_angles()
            left_arm_joint_angles = gci.get_arm_joint_angles("left_arm")
            left_gripper_width = gci.get_gripper_status("left_gripper")["width"]
            return {
                "leg_joint_angles": leg_joint_angles,
                "head_joint_angles": head_joint_angles,
                "left_arm_joint_angles": left_arm_joint_angles,
                "left_gripper_width": left_gripper_width,
            }
        except Exception as e:
            print(f"Error getting robot state: {str(e)}")
            return None

    async def _execute_control(self, websocket, command):
        """Execute control command"""
        if command['type'] == 'humanoid_control':
            print(f"Executing trajectory: {command['trajectory'][0]}...")
            if len(command['trajectory']) == 1:
                gci.set_arm_joint_angles(
                    arm_joint_angles=command['trajectory'][0],
                    arm="left_arm",
                    asynchronous=False,
                )
            elif len(command['trajectory']) > 1:
                gci.follow_trajectory(
                    solution= {"positions": command['trajectory']},
                    hardware="left_arm",
                    asynchronous=False,
                    frequency=50,
                )
        elif command['type'] == 'gripper_control':
            if command['gripper_value'] == 0:
                gci.set_gripper_close(
                    gripper="left_gripper",
                    asynchronous=False,
                )
            elif command['gripper_value'] == 1:
                gci.set_gripper_open(
                    gripper="left_gripper",
                    asynchronous=False,
                )
        elif command['type'] == 'head_control':
            print(f"Setting head angles: {command['head_value']}...")
            gci.set_head_joint_angles(
                head_joint_angles=command['head_value'],
                asynchronous=False,
            )
        elif command['type'] == 'leg_control':
            print(f"Setting leg angles: {command['leg_value']}...")
            gci.set_leg_joint_angles(
                leg_joint_angles=command['leg_value'],
                asynchronous=False,
            )
        elif command['type'] == 'humanoid_data':
            data = self._humanoid_data()
            await websocket.send(json.dumps(data))
        else:
            print(f"Unknown command type: {command['type']}")
            
    def _handle_interrupt(self, signum, frame):
        """处理 CTRL+C 中断信号"""
        print("\nReceived interrupt signal (CTRL+C), shutting down...")
        self.running = False

def main():
    host = os.getenv("SERVER_HOST", "localhost")
    port = os.getenv("SERVER_PORT", "8765")
    print(host, port)
    client = HumanoidClient(f"ws://{host}:{port}/humanoid")
    client.start_receive_thread()
    # try:
    #     await client.connect("ws://192.168.54.158:8765/humanoid")
    # except KeyboardInterrupt:
    #     print("Shutting down gracefully...")
    while client.running:
        time.sleep(1) 

if __name__ == "__main__":
    # asyncio.run(main())
    main()
