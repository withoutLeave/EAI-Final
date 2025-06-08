"""
File: realsense_camera_ros.py
Author: Xiaomeng Fang
Description: Rospy (ROS1) script for realsense camera

History:
    - Version 0.0 (2024-01-26): careated

Dependencies:
    - rospy
"""

import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
import open3d as o3d
from packaging import version
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RealSenseCameraROS:

    TIME_MAX: float = 5.0
    ATTEMPTS_MAX: int = 10

    @staticmethod
    def img_msg_to_cv2(img_msg: Image, desired_encoding="passthrough") -> np.ndarray:
        """Convert a ROS Image message to an OpenCV image.
        Args:
            img_msg (sensor_msgs.msg.Image): The ROS Image message.
            desired_encoding (str): The desired encoding for the output image.
        Returns:
            np.ndarray: The converted OpenCV image.
        """
        if desired_encoding == "passthrough":
            encoding = img_msg.encoding
        else:
            encoding = desired_encoding

        if encoding == "mono8":
            dtype = np.uint8
        elif encoding in ("mono16", "16UC1"):
            dtype = np.uint16
        elif encoding == "bgr8":
            dtype = np.uint8
        elif encoding == "rgb8":
            dtype = np.uint8
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Convert the ROS Image message to a numpy array
        image_raw = np.frombuffer(img_msg.data, dtype=dtype).reshape(
            img_msg.height, img_msg.width, -1
        )

        image = image_raw.copy()
        if encoding == "bgr8":
            return image
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding in ("mono8", "mono16", "16UC1"):
            return image[:, :, 0]
        raise ValueError(f"Unsupported encoding: {encoding}")

    @classmethod
    def get_rgb_image(cls, rgb_topic: str) -> Optional[np.ndarray]:
        """Subscribe rgb image.
        Args:
            rgb_topic (str): topic name
        Returns:
            np.ndarray: rgb image or None if not received
        """
        try:
            rgb = rospy.wait_for_message(
                topic=rgb_topic, topic_type=Image, timeout=cls.TIME_MAX
            )
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None
        if rgb is not None:
            rgb_frame = cls.img_msg_to_cv2(rgb, "bgr8")
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            return rgb_frame
        return None

    @classmethod
    def get_depth_image(cls, depth_topic: str) -> Optional[np.ndarray]:
        """Subscribe depth image.
        Args:
            depth_topic (str): topic name
        Returns:
            np.ndarray: depth image or None if not received
        """
        try:
            depth = rospy.wait_for_message(
                topic=depth_topic, topic_type=Image, timeout=cls.TIME_MAX
            )
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None
        if depth is not None:
            depth_frame = cls.img_msg_to_cv2(depth, "passthrough")
            return depth_frame
        return None

    @classmethod
    def get_left_ir_image(cls, left_ir_topic: str) -> Optional[np.ndarray]:
        """Subscribe left infrared image.
        Args:
            left_ir_topic (str): topic name
        Returns:
            np.ndarray: left infrared image or None if not received
        """
        try:
            left_ir = rospy.wait_for_message(
                topic=left_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
            )
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None
        if left_ir is not None:
            left_ir_frame = cls.img_msg_to_cv2(left_ir, "bgr8")
            return left_ir_frame
        return None

    @classmethod
    def get_right_ir_image(cls, right_ir_topic: str) -> Optional[np.ndarray]:
        """Subscribe right infrared image.
        Args:
            right_ir_topic (str): topic name
        Returns:
            np.ndarray: right infrared image or None if not received
        """
        try:
            right_ir = rospy.wait_for_message(
                topic=right_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
            )
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None
        if right_ir is not None:
            right_ir_frame = cls.img_msg_to_cv2(right_ir, "bgr8")
            return right_ir_frame
        return None

    @classmethod
    def get_ir_image(
        cls, left_ir_topic: str, right_ir_topic: str, time_align_threshold=0.01
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Subscribe left and right infrared images and ensure they are time-aligned.
        Args:
            left_ir_topic (str): left infrared topic name
            right_ir_topic (str): right infrared topic name
            time_align_threshold (float): time alignment threshold, default is 0.01s
        Returns:
            np.ndarray: left and right infrared images
        """
        try:
            attempts = 0
            while attempts < cls.ATTEMPTS_MAX:
                left_ir = rospy.wait_for_message(
                    topic=left_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                right_ir = rospy.wait_for_message(
                    topic=right_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                if left_ir is None or right_ir is None:
                    rospy.logwarn(
                        f"Failed to receive infrared images. {attempts+1}/{cls.ATTEMPTS_MAX-1}th retrying..."
                    )
                    attempts += 1
                    continue

                left_ir_stamp = left_ir.header.stamp
                right_ir_stamp = right_ir.header.stamp

                if (
                    abs((left_ir_stamp - right_ir_stamp).to_sec())
                    < time_align_threshold
                ):
                    left_ir_frame = cls.img_msg_to_cv2(left_ir, "bgr8")
                    right_ir_frame = cls.img_msg_to_cv2(right_ir, "bgr8")
                    return left_ir_frame, right_ir_frame
                else:
                    rospy.logwarn(
                        f"Infrared images are not time-aligned. {attempts+1}/{cls.ATTEMPTS_MAX-1}th retrying..."
                    )
                attempts += 1

            rospy.logwarn(
                "Failed to get time-aligned infrared images after maximum attempts."
            )
            return None, None
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None, None

    @classmethod
    def get_ir_and_rgb_image(
        cls,
        rgb_topic: str,
        left_ir_topic: str,
        right_ir_topic: str,
        time_align_threshold=0.01,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subscribe rgb, left and right infrared images and ensure they are time-aligned.
        Args:
            rgb_topic (str): rgb topic name
            left_ir_topic (str): left infrared topic name
            right_ir_topic (str): right infrared topic name
            time_align_threshold (float): time alignment threshold, default is 0.01s
        Returns:
            np.ndarray: rgb, left and right infrared images
        """
        try:
            attempts = 0
            while attempts < cls.ATTEMPTS_MAX:
                rgb = rospy.wait_for_message(
                    topic=rgb_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                left_ir = rospy.wait_for_message(
                    topic=left_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                right_ir = rospy.wait_for_message(
                    topic=right_ir_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                rgb_stamp = rgb.header.stamp
                left_ir_stamp = left_ir.header.stamp
                right_ir_stamp = right_ir.header.stamp

                if (
                    abs((rgb_stamp - left_ir_stamp).to_sec()) < time_align_threshold
                    and abs((rgb_stamp - right_ir_stamp).to_sec())
                    < time_align_threshold
                    and abs((left_ir_stamp - right_ir_stamp).to_sec())
                    < time_align_threshold
                ):
                    rgb_frame = cls.img_msg_to_cv2(rgb, "bgr8")
                    rgb_frame = rgb_frame[:, :, [2, 1, 0]]
                    left_ir_frame = cls.img_msg_to_cv2(left_ir, "bgr8")
                    right_ir_frame = cls.img_msg_to_cv2(right_ir, "bgr8")
                    return rgb_frame, left_ir_frame, right_ir_frame
                else:
                    rospy.logwarn(
                        f"RGB and infrared images are not time-aligned. {attempts+1}/{cls.ATTEMPTS_MAX-1}th retrying..."
                    )
                attempts += 1

            rospy.logwarn(
                "Failed to get time-aligned RGB and infrared images after maximum attempts."
            )
            return None, None, None
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None, None, None

    @classmethod
    def get_rgbd_image(
        cls, rgb_topic: str, depth_topic: str, time_align_threshold=0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Subscribe rgb and depth images and ensure they are time-aligned.
        Args:
            rgb_topic (str): rgb topic name
            depth_topic (str): depth topic name
            time_align_threshold (float): time alignment threshold, default is 0.01s
        Returns:
            np.ndarray: rgb and depth images
        """
        try:
            attempts = 0
            while attempts < cls.ATTEMPTS_MAX:
                rgb = rospy.wait_for_message(
                    topic=rgb_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                depth = rospy.wait_for_message(
                    topic=depth_topic, topic_type=Image, timeout=cls.TIME_MAX
                )
                rgb_stamp = rgb.header.stamp
                depth_stamp = depth.header.stamp

                if abs((rgb_stamp - depth_stamp).to_sec()) < time_align_threshold:
                    rgb_frame = cls.img_msg_to_cv2(rgb, "bgr8")
                    rgb_frame = rgb_frame[:, :, [2, 1, 0]]
                    depth_frame = cls.img_msg_to_cv2(depth, "passthrough")
                    return rgb_frame, depth_frame
                else:
                    rospy.logwarn(
                        f"RGB and depth images are not time-aligned. {attempts+1}/{cls.ATTEMPTS_MAX-1}th retrying..."
                    )
                attempts += 1

            rospy.logwarn(
                "Failed to get time-aligned RGB and depth images after maximum attempts."
            )
            return None, None
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None, None

    @classmethod
    def get_camera_info(cls, topic: str) -> Optional[dict]:
        """Get camera info.
        Args:
            topic (str): topic name
        Returns:
            dict: camera info dictionary or None if not found
        """
        try:
            camera_info = rospy.wait_for_message(
                topic=topic, topic_type=CameraInfo, timeout=cls.TIME_MAX
            )
            if camera_info is not None:
                return {
                    "fx": camera_info.K[0],
                    "fy": camera_info.K[4],
                    "cx": camera_info.K[2],
                    "cy": camera_info.K[5],
                    "width": camera_info.width,
                    "height": camera_info.height,
                }
            else:
                rospy.logerr("Received None for camera_info.")
                return None
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None

    @staticmethod
    def is_point_in_mask(point: np.ndarray, mask: np.ndarray, camera_info: CameraInfo):
        """Check if a point is in a mask.
        Args:
            point (np.array): 3D point in camera frame
            mask (np.array): mask image
            camera_info (CameraInfo): camera info
        Returns:
            bool: True if the point is in the mask, False otherwise
        """
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        camera_matrix = np.array(camera_info.K).reshape((3, 3))
        dist_coeffs = np.array(camera_info.D)
        image_point, _ = cv2.projectPoints(
            point, rvec, tvec, camera_matrix, dist_coeffs
        )
        image_point = image_point.squeeze().astype(np.int)
        x = np.clip(image_point[0], 0, camera_info.width - 1)
        y = np.clip(image_point[1], 0, camera_info.height - 1)
        return mask[y, x] is True

    @staticmethod
    def create_point_cloud(
        depth: np.ndarray,
        intrinsics: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
        scale: float = 1000.0,
    ) -> o3d.geometry.PointCloud:
        """Create a point cloud from depth and RGB images
        Args:
            depth (np.ndarray, optional): depth image
            rgb (np.ndarray, optional): rgb image
            mask (np.ndarray, optional): mask image. Defaults to None.
            intrinsics (np.ndarray, optional): color intrinsics. Defaults to None. 3x3 matrix
            camera_pose (np.ndarray, optional): camera pose. Defaults to None. 4x4 matrix
            scale (float, optional): depth scale. Defaults to 1000.0.
        Returns:
            pcd (o3d.geometry.PointCloud()): point cloud
        """
        assert scale > 0, "Depth scale cannot be negative or zero."
        assert depth is not None, "Depth image is required to create a point cloud."
        assert intrinsics is not None, "Intrinsics is required to create a point cloud."

        depth = depth / scale
        height, width = depth.shape
        if mask is not None:
            depth = depth * mask

        fx, fy, cx, cy = (
            intrinsics[0, 0],
            intrinsics[1, 1],
            intrinsics[0, 2],
            intrinsics[1, 2],
        )

        # Create a meshgrid of pixel coordinates
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        u, v = np.meshgrid(x, y)

        # Backproject depth to 3D points
        z = depth.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        points_camera = np.vstack((x, y, z)).T

        # Create a point cloud
        pcd = o3d.geometry.PointCloud()

        if camera_pose is None:
            points = points_camera
        else:
            rotation_matrix = camera_pose[:3, :3]
            translation_vector = camera_pose[:3, 3]
            points = rotation_matrix.dot(points_camera.T).T + translation_vector
        pcd.points = o3d.utility.Vector3dVector(points)

        if rgb is not None:
            colors = rgb.reshape(-1, 3) / 255  # Normalize colors
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    @classmethod
    def get_point_cloud(
        cls,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        rgb: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        camera_pose: Optional[np.ndarray] = None,
        scale: float = 1000.0,
    ) -> o3d.geometry.PointCloud:
        """Create a point cloud from depth and RGB images
        Args:
            depth (np.ndarray, optional): depth image
            rgb (np.ndarray, optional): rgb image
            mask (np.ndarray, optional): mask image. Defaults to None.
            intrinsics (np.ndarray, optional): color intrinsics. Defaults to None. 3x3 matrix
            camera_pose (np.ndarray, optional): camera pose. Defaults to None. 4x4 matrix
            scale (float, optional): depth scale. Defaults to 1000.0.
        Returns:
            pcd (o3d.geometry.PointCloud()): point cloud
        """
        return cls.create_point_cloud(depth, intrinsics, rgb, mask, camera_pose, scale)

    @staticmethod
    def get_intrinsics(camera_info: CameraInfo) -> np.ndarray:
        """Get intrinsics from camera.
        Args:
            camera_info (CameraInfo): camera info
        Returns:
            np.ndarray: 3x3 matrix of intrinsics
        """
        return np.array(camera_info.K).reshape((3, 3))

    @classmethod
    def get_extrinsics(cls, extrinsics_topic: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get extrinsics from camera.
        Args:
            extrinsics_topic (str): topic name
        Returns:
            np.ndarray: rotation matrix, 4x4 matrix
        """
        from realsense2_camera.msg import Extrinsics

        try:
            extrinsics = rospy.wait_for_message(
                extrinsics_topic,
                Extrinsics,
                timeout=cls.TIME_MAX,
            )
            T = np.eye(4)
            T[:3, :3] = np.array(extrinsics.rotation).reshape((3, 3))
            T[:3, 3] = np.array(extrinsics.translation)
            return T
        except Exception as e:
            rospy.logerr(f"Error: {e}")
            return None

    @staticmethod
    def align_depth_to_color(
        depth_img: np.ndarray,
        color_intrinsics: np.ndarray,
        depth_intrinsics: np.ndarray,
        depth_to_color_extrinsics: np.ndarray,
    ) -> np.ndarray:
        """Align depth image to color image.
        Args:
            depth_img (np.ndarray): depth image
            color_intrinsics (np.ndarray): color intrinsics, 3x3 matrix
            depth_intrinsics (np.ndarray): depth intrinsics, 3x3 matrix
            depth_to_color_extrinsics (np.ndarray): depth to color extrinsics, 4x4 matrix
        Returns:
            np.ndarray: aligned depth image
        """

        # depth intrinsics
        fx_d = depth_intrinsics[0, 0]
        fy_d = depth_intrinsics[1, 1]
        cx_d = depth_intrinsics[0, 2]
        cy_d = depth_intrinsics[1, 2]

        # color intrinsics
        fx_rgb = color_intrinsics[0, 0]
        fy_rgb = color_intrinsics[1, 1]
        cx_rgb = color_intrinsics[0, 2]
        cy_rgb = color_intrinsics[1, 2]

        # camera extrinsics
        R, T = depth_to_color_extrinsics[:3, :3], depth_to_color_extrinsics[:3, 3]

        # get images
        height, width = depth_img.shape

        # create aligned depth image
        aligned_depth_img = np.zeros((height, width), dtype=np.uint16)

        for v in range(height):
            for u in range(width):
                depth = depth_img[v, u] / 1000.0
                if depth > 0:
                    z = depth
                    x = (u - cx_d) * z / fx_d
                    y = (v - cy_d) * z / fy_d

                    xyz_depth = np.array([x, y, z])
                    xyz_rgb = np.dot(R, xyz_depth) + T

                    u_rgb = int(fx_rgb * xyz_rgb[0] / xyz_rgb[2] + cx_rgb)
                    v_rgb = int(fy_rgb * xyz_rgb[1] / xyz_rgb[2] + cy_rgb)

                    if 0 <= u_rgb < width and 0 <= v_rgb < height:
                        aligned_depth_img[v_rgb, u_rgb] = depth_img[v, u]

        return aligned_depth_img

    @staticmethod
    def rotation_vector_to_quaternion(rvec: np.ndarray) -> np.ndarray:
        """Convert rotation vector to quaternion.
        Args:
            rvec (np.ndarray): rotation vector, 3x1 vector
        Returns:
            np.ndarray: quaternion, qx, qy, qz, qw
        """
        # Normalize the rotation vector
        norm = np.linalg.norm(rvec)
        if norm < 1e-5:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        axis = rvec / norm

        # Compute sine and cosine of half the rotation angle
        half_angle = norm / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)

        # Compute quaternion components
        qw = cos_half
        qx = axis[0] * sin_half
        qy = axis[1] * sin_half
        qz = axis[2] * sin_half

        return np.array([qx, qy, qz, qw])

    @classmethod
    def get_aruco_marker_pose(
        cls, rgb: np.ndarray, color_intrinsics: np.ndarray, visualizer=False
    ):
        """get aruco marker pose
        Args:
            visualizer (bool, optional): whether to visualize the process. Defaults to False.
        Returns:
            np.ndarray: pose of the aruco marker, [x, y, z, qx, qy, qz, qw]
        """
        intr_matrix = color_intrinsics
        intr_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(rgb)
        else:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(rgb, aruco_dict, parameters)
        if ids is None:
            return None

        marker_length = 0.1  # 10 cm, the length of the marker's side
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, intr_matrix, intr_coeffs
        )

        if visualizer:
            try:
                aruco.drawDetectedMarkers(rgb, corners)
                axis_length = 0.05
                result_img = cv2.drawFrameAxes(
                    rgb, intr_matrix, intr_coeffs, rvec, tvec, axis_length
                )
                cv2.imshow("RGB image", result_img)
                cv2.waitKey(0)
            except Exception as e:
                rospy.logerr(f"Error: {e}")

        quat = cls.rotation_vector_to_quaternion(np.array(rvec[0][0]))
        pose = np.concatenate((tvec[0][0], quat))

        return pose

    @classmethod
    def get_camera_pose(cls, base_frame: str, camera_frame: str) -> Optional[np.ndarray]:
        """Get the camera pose from tf tree.
        Args:
            base_frame (str): the target frame, e.g., robot_base_link
            camera_frame (str): the source frame, e.g., left_arm_camera_link
        Returns:
            np.ndarray: camera pose as [x, y, z, qx, qy, qz, qw] or None if not found
        """
        import time
        import tf2_ros

        tf_buffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tf_buffer)

        timeout = rospy.Duration(cls.TIME_MAX)
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
                if time.time() - start_time > cls.TIME_MAX:
                    rospy.logwarn(f"TF lookup timed out after {cls.TIME_MAX} seconds: {e}")
                    return None
                rospy.logwarn(f"TF lookup failed: {e}, retrying...")
                rospy.sleep(0.1)


if __name__ == "__main__":
    rospy.init_node("realsense_camera_ros", anonymous=True)
    camera = RealSenseCameraROS()
    rgb_topic = "/cam/head_f/color/image_raw"
    depth_topic = "/cam/head_f/aligned_depth_to_color/image_raw"
    camera_info_topic = "/cam/head_f/color/camera_info"

    rgb_image = camera.get_rgb_image(rgb_topic)
    depth_image = camera.get_depth_image(depth_topic)
    camera_info = camera.get_camera_info(camera_info_topic)
    camera_pose = camera.get_camera_pose(
        base_frame="base_link", camera_frame="front_head_camera_color_optical_frame"
    )

    if rgb_image is not None:
        print("Color image received.")
        cv2.imwrite("rgb_image.png", rgb_image)

    if depth_image is not None:
        print("Depth image received.")
        cv2.imwrite("depth_image.png", depth_image)

    if camera_info is not None:
        print("Camera Info:", camera_info)

    if camera_pose is not None:
        print("Camera Pose:", camera_pose)
