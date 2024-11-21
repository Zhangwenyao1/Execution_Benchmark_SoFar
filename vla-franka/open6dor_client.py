import os
import sys
import zmq
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging
from zmq import Poller
import pickle
import json
import panda_py
from panda_py import libfranka
import cv2
from spatialmath import SE3

from realsense import RealSenseCamera

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
import time
import argparse
from transforms3d.euler import quat2euler

REQUEST_DATA = {}
# 'text': 'pick up the elephant',
# 'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
# 'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
# 'proprio_array': [np.zeros((7,), dtype=np.float32)],
# 'traj_metadata': None,
# 'env_id': 1,

import pyrealsense2 as rs
import open3d as o3d


def list_all_devices():
    context = rs.context()
    devices = context.query_devices()
    for i, dev in enumerate(devices):
        print(f"Device {i}: {dev}")


def show_camera(serial_number, clip_duration=3.0):
    camera = RealSenseCamera(serial_number, fps=30)
    # for _ in range(int(1000 * clip_duration)):
    while (True):
        camera.show_frames()
        # norm_color = reshape(color, 480, 480, True)
    del camera


def quat_to_rpy(quat):
    r = R.from_quat(quat)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return roll, pitch, yaw


def matpose_to_quatpose(matpose):
    translation = matpose[:3, 3]
    rotation = matpose[:3, :3]
    quaternion = R.from_matrix(rotation).as_quat()
    return np.hstack([translation, quaternion])


def creatematpose(rotation, translation):
    tpose = np.eye(4)
    tpose[:3, :3] = rotation
    tpose[:3, 3] = translation
    return tpose


def quatpose_to_matpose(qpose):
    translation = qpose[:3]
    quaternion = qpose[3:]
    rotation = R.from_quat(quaternion).as_matrix()
    return creatematpose(rotation, translation)


def rpy_to_quat(roll, pitch, yaw):
    # Create a Rotation object from RPY angles
    r = R.from_euler('xyz', [roll, pitch, yaw])
    # Convert the Rotation object to a quaternion
    quat = r.as_quat()
    return quat


def reshape(image, cropx, cropy, color=False):
    cv2.destroyAllWindows()
    y, x, z = image.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    new_image = image[starty:starty + cropy, startx:startx + cropx, :]
    res = cv2.resize(new_image,
                     dsize=(256, 256),
                     interpolation=cv2.INTER_CUBIC)
    show_res = rgb_to_bgr(res)
    if color:
        cv2.imshow('Reshaped Image', show_res)
        cv2.waitKey(1000)
        # cv2.destroyAllWindows()
    return res


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class vla_controller:

    def __init__(self, host, port, robot_host, serial_number):
        # self.desk = panda_py.Desk(robot_host, username, password)
        # self.desk.unlock()
        # self.desk.activate_fci()
        self.panda = panda_py.Panda(robot_host)
        self.gripper = libfranka.Gripper(robot_host)
        # self.panda.move_to_start()
        start_joint_pose = [0, -0.5585, 0, -2.3038, 0, 1.6580, 0]
        joint_speed_factor = 0.2
        cart_speed_factor = 0.2
        stiffness = [600, 600, 600, 600, 250, 150, 50]
        self.panda.move_to_joint_position(start_joint_pose,
                                          speed_factor=joint_speed_factor,
                                          stiffness=stiffness)
        # start_quat = rpy_to_quat(1.57, 0, 0)
        # print(f"start qut: {start_quat}")
        # start_quatpose = np.hstack([[0.36267508, -0.00118338,  0.43808436], start_quat])
        # start_matpose = quatpose_to_matpose(start_quatpose)
        # self.panda.move_to_pose(start_matpose)
        print("start to set gripper")
        # self.gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        self.gripper.move(0.08, 0.2)
        current_pose = self.panda.get_pose()
        current_pose_qut = matpose_to_quatpose(current_pose)
        print("========= current pose", current_pose_qut)
        row, pitch, yaw = quat_to_rpy(current_pose_qut[3:])
        print(f"======== row, pitch, yaw: {row}, {pitch}, {yaw}")
        # self.gripper.move(2.0, 0.2)
        print("set gripper done")
        self.camera = RealSenseCamera(serial_number, fps=30)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.current_pose = None
        self.gripper_status = 0
        print("connected to vla server")

    def move_gripper(self, action):
        if action == 1:
            self.gripper.grasp(0, 0.2, 10, 0.04, 0.04)
        if action == 0:
            self.gripper.move(0.08, 0.2)

    def set_action(self, action, gripper_action):
        """
        set ee pose and gripper state
        """
        # target_quat = rpy_to_quat(action[3], action[4], action[5])
        # target_quatpose = np.hstack([action[:3], target_quat])
        # self.panda.move_to_pose(target_quatpose,speed_factor=0.2)
        self.panda.move_to_pose(action)

        if (gripper_action <= -0.9 and self.gripper_status == 0):
            self.move_gripper(1)
            self.gripper_status = 1
        elif (gripper_action > -0.9 and self.gripper_status == 1):
            self.move_gripper(0)
            self.gripper_status = 0

        return

    def get_ee_pos(self):
        ee_matpos = self.panda.get_pose()
        self.current_pose = ee_matpos.copy()
        ee_quatpos = matpose_to_quatpose(ee_matpos)
        return ee_quatpos

    def get_camera(self):
        color, depth, _, _ = self.camera.get_aligned_frames()
        return color, depth

    def send(self, message):
        # print('Sending message:', message)
        # request = json.dumps(message).encode()
        self.socket.send_pyobj(message)
        response = self.socket.recv_pyobj()
        # print('Received response:', response)
        print("========== received response ==========")
        # print(f"proprio: {response['debug']['proprio']}")
        # print(f"pose: {response['debug']['pose']}")
        # print(f"action: {response['action']}")
        print(f"goal pose: {response['debug']['goal_pose']}")
        print("========== received response ==========")
        return response

    # def __del__(self):
    # self.socket.close()
    # self.context.destroy()


def run_client():
    host = '103.237.29.216'
    port = 6667
    robot_host = '172.16.0.2'
    serial_number = "233522075838"
    vla = vla_controller(host, port, robot_host, serial_number)

    while True:

        ee_pos = vla.get_ee_pos()
        color, depth = vla.get_camera()
        message = {}
        message["text"] = "Pick up toy large elephant."
        # print(color.shape)
        norm_color = reshape(color, 480, 480, True)
        # print(norm_color.shape)
        message["image_array"] = [norm_color]
        # norm_depth = reshape(depth, 256, 256)
        message["depth_array"] = [depth]
        message["proprio_array"] = [ee_pos]
        message["traj_metadata"] = None
        message["env_id"] = 1
        # print(message)

        ee_pos_rpy = quat2euler(ee_pos[3:])
        ee_pos_trans = ee_pos[:3]

        # input("Press Enter to continue...")
        poses = []

        poses.append(ee_pos)

        # vla.set_action(SE3.Trans(0, 0.02, 0) * vla.current_pose )
        print(
            f"======= ee pos transition: {ee_pos_trans}, rpy: {ee_pos_rpy} ========="
        )
        response = vla.send(message)
        
        if response["info"] == "success":
            action = response["action"][0]
            ee_action = SE3.Trans(
                action[0], action[1], action[2]) * vla.current_pose * SE3.Rx(
                    action[3]) * SE3.Ry(action[4]) * SE3.Rz(action[5])

            # target_rpy = response["debug"]["pose"][1]
            # print(f"target rpy: {target_rpy}")
            # target_quat = rpy_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
            # target_quatpose = np.hstack([response["debug"]["pose"][0], target_quat])
            # target_matpose = quatpose_to_matpose(target_quatpose)
            # ee_pos = target_matpose
            # vla.set_action(ee_pos, -1.0)

            grasp_pos = action[:3]
            grasp_quat = action[3:]
            pre_grasp_pos = grasp_pos - 0.15 * R.from_quat(
                grasp_quat).as_matrix()[:, 0]
            # R.from_quatpose()
            # modify_grasp_pre.translation -= 0.15 * modify_grasp_pre.rotation_matrix[:,
            #                                                                         0]  # tcp
            # return
            vla.set_action(ee_pos, action[6])
        else:
            print("Failed to get action")
            time.sleep(0.1)
        # break


if __name__ == '__main__':
    # print(f"Device: {list_all_devices()}")
    # show_camera("233522075838")
    run_client()
