# https://github.com/lyfkyle/pybullet_ompl
import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

import math
import sys
import torch
from transforms3d.quaternions import quat2mat, mat2quat
from src.plan import pb_ompl
from src.utils.vis_plotly import Vis
from src.utils.config import DotDict
from src.utils.utils import to_list, to_torch
from src.utils.robot_model import RobotModel
import numpy as np
from src.utils.ik import IK


class Planner:
    def __init__(self, config, fix_joints=[], planner="AITstar"):
        self.config = config

        # load robot
        robot = RobotModel(config.urdf)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot, config, fix_joints=fix_joints)
        self.pb_ompl_interface.set_planner(planner)

    def clear_obstacles(self):
        raise NotImplementedError
        self.obstacles = []

    def plan(self, start=None, goal=None, interpolate_num=None, fix_joints_value=dict(), time=None, first=None, only_test_start_end=False):
        if start is None:
            start = [0,0,0,-1,0,1.5,0, 0.02, 0.02]
        if goal is None:
            goal = [1,0,0,-1,0,1.5,0, 0.02, 0.02]
        # goal = [0,1.5,0,-0.1,0,0.2,0, 0.02, 0.02]

        self.pb_ompl_interface.fix_joints_value = fix_joints_value
        start, goal = to_list(start), to_list(goal)
        for name, pose in [('start', start), ('goal', goal)]:
            if not self.pb_ompl_interface.is_state_valid(pose):
                print(f'unreachable {name}')
                return False, None
        if only_test_start_end:
            return True, None

        res, path = self.pb_ompl_interface.plan(start, goal, interpolate_num=interpolate_num, fix_joints_value=fix_joints_value, allowed_time=time, first=first)
        if res:
            path = np.array(path)
        return res, path

    def close(self):
        pass

if __name__ == '__main__':
    pcd = np.load('/data/workspace/SimplerEnv/scene.npy')
    cfg = DotDict(
    #    urdf='robot_models/franka/franka_with_gripper_extensions.urdf',
        urdf='/data/workspace/SimplerEnv/ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf',
        pc=torch.tensor(pcd),
    )
    
    
    
    # ik = IK(robot='franka')
    ik = IK(robot='google_robot')
    ee_pose_t = np.array([0.5, 0.5, 0.8])
    ee_pose_r = np.eye(3)
    init = np.array([0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.1, 0.02, 0.02])
    goal = np.array([0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.1, 0.02, 0.02])
    goal = np.array([ 1.06693511,  1.74489278,  0.41229917, -1.38537161, -0.86461342,
       -0.54534418, -0.20472945])
    # init = np.array([-0.26394573,  0.08319134,  0.50176114,  1.156859  ,  0.02858367,
    #     1.5925982 , -1.080653  ,  0.        ,  0.        , -0.00285961,
    #     0.7851361 ])
    # ik.fk(init)
    goal = np.concatenate([ik.ik(ee_pose_t, ee_pose_r, joints=ik.robot_to_arm_joints(init)), np.array([0.02, 0.02])])
    
    init = np.array([-0.26394573, 0.08319134, 0.50176114, 1.156859, 0.02858367, 1.5925982, -1.080653  ])
    goal = np.array([ 1.06693511,  1.74489278,  0.41229917, -1.38537161, -0.86461342, -0.54534418, -0.20472945])
    
    
    env = Planner(cfg, planner='RRTConnect', fix_joints=['joint_head_pan', 'joint_head_tilt', 'joint_finger_right', 'joint_finger_left'])
    path = []
    # while len(path) == 0:
    res, path = env.plan(init, goal, fix_joints_value={'joint_finger_right': 0.1, 'joint_finger_left': 0.1, 'joint_head_pan': 0.1, 'joint_head_tilt': 0.1}, interpolate_num=50, time=15)
    print(path)
    if len(path) != 0:
        env.pb_ompl_interface.scene.to_plotly_traj(path, cfg.pc)
    # ee_path = []
    # for p in path:
    #     ee_path.append(ik.fk(ik.robot_to_arm_joints(p)))
    # print(res)
    # print(path)