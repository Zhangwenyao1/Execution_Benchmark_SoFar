import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

import argparse
from copy import deepcopy
from PIL import Image
import shutil
from random import choice
import torch
import numpy as np
from rich import print as rich_print
from tqdm import tqdm
from transforms3d.quaternions import quat2mat, mat2quat

from src.mujoco.env_utils import add_state_to_config, convert_graspnet_mesh_type
from src.utils.data import get_mesh_file, get_traj_list, get_traj
from src.utils.config import load_config, to_dot_dict
from src.utils.vis_plotly import Vis
from src.utils.convert_gripper import convert_gripper, goal_rot_to_gripper_rot
from src.utils.utils import silent_call, get_random_init_qpos, get_random_init_qpos_neutral, coll
from src.utils.ik import IK
from src.plan.plan import Planner
from src.utils.scene import Scene
try:
    from src.curobo.plan import CuroboPlanner
except:
    CuroboPlanner = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/v1.1')
    parser.add_argument('--old_root', type=str, default='data/v1.1_benchmark')
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--total', type=int, default=None)
    parser.add_argument('--part', type=int, default=None)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--time', type=int, default=60)
    parser.add_argument('--first', type=int, default=0)
    parser.add_argument('--curobo', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/mujoco.yaml')
    args = parser.parse_args()

    vis = Vis()
    ik = IK()
    scene = Scene()

    traj_list = get_traj_list(args.old_root, args.mode)[args.part::args.total][:args.N]
    PlannerClass = CuroboPlanner if args.curobo else Planner

    for traj_name in tqdm(traj_list):
        traj = get_traj(traj_name, args.old_root, args.mode, unpack=True)
        
        config = load_config(args.config)['mujoco']
        config.obj = [to_dot_dict(x) for x in convert_graspnet_mesh_type(traj, mujoco_mesh=(not args.curobo))['obj']]
        planner = PlannerClass(config, fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])

        init, goal = traj['joint_qpos'][0], traj['joint_qpos'][-1]
        try:
            res, path = planner.plan(init, goal, interpolate_num=75, fix_joints_value={'panda_finger_joint1': init[-2].item(), 'panda_finger_joint2': init[-1].item()}, time=args.time, first=args.first)
        except:
            res = False
        if not res:
            continue

        is_collision = False
        obj_pc, obj_pc_sparse = (scene.get_obj_pc(dict(obj=config['obj']), sparse=sparse) for sparse in [False, True])
        for p in path:
            if scene.check_coll(dict(qpos=p, **config), obj_pc=obj_pc, obj_pc_sparse=obj_pc_sparse)[0]:
                is_collision = True
                break
        print(is_collision)
        if is_collision:
            continue

        state = np.ones(len(path))
        state[0] = 0
        state[-1] = 2

        eef_trans, eef_quat = [], []
        for p in path:
            rot, trans = ik.fk(p[:7])
            eef_trans.append(trans)
            eef_quat.append(mat2quat(rot))

        data = dict(robot_trans=np.eye(4),
            joint_qpos=path,
            state=state,
            final_width=traj['final_width'],
            eef_traj=np.concatenate([np.stack(eef_trans), np.stack(eef_quat)], axis=-1),
            obj=traj['obj'],
            grasp_obj=traj['grasp_obj'],
        )
        save_path = os.path.join(args.root, traj_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **data)

        planner.close()

if __name__ == "__main__":
    main()