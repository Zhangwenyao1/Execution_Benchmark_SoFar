
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import debugpy
debugpy.listen(('0.0.0.0', 5683))
print('Waiting for debugger attach')
debugpy.wait_for_client()
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import glob
import draccus
import numpy as np
import tqdm

from PIL import Image
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
import json
import imageio
from typing import Optional, Union
from dataclasses import dataclass
import torch
from collections import Counter
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import axangle2quat, quat2axangle, mat2quat, quat2mat
import open3d as o3d




from GSNet.gsnet_simpler import grasp_inference, visualize_plotly
from SpatialAgent.segmentation import florence, sam

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")



from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
import argparse
import re
from libero.libero.utils.bddl_generation_utils import *


import plan.src.utils.config as config
from plan.src.plan import pb_ompl
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.config import DotDict
from plan.src.utils.utils import to_list, to_torch
from plan.src.utils.robot_model import RobotModel
from plan.src.utils.scene import Scene
from plan.src.utils.ik import IK
from plan.src.utils.vis_plotly import Vis
from plan.src.utils.constants import ARM_URDF, ARM_URDF_FULL, ROBOT_JOINTS, ROBOT_URDF, FRANKA_COLLISION_FILE, FRANKA_CUROBO_FILE


sys.path.append(BASE_DIR+'/SpatialAgent')
from SpatialAgent.depth.utils import transform_point_cloud_nohw, inverse_transform_point_cloud
# from SpatialAgent_o.open6dor import sofar
from SpatialAgent.simpler_env import sofar
from termcolor import colored

# absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
# 





class Planner:
    def __init__(self, config, fix_joints=[], planner="RRTConnect"):
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


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, mp4_path = None):
    """Saves an MP4 replay of an episode."""
    # rollout_dir = f"/data/workspace/LIBERO/rollouts/{DATE}"
    # # os.makedirs(rollout_dir, exist_ok=True)
    # processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    # mp4_path = f"{rollout_dir}/{result}"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


@dataclass
class GenerateConfig:
    # fmt: off
                 # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                   # Number of rollouts per task
    max_steps: int = 650
    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def get_grasp_pose(task_description, object_pc_cam, scene_pc_cam, scene_pc, extrinsics, relative_translation_table, relative_rotation_table):

    # try:
    gg_group, gg_goal_group = grasp_inference(task_description, object_pc_cam, scene_pc_cam, scene_pc, extrinsics, relative_translation_table, relative_rotation_table)
    if gg_group is None:
        return None, None
    print('Grasp Inference Completed')
    # except Exception as e:
    # print(f"An error occurred: {e}")
        
    print('len of gg_group', len(gg_group))


    gg_group_list = []
    gg_goal_group_list = []
    for i in range(len(gg_group)):
        gg_group_list.append(gg_group[i].transform(extrinsics))
        gg_goal_group_list.append(gg_goal_group[i].transform(extrinsics))


    return gg_group_list, gg_goal_group_list


def filter_pc(scene_pc, obs):
    rm = RobotModel(ARM_URDF_FULL)
    init_q = to_torch(np.concatenate((obs['robot0_joint_pos'], obs['robot0_gripper_qpos']), axis=0)[None]).float()
    # import pdb; pdb.set_trace()
    init_qpos = {k: init_q[:, i] for i, k in enumerate(ROBOT_JOINTS)}
    robot_pc, link_trans, link_rot, link_pc = rm.sample_surface_points_full(init_qpos, n_points_each_link=2**11, with_fk=True)
    robot_pc = robot_pc[0]
    state_pc = o3d.geometry.PointCloud()
    state_pc.points = o3d.utility.Vector3dVector(scene_pc)
    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_pc)

    # 使用 KDTree 查找与 robot_pc 重叠的点
    kd_tree = o3d.geometry.KDTreeFlann(state_pc)
    indices_to_remove = []
    for point in robot_pcd.points:
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)  # 设置合适的半径
        indices_to_remove.extend(idx)
    # 移除重复点
    state_pc = state_pc.select_by_index(indices_to_remove, invert=True)
    scene_pc_filter = torch.tensor(np.asarray(state_pc.points))

    return scene_pc_filter


def get_grasp_and_place_path(depth, image, intrinsic, extrinsics, task_description):

    scene_pc_cam, object_pc_cam, scene_pc, relative_translation_table, relative_rotation_table = sofar(image, depth, intrinsic, extrinsics, task_description)
    # import pdb; pdb.set_trace()
    
    gg_group, gg_goal_group = get_grasp_pose(task_description, object_pc_cam, scene_pc_cam, scene_pc, extrinsics, relative_translation_table, relative_rotation_table)
    
    return gg_group, gg_goal_group








@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self, json_data = None):
        self.json_data = json_data
        self.objs = [obj.replace(' ', '_') for obj in self.json_data['selected_obj_names']]
        self.number_obj = len(self.objs)
        self.init_obj_pos = self.json_data['init_obj_pos']
        self.quat_dict =  dict()
        self.goal_object = self.json_data['target_obj_name']
        self.xml_dict = [os.path.dirname(select) for select in self.json_data['selected_urdfs']]
            
        fixture_num_info = {
            "open6dor": 1,
        }

        
        objects_dict = Counter(self.objs)
        object_num_info = {
            **objects_dict,
            # "libero_mug_yellow": 1,
        }

        
        super().__init__(
            workspace_name="open6dor",  # define the scene base
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )
        
    def define_regions(self):
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            self.regions.update(
                self.get_region_dict(
                region_centroid_xy = self.init_obj_pos[id],
                region_name=self.objs[id].replace(' ', '_')+'_'+ str(objects_dict[self.objs[id]])+'_init_region',
                target_name=self.workspace_name,
                region_half_len=0.02,
                yaw_rotation = tuple(self.init_obj_pos[id][3:7]),
                goal_quat = self.quat_dict.get(self.objs[id], [0, 0, 0, 1]),
                xml = self.xml_dict[id],
                init_pos = self.init_obj_pos[id][:3],
                init_quat = self.init_obj_pos[id][3:7],
            )
            )
            objects_dict[self.objs[id]] +=1
            # print(self.quat_dict[self.objs[id]] if self.quat_dict[self.objs[id]] else [1,0,0,0])
            
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = []
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            states.append(
                ("On", self.objs[id]+'_'+ str(objects_dict[self.objs[id]]), "open6dor_"+self.objs[id]+'_'+ str(objects_dict[self.objs[id]])+'_init_region'
            )
            )
            objects_dict[self.objs[id]] +=1

        return states
    


import re

def extract_object_from_instruction(orientation, instruction, selected_obj, goal_objs):
    # 加入对冠词的忽略（如 "the", "a", "an"）
    found_objects = []
    if orientation == "between":
    # 匹配 between 后的两个物体名称
        pattern = rf"between (?:the|a|an)?\s*([a-zA-Z_\s]+?)\s*and (?:the|a|an)?\s*([a-zA-Z\s]+?)(?:\s|$)"
        match = re.search(pattern, instruction)
        if match:
            obj1 = match.group(1).strip()
            obj2 = match.group(2).strip()
            # 在 selected_obj_names 中查找两个物体
            
            for obj in selected_obj:
                if obj in obj1:
                    found_objects.append(obj)
                if obj in obj2:
                    found_objects.append(obj)
            return found_objects if found_objects else None
    elif orientation in ["behind"]:
        pattern = rf"{orientation} (?:the|a|an)?\s*([a-zA-Z_\s]+?)(?:\s|$)"
    elif orientation in ["center"]:
        return [obj for obj in selected_obj if obj != goal_objs]
    else:
        pattern = rf"{orientation} of (?:the|a|an)?\s*([a-zA-Z_\s]+?)(?:\s|$)"
    match = re.search(pattern, instruction)
    if match:
        # 提取并返回物体名称，去除可能的空格
        following_text = match.group(1).strip()
        for obj in selected_obj:
            if following_text in obj:
                
                found_objects.append(obj)
                return found_objects



def create_task_dict(bddl_path, task_json_path, video_path, final_positions):
    task_dict = {
        "bddl_path": bddl_path,
        "task_json_path": task_json_path,
        "video_path": video_path,
        "final_positions": final_positions,  # List of positions for 10 trials
    }
    return task_dict













def eval_libero(etection_model, sam_model, save_path, root, json_data, json_file, cfg: GenerateConfig):
    
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)

    # if the orientation is the center, we have to consider all objects
    print('json_file', json_file)
    register_task_info(
            bddl_name, # register the task 
            scene_name=scene_name,
            objects_of_interest=[],
            goal_states=[
                (orientation.title(), matching_word[i]+'_1', goal_objs+'_1') for i in range(len(matching_word))
            ],
            json_data = json_data,
        )
    


    
    bddl_file_names, failures = generate_bddl_from_task_info(folder=save_path, json_data=json_data) # generate the bddl_file

    # bddl_file_names = ["/data/workspace/LIBERO/openvla/task_refine_pos/behind/Place_the_apple_behind_the_cup_on_the_table._/KITCHEN_DEMO_SCENE_20240824-223840_no_interaction.bddl"]
    mp4_path = bddl_file_names[0].replace('bddl','mp4')
    

    local_log_filepath = os.path.join(save_path, "result.txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Initialize LIBERO environment and task description
    env_args = {
        "bddl_file_name": bddl_file_names[0],
        # "bddl_file_name": bddl_file_namesssss,
        "camera_heights": 1024,
        "camera_widths": 1024,
        "camera_depths": True,
        # "camera_names": ["frontview","agentview"],
        "controller": "JOINT_POSITION",
        # "control_freq": 5, 
        "controller_config_file": "/data/workspace/LIBERO/workspace/controller/joint.json",
        # "controller": "OSC_POSE",
        # "controller_config_file": "/data/workspace/LIBERO/workspace/controller/no_delta.json",
        # "ignore_done": True
    }
    task_description = instruction
    
    env = OffScreenRenderEnv(**env_args)
    intrinsic = get_camera_intrinsic_matrix(env.sim, 'frontview', camera_height=1024, camera_width=1024)
    extrinsic = get_camera_extrinsic_matrix(env.sim, 'frontview')
    # 旋转矩阵
    rotation_matrix = np.eye(3)  # 单位矩阵
    # rotation_matrix[0,0] = -1
    # 平移向量
    translation_vector = np.array([-0.15, 0, 0.912])

    # 构建齐次矩阵
    base_matrix = np.eye(4)  # 初始化为4x4单位矩阵
    base_matrix[:3, :3] = rotation_matrix  # 设置旋转矩阵
    base_matrix[:3, 3] = translation_vector  # 设置平移向量
    extrinsic_new =  np.linalg.inv(base_matrix) @ extrinsic


    # set the initial robot state:


#       import robosuite.utils.transform_utils as T
        # obs = env.reset()
        # print('reset_quat:', obs["robot0_eef_quat"])
        # rot = T.quat2axisangle(T.convert_quat(obs['robot0_eef_quat'], to = 'xyzw'))
        # print('rot:', rot)
        # # rot = np.array([3.14159 / 2, 0 ,0])
        # pos = obs["robot0_eef_pos"]
        # action = np.concatenate([pos, np.append(rot, 0)])
        # print('action:', action)
        # for i in range(200):
        #     obs, reward, done, info = env.step(action.tolist())
        # print('now_quat:', obs["robot0_eef_quat"])
    obj_pos_dict = dict()
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        assert cfg.num_trials_per_task==1
        
        grasp_path = None
        
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        obs = env.reset()
        # Setup
        t = 0
        replay_images = []
        while t < 100:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            # obs, _, _, _ = env.step([0, -0.61521699,  1.40194223, -0.43245613, -2.1516109 , -0.86708925, 0.57265033, -2.44411672])
                # import pdb; pdb.set_trace()
            # cur_qpos = obs["robot0_joint_pos"]
            # target_qpos = cur_qpos + np.array([0, 0, 0, 0, 0, 0, 0.1])
            # for _ in range(10):
            #     cur_qpos = obs["robot0_joint_pos"]
            #     obs, _, _, _ = env.step(np.concatenate([target_qpos - cur_qpos, np.zeros(1)]))
            obs, _, _, _ = env.step([0.1, 0, 0.1, 0, 0.1, 0, 0, 0])
            t += 1



        # waypoint_path = "/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_open6dor_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_open6dor_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_open6dor_ours_test4_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_rot_test6.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_rot_test4.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_pre0.3_n20_filter5_osc_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_osc_formatted.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours_osc_formatted.npy" #"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints_ours1_formatted.npy"#"/home/yufei/Projects/GSNet/output/traj/interpolated_waypoints.npy"
        # actions = np.load(waypoint_path, allow_pickle=True)
        print("task_description:", task_description)
        print(f"Starting episode {task_episodes+1}...")
        log_file.write(f"Starting episode {task_episodes+1}...\n")
        # for t in range(10):
        #     obs, reward, done, info = env.step([-0.2, 0, 0, 0, 0, 0, 0])
        # obs, reward, done, info = env.step([0, 1, 0, 0, 0, 0, 0])

        # image = image[int(0.2*H):int(0.8*H),:]
        # depth = depth[int(0.2*H):int(0.8*H),:]
        # grasp_path, place_path = get_grasp_and_place_path(depth, intrinsic, extrinsic, task_description)
        # i = 0
        # while i<200:
        #     i+=1
        #     # obs, _, _, _ = env.step([0, -0.61521699,  1.40194223, -0.43245613, -2.1516109 , -0.86708925,
        #     #     0.57265033, -2.44411672])
        #     obs, _, _, _ = env.step(np.array([0, -0.4, 0.4, 0, 0, 0, 0]))
        # exit(0)
        image = obs["frontview_image"][::-1]
        depth = obs['frontview_depth'][::-1]
        H = image.shape[0]
        
        
        near = 0.01183098675640314
        far = 591.5493097230725
        depth = near / (1 - depth * (1 - near / far))
        
        # from SpatialAgent.depth.utils import depth2pcd
        # intrinsic_matrix = intrinsic
        # image2 = Image.fromarray(image)
        # fx = intrinsic_matrix[0, 0]
        # fy = intrinsic_matrix[1, 1]
        # cx = intrinsic_matrix[0, 2]
        # cy = intrinsic_matrix[1, 2]
        # intrinsic = [fx, fy, cx, cy]
        # pcd_camera, pcd_base = depth2pcd(depth, intrinsic, extrinsic_new)
        # pcd_base = pcd_base.reshape(-1,3)
        # visualize_plotly(pcd_base, 200 , None, extrinsic_new, gg_glob=True)
        
        
        scene_pc_cam, object_pc_cam, scene_pc, relative_translation_table, relative_rotation_table = sofar(etection_model, sam_model, image, depth, intrinsic, extrinsic_new, task_description)
        
        
        print(colored('Grasp Pose Inference', 'green'))
        gg_group, gg_goal_group = get_grasp_pose(task_description, object_pc_cam, scene_pc_cam, scene_pc, extrinsic_new, relative_translation_table, relative_rotation_table)
        if gg_group is None:
            task_dict = {
                "bddl_path":  [],
                "task_json_path": [],
                "video_path": [],
                "final_positions": [],  # List of positions for 10 trials
            }
            return task_dict
   
        # import pdb; pdb.set_trace()
        
        # filter the scene pcd
        scene_pc_filter = filter_pc(scene_pc, obs)
        
        robot_urdf = ARM_URDF_FULL
        cfg_robot = config.DotDict(
            urdf=robot_urdf,
            pc = scene_pc_filter,
        )
        
        print(colored('Grasp Inference Completed', 'green'))
        
        try:
            for i in range(len(gg_group)):
            # get the grasp pose and pose pose
                images = []
                obs = env.reset()
                gg = gg_group[i]  
                gg_goal = gg_goal_group[i]
                # from graspnetAPI import GraspGroup
                # visuallize_list = GraspGroup()
                # visuallize_list.add(gg)
                # visuallize_list.add(gg_goal)
                # visualize_plotly(scene_pc, 200 ,visuallize_list, extrinsic, gg_glob=True)
                print(colored("\nStar Planning Grasp Phase", 'green'))
                init = np.array(obs['robot0_joint_pos']) 
                
                robot_state = np.array(obs['robot0_joint_pos'])
                goal = mat2quat(gg.rotation_matrix)
                goal = np.concatenate([gg.translation, goal])
                ik = IK(robot='franka')
                # goal = ik.ik(gg.translation, gg.rotation_matrix, joints=ik.robot_to_arm_joints(init))
                # goal = np.concatenate([ik.ik(ee_pose_t, ee_pose_r, joints=ik.robot_to_arm_joints(init)), np.array([0.02, 0.02])])
                align = np.array([
                    [0,0,1],
                    [0,-1,0],
                    [1,0,0],
                ])
                rot = gg.rotation_matrix @ align
                trans = gg.translation

                goal = ik.ik(trans, rot, joints=ik.robot_to_arm_joints(init))
                if goal is None:
                    print(colored("Grasp Path IK No Solution", 'red'))
                    continue
                
                # rm = RobotModel(ARM_URDF)
                # goal_qpos = to_torch(np.concatenate([goal, np.zeros(4)])[None]).float()
                # goal_qpos = {k: goal_qpos[:, i] for i, k in enumerate(ROBOT_JOINTS)}
                # rm.forward_kinematics(goal_qpos)[1]['link_gripper']
                vis = Vis(ARM_URDF_FULL)
                # R.from_matrix().to_quat()
                # vis.show(vis.robot_plotly(qpos=np.concatenate([goal, np.zeros(2)])))
                for _ in range(3):
                    planner = Planner(cfg_robot, fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])
                    res_grasp, grasp_path = planner.plan(robot_state[:7], goal, fix_joints_value = {'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2}, interpolate_num=30)
                    # res, path = planner.plan(robot_state[:7], goal, interpolate_num=50, fix_joints_value={'joint_head_pan': robot_state[9], 'joint_head_tilt': robot_state[10]})
                    # vis.show(vis.robot_plotly(qpos=np.concatenate([goal, np.zeros(2)]))+vis.pc_plotly(cfg_robot['pc'][::10]))
                    if res_grasp: # and isinstance(grasp_path, np.ndarray)
                        break
                if grasp_path is None or res_grasp==False: # or not isinstance(grasp_path, np.ndarray)
                    continue
                print(colored('\nGrasp Path Completed', 'green'))
                # vis.traj_plotly(grasp_path)
                # place_init_qpos = obs['agent']['qpos'][:7]
                print('\nPlace Path Starting')
                place_init_qpos = goal
                rot2 = gg_goal.rotation_matrix @ align
                trans_2 = gg_goal.translation
                # try:
                # for _ in range(3):
                place_goal_qpos = ik.ik(trans_2, rot2, joints=ik.robot_to_arm_joints(init))
                if place_goal_qpos is None:
                    print(colored("Place Path IK No Solution", 'red'))
                    continue

                
                for _ in range(3):
                    planner = Planner(cfg_robot, planner='AITstar', fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])
                    # res_place, place_path = planner.plan(place_init_qpos[:7], place_goal_qpos, interpolate_num=50, fix_joints_value={'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2})
                    res_place, place_path = planner.plan(place_init_qpos[:7], place_goal_qpos, fix_joints_value = {'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2},  interpolate_num=30)
                    # res, path = planner.plan(robot_state[:7], goal, interpolate_num=50, fix_joints_value={'joint_head_pan': robot_state[9], 'joint_head_tilt': robot_state[10]})
                    if res_place: #  and isinstance(place_path, np.ndarray)
                        break
                if place_path is None or res_grasp==False: #or not isinstance(place_path, np.ndarray)
                    continue

                print('\nPlace Path Completed')


                def upper_step(env, obs, target_qpos, images, grasp=True, step_num=5):
                    # print('init:', np.linalg.norm(target_qpos[:7]-obs['robot0_joint_pos']))
                    for _ in range(step_num):
                        obs, reward, done, info  = env.step(np.concatenate([target_qpos[:7] - obs['robot0_joint_pos'], np.zeros(1)]))
                        # print(np.concatenate([target_qpos[:7] - obs['robot0_joint_pos'], np.zeros(1)])[-1])
                        img = obs['agentview_image'][::-1]
                        images.append(img)  
                    # print('final:', np.linalg.norm(target_qpos[:7] - obs['robot0_joint_pos']))
                    
                    return obs, images
                

                gripper = [0,0,0,0,0,0,0, -10]
                for i in range(2):
                    obs, _, _,_ = env.step(gripper)
                    images.append(obs['agentview_image'][::-1])   
                          
                for index in range(len(grasp_path)):
                    obs, images = upper_step(env, obs, grasp_path[index][:8], images)
                    # obs, reward, done, info  = env.step(grasp_path[index][:8])  
                    # while np.linalg.norm(obs['robot0_joint_pos']  - grasp_path[index][:7]) > 0.1:
                        # obs, reward, done, info  = env.step(grasp_path[index][:8])  
                
                print('Grasp Path Execution Completed')
                
                
                gripper = [0,0,0,0,0,0,0, 10]
                for i in range(2):
                    obs, _, _,_ = env.step(gripper)
                    images.append(obs['agentview_image'][::-1])
                
                
                for index in range(len(place_path)):
                    # obs, reward, done, info  = env.step(place_path[index][:8])
                    obs, images = upper_step(env, obs, place_path[index][:8], images)
                    # obs, reward, done, info  = env.step(grasp_path[index][:8])  
                    # while np.linalg.norm(obs['robot0_joint_pos']  - grasp_path[index][:7]) > 0.1:
                        # obs, reward, done, info  = env.step(grasp_path[index][:8])   
                print('Place Path Execution Completed')
                
                gripper = [0,0,0,0,0,0,0, -10]
                for i in range(2):
                    obs, _, _,_ = env.step(gripper)
                    images.append(obs['agentview_image'][::-1])
                save_rollout_video(
                images, total_episodes, success=False, task_description=task_description, log_file=log_file, mp4_path=mp4_path
            )
                if done:
                    break
                        # image = Image.fromarray(image)
                        # image.save("/data/workspace/SimplerEnv/test.png")             
        except Exception as e:
            print(f"An error occurred: {e}")
            if isinstance(grasp_path, np.ndarray):
                print('\nPlace Path Completed')
                grasp_path[:, 7] = -0.00285961
                grasp_path[:, 8] = 0.7851361
                
                num_copies = 5
                repeated_elements = np.tile(grasp_path[-1], (num_copies, 1))
                for index in range(num_copies):
                    repeated_elements[index, -2:] = [index/5, index/5]
                grasp_path = np.vstack([grasp_path, repeated_elements])
                for index in range(len(grasp_path)):
                    obs, reward, done, info  = env.step(grasp_path[index][:8])   
                image = img = obs['frontview_image'][::-1]
                images.append(image)    
            elif isinstance(grasp_path, list):
                print('\nPlace Path Failed')
                obs, reward, done, info  = env.step(np.array(grasp_path[0][:8]))
            else:
                obs, reward, done, info  = env.step(np.zeros(8))
        task_info = None
        return task_info
        
        
        
        
        
        

        # while t < cfg.max_steps + cfg.num_steps_wait:
        #     # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
        #     # and we need to wait for them to fall
        #     if t < cfg.num_steps_wait:
        #         obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, 0, 0])
        #         # import pdb; pdb.set_trace()
        #         t+=1
  
        #     # Get image (TODO: here we should adjust the extrinsic matrix of camera)
        #     img = obs["frontview_image"][::-1]
        #     depth = obs['frontview_depth'][::-1]
        #     # img1 = obs["agentview_image"][::-1]
        #     # depth1 = obs['agent1view_depth'][::-1]

        #     i = (t-cfg.num_steps_wait) // 4
        #     action = actions[i]
        #     if action[6] != 0:
        #         action[6] = 8
     
       

        #     # Save preprocessed image for replay video
        #     replay_images.append(img)
       
        #     obs, reward, done, info = env.step(np.array(action))


            # img = Image.fromarray(img)
            # img1 = Image.fromarray(img1)
            # img1.save(f'/data/benchmark/test/rgb_open6dor_ours_agent_test5.png')
            # img.save(f'/data/benchmark/test/rgb_open6dor_ours_test6.png')
            # import pdb; pdb.set_trace()
            # # store depth as npy:
            # np.save(f'/data/benchmark/test/depth_open6dor_ours_test6.npy', depth)
            # np.save(f'/data/benchmark/test/depth_open6dor_ours_agent_test5.npy', depth1)
            # # # store rgb as npy:
            # np.save(f'/data/benchmark/test/rgb_open6dor_ours_test6.npy', img)  
            # img.save(os.path.join(save_path, f"rgb_front.png"))
            # np.save(os.path.join(save_path, f"depth_front.npy"), depth)
            # np.save(os.path.join(save_path, f"rgb_front.npy"), img)

        # Prepare observations dict
        # Note: OpenVLA does not take proprio state as input
        # TODO: adapt our model
        # action = ourmodel(img, depth, task_description, cfg, processor)
        # Query model to get action
        # action = get_action(
        #     cfg,
        #     model,
        #     observation,
        #     task_description,
        #     processor=processor,
        # )
        # action = np.array([0,0,0,0,0,0,0])
        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
        # action = normalize_gripper_action(action, binarize=True)

        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
        # if cfg.model_family == "openvla":
        #     action = invert_gripper_action(action)
        # print(action)
        # Execute action in environment
        # obs, reward, done, info = env.step(action.tolist())


        #     if done:
        #         task_successes += 1
        #         total_successes += 1
        #         break
        #     t += 1
            
        # # log_file.write(f"Inputs: stored\n")         
        # log_file.write(f"Position: {done}\n")           
        # task_episodes += 1
        # total_episodes += 1
        # for obj in selected_obj:
        #     obj = obj.lower().replace(' ', '_')
        #     obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
        #     obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
        #     # log_file.write(obj: obs[obj+"_1"] \n")
        # # Save a replay video of the episode
        # # save_rollout_video(
        # #     replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path
        # # )
        
        # # Log current results
        # print(f"Success: {done}")
        # print(f"# episodes completed so far: {total_episodes}")
        # print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        # log_file.write(f"Success: {done}\n")
    #     log_file.write(f"# episodes completed so far: {total_episodes}\n")
    #     log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
    #     log_file.flush()
    # save_rollout_video(
    #         replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path
    #     )
    # for obj in selected_obj:
    #     obj = obj.lower().replace(' ', '_')
    #     obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
    #     obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
    # print("save_rollout_video success", mp4_path)    
    # log_file.write(f"save_rollout_video success: {mp4_path}\n")
    # task_info = create_task_dict(bddl_file_names[0], json_file, mp4_path,  obj_pos_dict)
    # # # Log final results
    # print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
    # print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    # log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
    # log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
    # log_file.flush()
    # # if cfg.use_wandb:
    # #     wandb.log(
    # #         {
    # #             f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
    # #             f"num_episodes/{task_description}": task_episodes,
    # #         }
    # #     )

    # # # Save local log file
    # log_file.close()

    # # Push total metrics and local log file to wandb
    # if cfg.use_wandb:
    #     wandb.log(
    #         {
    #             "success_rate/total": float(total_successes) / float(total_episodes),
    #             "num_episodes/total": total_episodes,
    #         }
    #     )
    #     wandb.save(local_log_filepath)
    # import pdb; pdb.set_trace()
    # return task_info



if __name__ == "__main__":
    # add args
    args = argparse.ArgumentParser()
    args.add_argument("--category", type=str, default="behind") # model path
    
    args = args.parse_args()
    category = args.category

    # load model
    detection_model = florence.get_model()
    sam_model = sam.get_model()


    cfg = GenerateConfig   
    # Load model
    model = ""
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # [OpenVLA] Get task dictionary
    # root_dir = "/data/workspace/LIBERO/spatial/task_refine_pos"#
    # root_dir = "/data/datasets/open6dor/task_refine_pos"## # task_refine_pos, task_refine_rot, task_refine_rot_only
    root_dir = f"/data/workspace/LIBERO/spatial/task_refine_pos"
    grasp_track_name = "task_refine_pos"
    output_root = "./execution_exp_1113_cont"
    # output_file = os.path.join(output_root, f"{grasp_track_name}/open6dor_exec_dict.json") #!!!! do not overwrite
    output_file = "./open6dor_exec_dict.json"
    # 初始化或加载已存在的 task_dict
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            task_dict = json.load(f)
    else:
        os.makedirs(os.path.join(output_root, grasp_track_name), exist_ok=True)
        task_dict = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            
            if file.endswith('refine_ghost.json'):
                json_file = os.path.join(root, file)
                task_name = os.path.basename(root)
                # 检查任务是否已经完成
                if task_name in task_dict:
                    print(f"Skipping {task_name} as it is already processed.")
                    continue
                with open(json_file, 'r') as f:
                    file_data = json.load(f)

                    parts = json_file.split("/")

                    save_path = os.path.join(output_root, f"{grasp_track_name}","/".join(parts[6:9]))
                    os.makedirs(save_path, exist_ok=True)
        
                    task_dict[os.path.basename(root)] = eval_libero(detection_model, sam_model, save_path, root, file_data, json_file, cfg)#, model)
                    # 将 task_dict 保存到 JSON 文件
                    with open(output_file, 'w') as f: 
                        json.dump(task_dict, f, indent=4)
                        
                        
                        
