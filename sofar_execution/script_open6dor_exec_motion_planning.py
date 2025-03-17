
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# import debugpy
# debugpy.listen(('0.0.0.0', 5683))
# print('Waiting for debugger attach')
# debugpy.wait_for_client()


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

from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
import argparse
import re
from libero.libero.utils.bddl_generation_utils import *
sys.path.append(BASE_DIR+'/SoFar')

# from GSNet.gsnet import grasp_inference_feifei
from GSNet.gsnet_simpler import grasp_inference, visualize_plotly
from SoFar.segmentation import florence, sam, grounding_dino

from SoFar.serve.pointso import get_model as get_pointofm_model
# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")






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



# from SoFar.depth.utils import transform_point_cloud_nohw, inverse_transform_point_cloud
# from SoFar_o.open6dor import sofar
from sofar_execution_libero import sofar_libero
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
    # rollout_dir = f"/mnt/afs/zhangwenyao/LIBERO/rollouts/{DATE}"
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


def get_grasp_pose(task_description, intrinsic, object_mask, object_pc_cam, scene_pc_cam, scene_pc, extrinsics, relative_translation_table, relative_rotation_table, graspness_threshold):
    # try:
    gg_group, gg_goal_group = grasp_inference(task_description,  intrinsic, object_mask, object_pc_cam, scene_pc_cam, scene_pc, extrinsics, relative_translation_table, relative_rotation_table, graspness_threshold)
    if gg_group is None:
        return None, None
    print(colored('Grasp Inference Completed','green'))
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
        [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)  # set the radius to your desired value
        indices_to_remove.extend(idx)
    # 移除重复点
    state_pc = state_pc.select_by_index(indices_to_remove, invert=True)
    scene_pc_filter = torch.tensor(np.asarray(state_pc.points))

    return scene_pc_filter










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
                # init_pos = self.init_obj_pos[id][:3],
                init_pos = [0.2 + self.init_obj_pos[id][0] / 2.5, self.init_obj_pos[id][1] / 2.5, self.init_obj_pos[id][2] + 0.03],
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





def upper_step(env, obs, target_qpos, images, grasp=True, step_num=10):
    # print('init:', np.linalg.norm(target_qpos[:7]-obs['robot0_joint_pos']))
    for _ in range(step_num):
        obs, reward, done, info  = env.step(np.concatenate([target_qpos[:7] - obs['robot0_joint_pos'], np.zeros(1)]))
        # print(np.concatenate([target_qpos[:7] - obs['robot0_joint_pos'], np.zeros(1)])[-1])
        img = obs['frontview_image'][::-1]
        images.append(img)  
    return obs, images







def eval_libero(detection_model, sam_model, orientation_model, save_path, root, json_data, json_file, cfg: GenerateConfig):
    
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    # matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)
    matching_word = [obj.lower().replace(' ', '_') for obj in json_data['selected_obj_names']] # .lower().replace(' ', '_')
    # if the orientation is the center, we have to consider all objects
    print('json_file', json_file)
    register_task_info(
            bddl_name, # register the task 
            scene_name=scene_name,
            objects_of_interest=[],
            goal_states=[
                ("rotation", matching_word[i]+'_1', goal_objs+'_1') for i in range(len(matching_word))
            ],
            json_data = json_data,
        )
    


    
    bddl_file_names, failures = generate_bddl_from_task_info(folder=save_path, json_data=json_data) # generate the bddl_file
    mp4_path = bddl_file_names[0].replace('bddl','mp4')
    image_path = bddl_file_names[0].replace('bddl','png')
    
    print(image_path)
    local_log_filepath = os.path.join(save_path, "result.txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")
    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Initialize LIBERO environment and task description
    env_args = {
        "bddl_file_name": bddl_file_names[0],
        "camera_heights": 1024,
        "camera_widths": 1024,
        "camera_depths": True,
        "controller": "JOINT_POSITION",
        "controller_config_file": "/mnt/afs/zhangwenyao/LIBERO/plan/workspace/controller/joint.json",
    }
    task_description = instruction.lower()
    
    env = OffScreenRenderEnv(**env_args)
    obj_pos_dict = dict()
    
    # get intrinsic and extrinsic
    intrinsic = get_camera_intrinsic_matrix(env.sim, 'agentview', camera_height=1024, camera_width=1024)
    extrinsic = get_camera_extrinsic_matrix(env.sim, 'agentview')
    rotation_matrix = np.eye(3) 
    translation_vector = np.array([-0.15, 0, 0.912]) # the position of robot base
    base_matrix = np.eye(4)  
    base_matrix[:3, :3] = rotation_matrix 
    base_matrix[:3, 3] = translation_vector  
    extrinsic_new =  np.linalg.inv(base_matrix) @ extrinsic


    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
        assert cfg.num_trials_per_task==1
        
        grasp_path = None
        
        print(f"\nTask: {task_description}")
        log_file.write(f"\nTask: {task_description}\n")

        # Reset environment
        obs = env.reset()
        init = np.array(obs['robot0_joint_pos']) 
    
        # Setup
        
        print(f"Starting episode {task_episodes+1}...")
        log_file.write(f"Starting episode {task_episodes+1}...\n")
        images = []
        for _ in range(20):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, 0, 0])
            images.append(obs["frontview_image"][::-1])
        for _ in range(15):
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            obs, _, _, _ = env.step([0, 0, 0, 0.5, 0, 0, 0, 0])
            images.append(obs["frontview_image"][::-1])
            
        Image.fromarray(obs['agentview_image'][::-1]).save(image_path)
            



        image = obs["agentview_image"][::-1]
        depth = obs['agentview_depth'][::-1]
        near = 0.01183098675640314
        far = 591.5493097230725
        depth = near / (1 - depth * (1 - near / far))
            
        scene_pc_cam, scene_pc,  object_pc_cam, object_pc_base, object_mask, relative_translation_table, relative_rotation_table = sofar_libero(detection_model, sam_model, orientation_model, image, depth, intrinsic, extrinsic_new, task_description)
        
        
        
        for _ in range(15):
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            obs, _, _, _ = env.step([0, 0, 0, -0.5, 0, 0, 0, 0])
            images.append(obs["frontview_image"][::-1])

        
        print(colored('Grasp Pose Inference', 'green'))
       
        vis = Vis(ARM_URDF_FULL)
        
        # filter the scene pcd
        scene_pc_filter = filter_pc(scene_pc, obs)
        
        robot_urdf = ARM_URDF_FULL
        cfg_robot = config.DotDict(
            urdf=robot_urdf,
            # pc = scene_pc_filter,
            pc = torch.tensor([[0,0,0]]),
            # pc =  torch.tensor(object_pc_base),
        )
       
        graspness_threshold = 0.01
        grasp_path = None
        mid_point = None
        place_path = None
        
        for _ in range(3):
            gg_group, gg_goal_group = get_grasp_pose(task_description, intrinsic, object_mask, object_pc_cam, scene_pc_cam, scene_pc, extrinsic_new, relative_translation_table, relative_rotation_table, graspness_threshold)
            if gg_group is None:
                print(colored('Grasp Inference Failed', 'red'))
                for obj in selected_obj:
                    obj = obj.lower().replace(' ', '_')
                    obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
                    obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
                task_info = create_task_dict(bddl_file_names[0], json_file, mp4_path,  obj_pos_dict)
                save_rollout_video(
                images, total_episodes, success=False, task_description=task_description, log_file=log_file, mp4_path=mp4_path
            )      
                return task_info
            
   
            
            for i in range(len(gg_group)):
                # get the grasp pose and pose pose
                try:
                    gg = gg_group[i]  
                    gg_goal = gg_goal_group[i]
                    print(colored("\nStar Planning Grasp Phase", 'green'))
                    
                    
                    robot_state = np.array(obs['robot0_joint_pos'])
                    goal = mat2quat(gg.rotation_matrix)
                    goal = np.concatenate([gg.translation, goal])
                    ik = IK(robot='franka')
                
                    align = np.array([
                        [0,0,1],
                        [0,-1,0],
                        [1,0,0],
                    ])
                    rot = gg.rotation_matrix @ align
                    trans = gg.translation

                    goal = ik.ik(trans, rot, joints=ik.robot_to_arm_joints(init))
                    if goal is None:
                        for ik_iter in range(6):
                            goal = ik.ik(trans, rot, joints=ik.robot_to_arm_joints(init))
                            if not goal is None:
                                break
                    if goal is None:
                        print(colored("Grasp Path IK No Solution", 'red'))
                        continue
                    
                    mid_init_qpos = goal
                    mid_point = ik.ik(trans+[0, 0, 0.15], rot,  joints=ik.robot_to_arm_joints(init))

                    place_init_qpos = mid_point
                    rot2 = gg_goal.rotation_matrix @ align
                    trans_2 = gg_goal.translation + [0, 0, 0.1]
                    place_goal_qpos = ik.ik(trans_2, rot2, joints=ik.robot_to_arm_joints(init))
                    if place_goal_qpos is None:
                        for ik_iter in range(6):
                            place_goal_qpos = ik.ik(trans_2, rot2, joints=ik.robot_to_arm_joints(init))
                            if not place_goal_qpos is None:
                                break
                    if place_goal_qpos is None:
                        print(colored("Place Path IK No Solution", 'red'))
                        continue    
            

                    for _ in range(2):
                        planner = Planner(cfg_robot, fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])
                        res_grasp, grasp_path = planner.plan(robot_state[:7], goal, fix_joints_value = {'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2}, interpolate_num=30)

                        if res_grasp:
                            break
                    if grasp_path is None or res_grasp==False: # or not isinstance(grasp_path, np.ndarray)
                        continue
                    print(colored('\nGrasp Path Completed', 'green'))

                    
                    # mid_path construction 
                    print(colored('\n Mid Path Starting', 'green'))
                    
                    for _ in range(2):
                        planner = Planner(cfg_robot, planner='AITstar', fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])
                        res_mid, mid_path = planner.plan(mid_init_qpos[:7], mid_point[:7], fix_joints_value = {'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2},  interpolate_num=30)
                        if res_mid: #  and isinstance(place_path, np.ndarray)
                            break
                    if mid_path is None or res_mid==False: #or not isinstance(place_path, np.ndarray)
                        continue           
                    mid_path[:,8] = 0
                    print(colored('\nPlace Path Starting', 'green'))
                    


                    
                    for _ in range(2):
                        planner = Planner(cfg_robot, planner='AITstar', fix_joints=['panda_finger_joint1', 'panda_finger_joint2'])
                        res_place, place_path = planner.plan(place_init_qpos[:7], place_goal_qpos, fix_joints_value = {'panda_finger_joint1': 0.04 - gg.width / 2, 'panda_finger_joint2': 0.04 - gg.width / 2},  interpolate_num=30)
                        if res_place: #  and isinstance(place_path, np.ndarray)
                            break
                    if place_path is None or res_place==False: #or not isinstance(place_path, np.ndarray)
                        continue

                    print('\nPlace Path Completed')



                    

                    gripper = [0,0,0,0,0,0,0, -10]
                    for i in range(10):
                        obs, _, _,_ = env.step(gripper)
                        images.append(obs['frontview_image'][::-1])   
                            
                    for index in range(len(grasp_path)):
                        obs, images = upper_step(env, obs, grasp_path[index][:8], images)
                    print('Grasp Path Execution Completed')

                    gripper = [0,0,0,0,0,0,0,10]
                    gripper_qpose = obs['robot0_gripper_qpos']
                    last_gripper_qpose = 10
                    for i in range(20):
                        obs, _, done, _ = env.step(gripper)
                        images.append(obs['frontview_image'][::-1])  
                    
                    
                    for index in range(len(mid_path)):
                        obs, images = upper_step(env, obs, mid_path[index][:8], images) 
                    print('Mid Path Execution Completed')   

                    
                    for index in range(len(place_path)):
                        obs, images = upper_step(env, obs, place_path[index][:8], images)
                        # while np.linalg.norm(obs['robot0_joint_pos']  - grasp_path[index][:7]) > 0.1:
                    print('Place Path Execution Completed')
                    
                    gripper = [0,0,0,0,0,0,0, -10]
                    for i in range(20):
                        obs, _, _,_ = env.step(gripper)
                        images.append(obs['frontview_image'][::-1])

                    break
            
            
                except Exception as e:
                    print(f"An error occurred: {e}")
                    continue
            if place_path is not None:
                break
            else:
                graspness_threshold /= 2
            
        save_rollout_video(
                    images, total_episodes, success=False, task_description=task_description, log_file=log_file, mp4_path=mp4_path
                )            
        for obj in selected_obj:
            obj = obj.lower().replace(' ', '_')
            obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
            obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
        task_info = create_task_dict(bddl_file_names[0], json_file, mp4_path,  obj_pos_dict)
        return task_info
        
        
        
        
        


if __name__ == "__main__":
    # add args
    parser = argparse.ArgumentParser("Task Refine Rot Configuration")
    parser.add_argument("--category", type=str, default="behind")
    parser.add_argument('--root_dir', type=str, default="/mnt/afs/zhangwenyao/LIBERO/datasets/task/task_refine_pos/",
                        help='Root directory for the task refine rot dataset')
    parser.add_argument('--grasp_track_name', type=str, default="task_refine_pos",
                        help='Name of the grasp track')
    parser.add_argument('--output_root', type=str, default="./execution_exp_0313_cont_rot",
                        help='Root directory for output files')
    parser.add_argument('--output_file', type=str, default="./sofar_output/open6dor_exec_dict_rot_0313.json",
                        help='Path to the output JSON file')
    parser.add_argument('--list_file_path', type=str, default="/mnt/afs/zhangwenyao/LIBERO/datasets/open6dor_list.json",
                        help='Path to the list file')
    
    
    args = parser.parse_args()
    category = args.category
    root_dir = args.root_dir
    grasp_track_name = args.grasp_track_name
    output_root = args.output_root
    output_file = args.output_file
    list_file_path = args.list_file_path

    # detection_model = grounding_dino.get_model()
    detection_model = florence.get_model()
    sam_model = sam.get_model()
    orientation_model = get_pointofm_model()


    cfg = GenerateConfig   
    # Load model
    model = ""

    with open(list_file_path, 'r') as f: 
        valid_key = json.load(f)
    # load task dict
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            task_dict = json.load(f)
    else:
        os.makedirs(os.path.join(output_root, grasp_track_name), exist_ok=True)
        task_dict = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('refine_ghost.json') and os.path.basename(root) in valid_key:
                json_file = os.path.join(root, file)
                task_name = os.path.basename(root)
                # test if the task is already processed
                if task_name in task_dict:
                    print(f"Skipping {task_name} as it is already processed.")
                    continue
                with open(json_file, 'r') as f:
                    file_data = json.load(f)

                    parts = json_file.split("/")

                    save_path = os.path.join(output_root, f"{grasp_track_name}","/".join(parts[5:9]))
                    os.makedirs(save_path, exist_ok=True)
                    # try:
                    task_dict[os.path.basename(root)] = eval_libero(detection_model, sam_model, orientation_model, save_path, root, file_data, json_file, cfg)#, model)
                    # except Exception as e:
                    #     print(e)
                    #     print(f"Error processing {task_name}")
                    #     continue
                    # exit()
                    # save task_dict to json file
                    with open(output_file, 'w') as f:
                        json.dump(task_dict, f, indent=4)
                        
                        
                        

