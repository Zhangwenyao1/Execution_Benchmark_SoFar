import sys
import os
sys.path.append('/data/workspace/LIBERO')
import re
import numpy as np

import os 
from collections import Counter
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


import re
from libero.libero.utils.bddl_generation_utils import *

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
import json
# absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
# 

# import debugpy
# debugpy.listen(5678)
# print(f'waiting for debugger to attach...')
# debugpy.wait_for_client()


from typing import Optional, Union
from dataclasses import dataclass


import time
import imageio
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, save_path = None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"/data/workspace/LIBERO/rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    # mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    # mp4_path = f"{rollout_dir}/{result}"
    mp4_path = save_path
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

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

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

        # states = [
        #     ("On", "libero_mug_1", "libero_mug_1_init_region"),
        #     ("On", "libero_mug_yellow_1", "libero_mug_yellow_1_init_region"),
        #     ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        # ]
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



def main(root, json_data, json_file, cfg):
    # kitchen_scene_1
    
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)

    # if the orientation is the center, we have to consider all objects
    print('json_file', json_file)
    
    # save_dir = os.path.dirname(json_file)
    
    
    register_task_info(
            bddl_name, # register the task 
            scene_name=scene_name,
            objects_of_interest=[],
            goal_states=[
                (orientation.title(), matching_word[i]+'_1', goal_objs+'_1') for i in range(len(matching_word))
            ],
            json_data = json_data,
        )
    
    # bddl_folder = os.path.join("/data/workspace/LIBERO/spatial/bddl_refine_pos")

    # if not os.path.exists(bddl_folder):
    #     os.makedirs(bddl_folder)
    bddl_file_names, failures = generate_bddl_from_task_info(folder=save_dir, json_data=json_data) # generate the bddl_file

    # with open(bddl_file_names[0], "r") as f:
    #     print(f.read())
    result = os.path.split(bddl_file_names[0])[-1].replace('bddl','mp4')
    # 获取最后一部分
    # print(result)
    # exit(0)

    # bddl_file_namesssss = '/data/workspace/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl'

    env_args = {
        "bddl_file_name": bddl_file_names[0],
        # "bddl_file_name": bddl_file_namesssss,
        "camera_heights": 256,
        "camera_widths": 256
    }
    language = "Place the knife behind the box on the table."

    env = OffScreenRenderEnv(**env_args)

    # obs = env.reset()
    # set the seed , the default seed is 0
    obs = env.reset()

    # exit(0)
    env.seed(0)
    replay_images = []
    t = 0
    max_steps = 100

    task_episodes, task_successes = 0, 0
    total_episodes = 0
    print("task_description", language)
    print(f"Starting episode {task_episodes+1}...")
    
    

    x_start, x_end = 0.452, 0.637888491153717
    y_start, y_end = 0, 0.11751404404640198
    z_start, z_end = 1.173, 0.9103142476081848
    
    x_start, x_end = 0.452*100, 0.637888491153717*100
    y_start, y_end = 0*100, 0.31751404404640198*100
    z_start, z_end = 1.173*100, 0.9103142476081848*100

    # 生成100个点的轨迹
    x_traj = np.linspace(x_start, x_end, max_steps)
    y_traj = np.linspace(y_start, y_end, max_steps)
    z_traj = np.linspace(z_start, z_end, max_steps)
    trajectory = np.vstack((x_traj, y_traj, z_traj)).T
    relative_trajectory = np.zeros_like(trajectory)

    # 初始点的相对位置为零
    relative_trajectory[0] = [0, 0, 0]

    # 计算相对位置：每一行减去前一行
    relative_trajectory[1:] = trajectory[1:] - trajectory[:-1]
    
    total_successes = 0
    index = 0
    while t < max_steps:
        try:
            
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            # if t < 5:
            #     obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
            #     t += 1
            #     continue
            # Prepare observations dict
            # Note: OpenVLA does not take proprio state as input
            # img = Image.fromarray(obs["agentview_image"][::-1])
            # observation = {
            #     "full_image": img,
            #     "state": np.concatenate(
            #         (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            #     ),
            # }
            # img.save("/data/workspace/LIBERO/agentview_image.png")
            # Query model to get action
            # if t < cfg.num_steps_wait:
            #     obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
            #     img = obs["agentview_image"][::-1]
            #     replay_images.append(img)

            #     t += 1
            #     continue
            # action = np.array([[0, 0, 0, 0, 0, 0, -1]])
                # print(obs.keys())
            # print('knife_1_pos', obs['knife_1_pos'])
            # print('box_1_pos', obs['box_1_pos'])
            # print('hammer_1_pos', obs['hammer_1_pos'])
            # print('watch_1_pos', obs['watch_1_pos'])
            img = obs["agentview_image"][::-1]
            replay_images.append(img)
            # action = np.array(relative_trajectory), np.array([0,0,0]), np.array([1.0])
            # if t % 20 == 0:
            #     action = np.hstack((np.array([1,0,0]), np.array([0,0,0]), np.array([0.0]))).reshape(1, 7)
            #     index += 1
            #     print(action)
            # Execute action in environment
            action = np.hstack((np.array(relative_trajectory[t]), np.array([0,0,0]), np.array([-1.0]))).reshape(1, 7)
            # if t<200:
            #     action = np.hstack((np.array([0, 0,-0.1]), np.array([0,0,0]), np.array([0.0]))).reshape(1, 7)
            # else: 
            #     action = np.hstack((np.array([0.1,0, 0]), np.array([0,0,0]), np.array([0.0]))).reshape(1, 7)
            # print('time', t)
            # print('action', action[0])
            # print('robot0_eef_pos', obs["robot0_eef_pos"])
            obs, reward, done, info = env.step(action[0].tolist())
            done = False
            if done:
                task_successes += 1
                total_successes += 1
                break
            t += 1
        except Exception as e:
            print(f"Caught exception: {e}")
            # log_file.write(f"Caught exception: {e}\n")
            break
    print('knife_1_pos', obs['knife_1_pos'])
    print('box_1_pos', obs['box_1_pos'])
    print('hammer_1_pos', obs['hammer_1_pos'])
    print('watch_1_pos', obs['watch_1_pos'])
    task_episodes += 1
    total_episodes += 1

    # Save a replay video of the episode
    save_rollout_video(
        replay_images, total_episodes, success=done, task_description=language, log_file=None, save_path = os.path.join(save_dir, result),  )
    save_rollout_video(
        replay_images, total_episodes, success=done, task_description=language, log_file=None, save_path = os.path.join('./rollouts', result),  )
    exit(0)   


    env.close()
    print("-------finish-------")
    # exit(0)
    
    # display(Image.fromarray(obs["agentview_image"][::-1]))



if __name__ == "__main__":
    cfg = GenerateConfig
    root_dir = "/data/workspace/LIBERO/spatial/task_refine_pos"
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('new5.json'):
                json_file = os.path.join(root, file)
                # json_file = '/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot/behind/Place_the_cup_behind_the_mixer_on_the_table.__upside_down/20240824-174506_no_interaction/task_config_new3.json'
                # json_file = '/data/workspace/LIBERO/spatial/task_refine_pos/top/Place_the_mug_on_top_of_the_wallet_on_the_table._/20240824-165633_no_interaction/task_config_new5.json'
                json_file = '/data/workspace/LIBERO/spatial/task_refine_pos/behind/Place_the_knife_behind_the_box_on_the_table._/20240824-203605_no_interaction/task_config_new5.json'
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    main(root, file_data, json_file, cfg)