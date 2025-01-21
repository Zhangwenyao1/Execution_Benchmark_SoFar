"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import wandb
from PIL import Image
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")


from collections import Counter
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image


import re
from libero.libero.utils.bddl_generation_utils import *

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
import json
import imageio
# absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
# 

import time
from typing import Optional, Union
from dataclasses import dataclass
# from octo_model import OctoInference

import debugpy
debugpy.listen(('0.0.0.0', 5681))
print('Waiting for debugger attach')
debugpy.wait_for_client()

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

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 100                     # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1                 # Number of rollouts per task

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
    else:
        return selected_obj


def create_task_dict(bddl_path, task_json_path, video_path, final_positions):
    task_dict = {
        "bddl_path": bddl_path,
        "task_json_path": task_json_path,
        "video_path": video_path,
        "final_positions": final_positions,  # List of positions for 10 trials
    }
    return task_dict









def eval_libero(save_path, root, json_data, json_file, ghost_task, cfg: GenerateConfig):
    
    ghost = True
    try_time = 0
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)
    while ghost:
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

        mp4_path = bddl_file_names[0].replace('bddl','mp4')
        
        local_log_filepath = os.path.join(save_path, "result.txt")
        log_file = open(local_log_filepath, "w")
        # print(f"Logging to local log file: {local_log_filepath}")
        # Start evaluation
        total_episodes, total_successes = 0, 0

        # Initialize LIBERO environment and task description
        env_args = {
        "bddl_file_name": bddl_file_names[0],
        # "bddl_file_name": bddl_file_namesssss,
        "camera_heights": 256,
        "camera_widths": 256
    }
        task_description = instruction
        
        env = OffScreenRenderEnv(**env_args)

        obj_pos_dict = dict()
        # Start episodes
        task_episodes, task_successes = 0, 0
        # Reset environment
        obs = env.reset()
        # Setup
        t = 0
        replay_images = []

        while t < cfg.num_steps_wait:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step([0,0,0,0,0,0,1])
                replay_images.append(obs["agentview_image"][::-1])
                t += 1
                continue

        log_file.write(f"Position: {done}\n")           
        task_episodes += 1
        total_episodes += 1
        
        
        
        if try_time > 6:
            task_name = mp4_path.split("/")[-1]  # 提取 mp4 文件名
            # timestamp = task_name.split("_")[-2]  # 提取时间戳部分
            ghost_task[task_name] = 1
            break
        # change_dict = {}
        for index, obj in enumerate(selected_obj):
            obj = obj.lower().replace(' ', '_')
            obj_pos_dict.update({obj: []}) if obj not in obj_pos_dict else None
            obj_pos_dict[obj].append(np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).reshape(1, 7).tolist())
            
            init_coordinate = json_data['init_obj_pos'][index][:3]
            x_change = obs[obj+"_1_pos"][0] - init_coordinate[0]
            y_change = obs[obj+"_1_pos"][1] - init_coordinate[1]
            
            if abs(y_change) > 0.1:
                print("------------warning_y-------------")
                print(f"Object '{obj}' has significant movement:")
                print(f" - X-axis change: {y_change:.3f}")
                json_data['init_obj_pos'][index][1] += float(y_change*0.1)
                ghost = True 
                try_time += 1
                break
            elif abs(x_change) > 0.1:
                print("------------warning_x-------------")
                print(f"Object '{obj}' has significant movement:")
                print(f" - X-axis change: {y_change:.3f}")
                json_data['init_obj_pos'][index][0] += float(x_change*0.1)
                ghost = True
                try_time += 1
                break
            else:
                json_data['init_obj_pos'][index][:7] = np.hstack((obs[obj+"_1_pos"], obs[obj+"_1_quat"])).tolist()
                ghost = False
                
                
                
                
    json_data            
    base, ext = os.path.splitext(json_file)  # 分离文件名和扩展名
    refined_file = f"{base}_refine_ghost{ext}"     # 添加 "_refine"
        
    with open(refined_file, 'w') as f:
        json.dump(json_data, f, indent=4) 
            
                
                
                
                
                
                
                
        #     task_name = mp4_path.split("/")[-1]  # 提取 mp4 文件名
        #     timestamp = task_name.split("_")[-2]  # 提取时间戳部分
            
        #     # 将变化记录到字典中
        #     if timestamp not in change_dict:
        #         change_dict[timestamp] = []
        #         change_dict[timestamp].append({
        #             "object": obj,
        #             "x_change": x_change,
        #             "y_change": y_change
        #         })

        # # 输出记录的变化字典
        # print(change_dict)

    # 可选：将字典保存为 JSON 文件
        # import json
        # output_path = "/path/to/save/changes.json"
        # with open(output_path, 'w') as f:
        #     json.dump(change_dict, f, indent=4)

            # log_file.write(obj: obs[obj+"_1"] \n")
        # Save a replay video of the episode
    save_rollout_video(
        replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, mp4_path=mp4_path
    )
    # Log current results
    log_file.flush()  
    task_info = create_task_dict(bddl_file_names[0], json_file, mp4_path,  obj_pos_dict)
    # Log final results

    log_file.flush()

    # Save local log file
    log_file.close()

            
        
        
        
        # Push total metrics and local log file to wandb
    return task_info, ghost_task


if __name__ == "__main__":
    
    cfg = GenerateConfig
    
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # # Set random seed
    # set_seed_everywhere(cfg.seed)


    root_dir = "/data/workspace/LIBERO/spatial/task_refine_pos"
    output_file = os.path.join("/data/workspace/LIBERO/spatial/task_refine_pos", f"task_dict_static.json")
    ghost_task_file = os.path.join("/data/workspace/LIBERO/spatial/task_refine_pos", f"ghost_task_dict_static.json")
    # 初始化或加载已存在的 task_dict
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            task_dict = json.load(f)
    else:
        task_dict = {}
        
    if os.path.exists(ghost_task_file):
        with open(ghost_task_file, 'r') as f:
            ghost_task = json.load(f)
    else:
        ghost_task = {}        
        
        
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('new5.json'):
                json_file = os.path.join(root, file)
                task_name = os.path.basename(root)
                # 检查任务是否已经完成
                if task_name in task_dict:
                    print(f"Skipping {task_name} as it is already processed.")
                    continue
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    # 构建保存路径并创建目录（如果不存在）
                    parts = json_file.split("/")
                    save_path = os.path.join('/data/workspace/LIBERO/octo', "/".join(parts[5:8]))
                    os.makedirs(save_path, exist_ok=True)
                    # print(f)
                    task_dict[os.path.basename(root)], ghost_dict = eval_libero(save_path, root, file_data, json_file, ghost_task, cfg)
                    # 将 task_dict 保存到 JSON 文件
                    with open(output_file, 'w') as f:
                        json.dump(task_dict, f, indent=4)
                    with open(ghost_task_file, 'w') as f:
                        json.dump(ghost_dict, f, indent=4)