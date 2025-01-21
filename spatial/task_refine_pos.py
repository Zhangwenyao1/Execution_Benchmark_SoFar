# for the task in pos, we have to load each object seperately

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

# import debugpy
# debugpy.listen(('0.0.0.0', 5681))
# print('Waiting for debugger attach')
# debugpy.wait_for_client()

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
    return mp4_path




@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self, json_data = None, obj=None):
        self.json_data = json_data
        index = json_data["selected_obj_names"].index(obj)
        self.init_coordinate = json_data['init_obj_pos'][index][:3]
        self.init_quat = json_data['init_obj_pos'][index][3:7]
        self.obj = obj
        self.xml = os.path.dirname(self.json_data['selected_urdfs'][index]) 
        

        fixture_num_info = {
            "open6dor": 1,
        }

        object_num_info = {
            self.obj: 1,
            # "libero_mug_yellow": 1,
        }

        
        super().__init__(
            workspace_name="open6dor",  # define the scene base
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )
        
    def define_regions(self):

        self.regions.update(
            self.get_region_dict(
            region_centroid_xy = self.init_coordinate,
            region_name = self.obj + '_1'+'_init_region',
            target_name=self.workspace_name,
            region_half_len=0.02,
            yaw_rotation = (np.pi, np.pi),
            goal_quat = [0, 0, 0, 1],
            xml = self.xml,
            init_pos = self.init_coordinate,
            init_quat = self.init_quat,
        )
        )
 
            # print(self.quat_dict[self.objs[id]] if self.quat_dict[self.objs[id]] else [1,0,0,0])
            
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = []
        states.append(
                ("On", self.obj+"_1", "open6dor_"+self.obj+'_1_init_region'
            )
            )

        return states
    



def create_task_dict(bddl_path, task_json_path, video_path, final_positions):
    task_dict = {
        "bddl_path": bddl_path,
        "task_json_path": task_json_path,
        "video_path": video_path,
        "final_positions": final_positions,  # List of positions for 10 trials
    }
    return task_dict









def eval_libero(save_path, root, json_data, json_file):
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  
    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    # define the static_position
    # every task has many object, and we can record the final position of each object
    print(json_data["selected_obj_names"])
    json_data["selected_obj_names"] = [
        obj_name.lower().replace(' ', '_') for obj_name in json_data["selected_obj_names"]
    ]
        
    print(json_data["selected_obj_names"])
    for index, obj in enumerate(json_data['selected_obj_names']):
        print(obj)
        init_coordinate = json_data['init_obj_pos'][index][:3]
        init_quat = json_data['init_obj_pos'][index][3:7]
        register_task_info(
            bddl_name, # register the task 
            scene_name=scene_name,
            objects_of_interest=[],
            goal_states=[
                ("On", obj+'_1', obj+'_1')
            ],
            json_data = json_data,
            init_coordinate = init_coordinate,
            init_quat = init_quat,
            obj = obj,
        )
       
        bddl_file_names, failures = generate_bddl_from_task_info(folder=save_path, json_data=json_data, obj=obj) # generate the bddl_file
        mp4_path = bddl_file_names[0].replace('bddl','mp4')
        env_args = {
        "bddl_file_name": bddl_file_names[0],
        # "bddl_file_name": bddl_file_namesssss,
        "camera_heights": 512,
        "camera_widths": 512
        }
        
        task_description = instruction
        env = OffScreenRenderEnv(**env_args)
        
        num_steps_wait = 20
        obj_pos_dict = dict()
        # Reset environment
        obs = env.reset()
        # Setup
        t = 0
        replay_images = []

        while t < num_steps_wait:

            obs, reward, done, info = env.step([0,0,0,0,0,0,1])
            t += 1
            replay_images.append(obs["agentview_image"][::-1])

        json_data['init_obj_pos'][index][:3] = obs[obj+"_1_pos"].tolist()
        json_data['init_obj_pos'][index][3:7] = obs[obj+"_1_quat"].tolist()
        
        
        save_rollout_video(
            replay_images, 1, success=done, task_description=task_description, log_file=None, mp4_path=mp4_path
        )
        
        

    # 创建新的文件名，在 .json 前添加 "_refine"
    base, ext = os.path.splitext(json_file)  # 分离文件名和扩展名
    refined_file = f"{base}_refine{ext}"     # 添加 "_refine"
        
    with open(refined_file, 'w') as f:
        json.dump(json_data, f, indent=4) 

    print(json_file)






    # Push total metrics and local log file to wandb
    # return json_file


if __name__ == "__main__":
    
    
    # # Set random seed
    # set_seed_everywhere(cfg.seed)


    root_dir = "/data/workspace/LIBERO/spatial/task_refine_pos"
    output_file = os.path.join("/data/workspace/LIBERO/spatial/task_refine_pos", f"task_refine_pos.json")
    # 初始化或加载已存在的 task_dict
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            task_dict = json.load(f)
    else:
        task_dict = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('new5.json'):
                json_file = os.path.join(root, file)
                task_name = os.path.basename(root)
                # 检查任务是否已经完成
                # if task_name in task_dict:
                #     print(f"Skipping {task_name} as it is already processed.")
                #     continue
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    # 构建保存路径并创建目录（如果不存在）
                    parts = json_file.split("/")
                    save_path = os.path.join('/data/workspace/LIBERO/octo', "/".join(parts[5:8]))
                    os.makedirs(save_path, exist_ok=True)
                    task_dict[os.path.basename(root)] = eval_libero(save_path, root, file_data, json_file)
                    # 将 task_dict 保存到 JSON 文件
                    with open(output_file, 'w') as f:
                        json.dump(task_dict, f, indent=4)