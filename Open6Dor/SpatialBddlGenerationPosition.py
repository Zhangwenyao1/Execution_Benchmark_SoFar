from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from collections import Counter
import json
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
import os
from libero.libero.envs import OffScreenRenderEnv
from IPython.display import display
from PIL import Image

import torch
import torchvision
import numpy as np
# import debugpy
# debugpy.listen(("0.0.0.0", 5681))
# debugpy.wait_for_client()

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
                # region_centroid_xy=[0.0, -0.30],
                # region_centroid_xy = self.init_obj_pos[id][:2],
                region_centroid_xy = self.init_obj_pos[id],
                region_name=self.objs[id].replace(' ', '_')+'_'+ str(objects_dict[self.objs[id]])+'_init_region',
                target_name=self.workspace_name,
                region_half_len=0.02,
                yaw_rotation = tuple(self.init_obj_pos[id][3:7]),
                goal_quat = self.quat_dict.get(self.objs[id], [0, 0, 0, 1]),
                # goal_quat =  self.quat_dict[self.objs[id]] if self.quat_dict[self.objs[id]] else [1,0,0,0],
                # yaw_rotation=(1,0,0,0)
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
# 测试示例
# orientation = "left"
# instruction = "Place the apple to the left of the book"

# # 调用函数
# result = extract_object_from_instruction(orientation, instruction)






def main(root, json_data, json_file):
    # kitchen_scene_1
    
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['target_obj_name']  

    orientation = json_data['position_tag']
    instruction = json_data['instruction']
    selected_obj = json_data['selected_obj_names']
    matching_word = extract_object_from_instruction(orientation,instruction, selected_obj, goal_objs)
    # print('orientation:', orientation)
    # print('instruction:', instruction)
    # print('selected_obj:', selected_obj)
    # print(matching_word)
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
    
    bddl_folder = os.path.join("/data/workspace/LIBERO/spatial/bddl_refine_pos")
    img_folder =  os.path.join("/data/workspace/LIBERO/spatial/bddl_refine_pos_img")
    if not os.path.exists(bddl_folder):
        os.makedirs(bddl_folder)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
    bddl_file_names, failures = generate_bddl_from_task_info(folder=bddl_folder, json_data=json_data) # generate the bddl_file

    




    # with open(bddl_file_names[0], "r") as f:
    #     print(f.read())


    

    env_args = {
        "bddl_file_name": bddl_file_names[0],
        "camera_heights": 1024,
        "camera_widths": 1024
    }

    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    img =Image.fromarray(obs["frontview_image"][::-1])
    img.save(root+"/test.png")
    img.save(os.path.join(img_folder, bddl_file_names[0].split("/")[-1].replace(".bddl", ".png")))
    print(bddl_file_names)
    print("Encountered some failures: ", failures)
    print('img', root+"/test.png")
    target_filename = "isaac_render-rgb-0-1.png"
    print(os.path.join(os.path.dirname(json_file), target_filename))
    print('json_file', json_file)
    # return
    exit(0)
    print("----------------finish----------------")
    img.save('/data/workspace/LIBERO/spatial/img/agentview_image.png')
    dummy_actions = [0, 0, 0, 0, 0, 0, -1]
    for t in range(5):
        obs, reward, done, info = env.step(dummy_actions)    
        img =Image.fromarray(obs["agentview_image"][::-1])
        img.save(f"/data/workspace/LIBERO/spatial/img/agentview_image_{t}.png")
        t += 1
        continue
    exit(0)
    # display(Image.fromarray(obs["agentview_image"][::-1]))

    # with open(bddl_file_names[0], "r") as f:
    #     print(f.read())
        
if __name__ == "__main__":
    root_dir = "/data/workspace/LIBERO/spatial/task_refine_pos"
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('new5.json'):
                json_file = os.path.join(root, file)
                # json_file = '/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot/behind/Place_the_cup_behind_the_mixer_on_the_table.__upside_down/20240824-174506_no_interaction/task_config_new3.json'
                # json_file = '/data/workspace/LIBERO/spatial/task_refine_pos/top/Place_the_mug_on_top_of_the_wallet_on_the_table._/20240824-165633_no_interaction/task_config_new5.json'
                # json_file = '/data/workspace/LIBERO/spatial/task_refine_pos/behind/Place_the_knife_behind_the_box_on_the_table._/20240824-203605_no_interaction/task_config_new5.json'
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    main(root, file_data, json_file)