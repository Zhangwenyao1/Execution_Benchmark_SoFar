import sys
import os
sys.path.append('/data/workspace/LIBERO')
import re
import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion

from libero.libero.envs.base_object import register_object

import pathlib
import os 
from collections import Counter
from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from libero.libero.envs import OffScreenRenderEnv
from IPython.display import display
from PIL import Image

import torch
import torchvision

import re
from libero.libero.envs import objects
from libero.libero.utils.bddl_generation_utils import *
from libero.libero.envs.objects import OBJECTS_DICT
from libero.libero.utils.object_utils import get_affordance_regions

from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
import json
# absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
# 




def main(folder_path, img_folder_path):

    # with open(bddl_file_names[0], "r") as f:
    #     print(f.read())

    
    # bddl_file_names = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-211316_no_interaction.bddl'
    # bddl_file_names = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-211311_no_interaction.bddl'
    
    

# 遍历文件夹中的每个文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
    # while True:
        # file_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot/KITCHEN_DEMO_SCENE_20240824-181952_no_interaction.bddl'
        # file_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-220537_no_interaction.bddl'
        # file_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-214144_no_interaction.bddl'
        # file_path ='/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-211450_no_interaction.bddl'
        print(file_path)
        env_args = {
            # "bddl_file_name": bddl_file_names,
            "bddl_file_name": file_path,
            "camera_heights": 1024,
            "camera_widths": 1024,
            }
            
        env = OffScreenRenderEnv(**env_args)
        obs = env.reset()
        img = Image.fromarray(obs["paperview_image"][::-1])
        
        img.save(os.path.join(img_folder_path, file_path.split("/")[-1].replace(".bddl", ".png")))
        print(os.path.join(img_folder_path, file_path.split("/")[-1].replace(".bddl", ".png")))
        return 
        # set the seed , the default seed is 0
        seed = 0
        env.seed(seed)
        
        done = None
        # for i in range(10):
        # while not done:
        #     # TODO:  wenyao: here we should output the action based on the observation using our model
        #     # action = get_policy_action(obs)         # use observation to decide on an action
        #     action = [0.] * 7
        #     obs, reward, done, _ = env.step(action) # play action


        env.close()
        print("-------finish-------")
    
    # display(Image.fromarray(obs["agentview_image"][::-1]))



if __name__ == "__main__":
    # main(json_path= '/data/workspace/LIBERO/task_config_new3.json')
    # folder_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only'
    folder_path = '/data/workspace/LIBERO/spatial/bddl_refine_pos'
    img_folder_path = '/data/workspace/LIBERO/spatial/bddl_refine_pos_imgs'
    # img_folder_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only_imgs'
    if not os.path.exists(img_folder_path):
        os.mkdir(img_folder_path)
    main(folder_path=folder_path, img_folder_path=img_folder_path)