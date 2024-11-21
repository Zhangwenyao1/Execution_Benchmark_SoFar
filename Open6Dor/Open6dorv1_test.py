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

class CustomObjects(MujocoXMLObject):
    def __init__(self, custom_path, name, obj_name, joints=[dict(type="free", damping="0.0005")], goal_position = None, goal_quat = None):
        # make sure custom path is an absolute path
        assert(os.path.isabs(custom_path)), "Custom path must be an absolute path"
        # make sure the custom path is also an xml file
        assert(custom_path.endswith(".xml")), "Custom path must be an xml file"
        super().__init__(
            custom_path,
            name=name,
            joints=joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.goal_quat = goal_quat
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.object_properties = {"vis_site_names": {}}
    # TODO: Wenyao add the realtionship between this_position and other_position
    def under(self, this_position, this_mat, other_position, other_height=0.10):
        """
        Checks whether an object is on this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return total_size[2] - 0.005 < delta_position[2] < total_size[
            2
        ] + other_height and np.all(np.abs(delta_position[:2]) < total_size[:2])
        
    def left(self, this_position, this_mat, other_position, other_height=0.10):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.

        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[0]<0 
    
    
    def right(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """


        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[0]>0 
    
    
    
    def front(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """

        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.

        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]>0 
    
    def behind(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    
    
    def top(self, this_position, this_mat, other_position):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    
    
    def quat(self, this_position, this_mat, goal_quat):
        """
        Checks whether the object is contained within this SiteObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the SiteObject as an axis-aligned grid.
        Args:
            this_position: 3D position of this SiteObject
            other_position: 3D position of object to test for insertion
        """
        # (TODO) Yifeng: The transformation for size is a little bit
        # hacky at the moment. Will dig deeper into it.
        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return delta_position[1]<0 
    

@register_object
class LiberoMug(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 goal_quat = [1, 0, 0, 0]
                 ):
        super().__init__(
            custom_path = os.path.abspath(os.path.join(
                # str(absolute_path),
                # f"assets/stable_scanned_objects/{obj_name}/{obj_name}.xml",
                f"./notebooks/custom_assets/libero_mug/{obj_name}.xml"
            )), 
            name=name,
            obj_name=obj_name,
            goal_quat= goal_quat,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
        # self.quat = [0.0, 0.0, 0.7071067690849304, 0.5]
        # self.quat = [1.0, 0.0, 0, 0.0]



@register_object
class LiberoMugYellow(CustomObjects):
    def __init__(self,
                 name="libero_mug",
                 obj_name="libero_mug",
                 goal_quat = [1, 0, 0, 0],
                 ):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                "./notebooks", "custom_assets", "libero_mug_yellow", "libero_mug_yellow.xml"
                )),
            name=name,
            obj_name=obj_name,
            goal_quat= goal_quat,
        )
        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None






@register_object
class Apple(CustomObjects):
    def __init__(self,
                 name="apple",
                 obj_name="apple",
                 goal_quat = [1, 0, 0, 0]
                 ):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                "/data/workspace/LIBERO/Open6Dor/open6dor_asset", "fbda0b25f41f40958ea984f460e4770b", "material", "material.xml"
                )),
            name=name,
            obj_name=obj_name,
            goal_quat= goal_quat
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None
        # self.quat = [0.0, 0.0, 0.7071067690849304, 0.1]

        # self.quat = [1.0, 0.0, 0, 0.0]




@register_object
class tissue_box(CustomObjects):
    def __init__(self,
                 name="tissue_box",
                 obj_name="tissue_box",
                 goal_quat = [1, 0, 0, 0]
                 ):
        super().__init__(
            custom_path=os.path.abspath(os.path.join(
                "/data/workspace/LIBERO/Open6Dor/open6dor_asset", "dc4c91abf45342b4bb8822f50fa162b2", "material", "material.xml"
                )),
            name=name,
            obj_name=obj_name,
            goal_quat= goal_quat
        )

        self.rotation = {
            "x": (-np.pi/2, -np.pi/2),
            "y": (-np.pi, -np.pi),
            "z": (np.pi, np.pi),
        }
        self.rotation_axis = None


        
@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self, json_data = None):
        self.json_data = json_data
        self.objs = self.json_data['selected_obj_names']
        self.number_obj = len(self.objs)
        self.init_obj_pos = self.json_data['init_obj_pos']
        self.quat_set = self.json_data['anno_target']['annotation'][list(self.json_data['anno_target']['annotation'].keys())[0]]['quat']
        fixture_num_info = {
            "kitchen_table": 1,
        }
        
        objects_dict = Counter(self.objs)
        object_num_info = {
            **objects_dict,
            # "libero_mug_yellow": 1,
        }

        
        super().__init__(
            workspace_name="kitchen_table",  # define the scene base
            fixture_num_info=fixture_num_info,
            object_num_info=object_num_info,
        )
        
    def define_regions(self):
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            self.regions.update(
                self.get_region_dict(
                # region_centroid_xy=[0.0, -0.30],
                region_centroid_xy = self.init_obj_pos[id][:2],
                region_name=self.objs[id]+'_'+ str(objects_dict[self.objs[id]])+'_init_region',
                target_name=self.workspace_name,
                region_half_len=0.02,
                # yaw_rotation=(np.pi, np.pi),
                yaw_rotation = tuple(self.init_obj_pos[id][3:7]),
                goal_quat =  self.quat_set[id] if id<len(self.quat_set) else [1,0,0,0],
                # yaw_rotation=(1,0,0,0)
            )
            )
            objects_dict[self.objs[id]] +=1
            
            
        
        # self.regions.update(
        #     self.get_region_dict(
        #         region_centroid_xy=[0.0, -0.30],
        #         region_name="wooden_cabinet_init_region",
        #         target_name=self.workspace_name,
        #         region_half_len=0.01,
        #         yaw_rotation=(np.pi, np.pi),
        #         # yaw_rotation=(1,0,0,0)
        #     )
        # )

        # self.regions.update(
        #     self.get_region_dict(
        #         region_centroid_xy=[0.0, 0.0],
        #         region_name="libero_mug_init_region",
        #         target_name=self.workspace_name,
        #         region_half_len=0.025,
        #         # wenyao
        #         # yaw_rotation=(np.pi, np.pi),
        #         yaw_rotation=(1,0,0,0)
        #         # wenyao
        #     )
        # )

        # self.regions.update(
        #     self.get_region_dict(
        #         region_centroid_xy=[-0.1, 0.15], # define the objct position
        #         region_name="libero_mug_yellow_1_init_region", # define the object region name 
        #         target_name=self.workspace_name,
        #         region_half_len=0.025, # here we difine the scale of the object
        #         yaw_rotation=(np.pi, np.pi),
        #     )
        # )
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(
            self.regions
        )

    @property
    def init_states(self):
        states = []
        objects_dict = dict.fromkeys(self.objs, 1)
        for id in range(self.number_obj):
            states.append(
                ("On", self.objs[id]+'_'+ str(objects_dict[self.objs[id]]), "kitchen_table_"+self.objs[id]+'_'+ str(objects_dict[self.objs[id]])+'_init_region'
            )
            )
            objects_dict[self.objs[id]] +=1

        # states = [
        #     ("On", "libero_mug_1", "libero_mug_1_init_region"),
        #     ("On", "libero_mug_yellow_1", "libero_mug_yellow_1_init_region"),
        #     ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        # ]
        return states
    

class Open6DorConfig:
    def __init__(self, json_path):
        with open(json_path, 'r') as file:
            self.json_data = json.load(file)

    def task_instruction(self):
        self.language = self.json_data['instruction']
        return self.language
    def task_object_state(self):
        goal_object = self.json_data["target_obj_name"]
        goal_state = self.json_data['anno_target']['annotation'][list(self.json_data['anno_target']['annotation'].keys())[0]]['quat'][0]
        return goal_object, goal_state
    
    
def main(json_path, folder_path, img_folder_path):
    scene_name = "kitchen_demo_scene"
    open6dor_config = Open6DorConfig(json_path)
    language = open6dor_config.task_instruction()
    goal_object, state = open6dor_config.task_object_state()
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    register_task_info(
                    language, # register the task 
                    scene_name=scene_name,
                    objects_of_interest=[],
                    goal_states=[
                        # ("Open", "wooden_cabinet_1_top_region"),
                        # ("In", "libero_mug_yellow_1", "wooden_cabinet_1_top_region"), # define the goal state
                        ("Left", "tissue_box_1", "apple_1"),
                        # "Quat", 
                        # ("Left", "libero_mug_yellow_1", "libero_mug_1"),
                        # ("Quat", goal_object, get_goal_state),
                        ],
                    json_data = json_data,
    )
    YOUR_BDDL_FILE_PATH = "./custom_pddl"
    bddl_file_names, failures = generate_bddl_from_task_info(folder=YOUR_BDDL_FILE_PATH, json_data=json_data) # generate the bddl_file
    print(bddl_file_names)

    print("Encountered some failures: ", failures)

    # with open(bddl_file_names[0], "r") as f:
    #     print(f.read())

    
    # bddl_file_names = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-211316_no_interaction.bddl'
    # bddl_file_names = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-211311_no_interaction.bddl'
    
    

# 遍历文件夹中的每个文件
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
        
    file_path = '/data/workspace/LIBERO/spatial/bddl_refine_pos/KITCHEN_DEMO_SCENE_20240824-155620_no_interaction.bddl'
    # /data/workspace/LIBERO/Open6Dor/task/bddl_rot_only/KITCHEN_DEMO_SCENE_20240826-212158_no_interaction.bddl
    env_args = {
        # "bddl_file_name": bddl_file_names,
        "bddl_file_name": file_path,
        "camera_heights": 2048,
        "camera_widths": 2048
        }
        
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
    intrinstic_matrix = get_camera_intrinsic_matrix(env.sim, 'agentview', 128, 128)
    extrinsic_matrix = get_camera_extrinsic_matrix(env.sim, 'agentview')
    img = Image.fromarray(obs["agentview_image"][::-1])
    img.save("/data/workspace/LIBERO/agent_view.png")

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
    folder_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only'
    
    img_folder_path = '/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only_imgs'
    
    if not os.path.exists(img_folder_path):
        os.mkdir(img_folder_path)
    main(json_path='/data/workspace/LIBERO/yufei.json', folder_path=folder_path, img_folder_path=img_folder_path)