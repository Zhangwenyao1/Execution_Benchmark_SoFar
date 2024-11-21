from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils.task_generation_utils import register_task_info, get_task_info, generate_bddl_from_task_info
from collections import Counter
import json
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
import os
@register_mu(scene_type="kitchen")
class KitchenDemoScene(InitialSceneTemplates):
    def __init__(self, json_data = None):
        self.json_data = json_data
        self.objs = self.json_data['selected_obj_names']
        self.number_obj = len(self.objs)
        self.init_obj_pos = self.json_data['init_obj_pos']
        self.quat_dict =  dict()
        self.goal_object = self.json_data['target_obj_name']
        quat_list = self.json_data['anno_target']['annotation'][list(self.json_data['anno_target']['annotation'].keys())[0]]['quat']
        print(quat_list)
        if len(quat_list)>2:
            if type(quat_list[0])!= float and len(quat_list[0])>2:
                self.quat_dict[self.goal_object] = quat_list[0]
            else:
                self.quat_dict[self.goal_object] = quat_list
        else:
            self.quat_dict[self.goal_object] = quat_list[0]
            
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
                # region_centroid_xy = self.init_obj_pos[id][:2],
                region_centroid_xy = [0.1 * x for x in self.init_obj_pos[id][:2]],
                region_name=self.objs[id].replace(' ', '_')+'_'+ str(objects_dict[self.objs[id]])+'_init_region',
                target_name=self.workspace_name,
                region_half_len=0.02,
                # yaw_rotation=(np.pi, np.pi),
                yaw_rotation = tuple(self.init_obj_pos[id][3:7]),
                goal_quat =  self.quat_dict[self.objs[id]] if self.quat_dict[self.objs[id]] else [1,0,0,0],
                # yaw_rotation=(1,0,0,0)
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
                ("On", self.objs[id].replace(' ', '_')+'_'+ str(objects_dict[self.objs[id]]), "kitchen_table_"+self.objs[id]+'_'+ str(objects_dict[self.objs[id]])+'_init_region'
            )
            )
            objects_dict[self.objs[id]] +=1

        # states = [
        #     ("On", "libero_mug_1", "libero_mug_1_init_region"),
        #     ("On", "libero_mug_yellow_1", "libero_mug_yellow_1_init_region"),
        #     ("On", "wooden_cabinet_1", "kitchen_table_wooden_cabinet_init_region"),
        # ]
        return states
    









def main(root, json_data):
    # kitchen_scene_1
    scene_name = "kitchen_demo_scene"
    bddl_name = os.path.basename(root)
    goal_objs = json_data['anno_target']['category']
    goal_quat = json_data['anno_target']['annotation'][list(json_data['anno_target']['annotation'].keys())[0]]['quat'][0]
                # goal_states=goal_states,    
    register_task_info(
                    bddl_name, # register the task 
                    scene_name=scene_name,
                    objects_of_interest=[],
                    goal_states=[
                    #     # ("Open", "wooden_cabinet_1_top_region"),
                    #     # ("In", "libero_mug_yellow_1", "wooden_cabinet_1_top_region"), # define the goal state
                    #     # ("Left", "tissue_box_1", "apple_1"),
                    #     # ("Left", "libero_mug_yellow_1", "libero_mug_1"),
                        ("Quat", goal_objs+'_1', goal_objs+'_1'),
                    # "Quat", goal_objs+'_1', [round(num, 2) for num in goal_quat]
                    ],
                    json_data = json_data,
    )
    bddl_folder = os.path.join("/data/workspace/LIBERO/Open6Dor/task/bddl_rot_only")
    if not os.path.exists(bddl_folder):
        os.makedirs(bddl_folder)
    bddl_file_names, failures = generate_bddl_from_task_info(folder=bddl_folder, json_data=json_data) # generate the bddl_file
    print(bddl_file_names)
    print("Encountered some failures: ", failures)

    with open(bddl_file_names[0], "r") as f:
        print(f.read())
        
if __name__ == "__main__":
    root_dir = "/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot_only/rot_ins"
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(root, file)
                # json_file = '/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot_only/rot_ins/ae7142127dd84ebbbe7762368ace452c_sideways/20240826-215104_no_interaction/task_config_new3.json'
                # json_file = '/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot_only/rot_ins/d5a5f0a954f94bcea3168329d1605fe9_sideways/20240826-212158_no_interaction/task_config_new3.json'
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                    if len(file_data['anno_target']['annotation']) > 0:
                        print(json_file)
    # json_file = "/data/workspace/LIBERO/Open6Dor/task/tasks_cfg_new/task0824_rot/behind/Place_the_apple_behind_the_bottle_on_the_table.__upright/20240824-165044_no_interaction/task_config_new3.json"
                        main(root, file_data)