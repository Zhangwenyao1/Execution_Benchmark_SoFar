import json
import os
import evaluator
import numpy as np
from scipy.spatial.transform import Rotation as R
import debugpy
# debugpy.listen(('0.0.0.0', 5683))
# print('Waiting for debugger attach')
# debugpy.wait_for_client()




def evaluate_exec_rot(result_path, ghost_json):
    
    eval_dict = {}
    pos = []
    rot = []
    all = []
    eval_dict = {}
    result_dict = {}
    success = 0
    with open(result_path, 'r') as f:
        result_dict = json.load(f)
        
    detail_acc = {
        "left": [],
        "right": [],
        "front": [],
        "behind": [],
        "top": [],
        "between": [],
        "center": []
    }
        
    # for root, dirs, files in os.walk(output_root):
    for task_id in result_dict:
        
        # First filter the uneffective tasks
        if task_id not in json.load(open("/data/workspace/LIBERO/open6dor_list.json", 'r')):
            continue

        
        config_file_path = result_dict[task_id]["task_json_path"]
        if "/data/workspace/LIBERO/spatial" in config_file_path:
            config_file_path = config_file_path.replace("/data/workspace/LIBERO/spatial", "/data/datasets/open6dor")
        elif "/mnt/afs/open6dor_rot" in config_file_path:
            config_file_path = config_file_path.replace("/mnt/afs/open6dor_rot", "/data/datasets/open6dor")
        task_config = json.load(open(config_file_path, 'r'))
        obj_name = task_config["selected_obj_names"][-1].lower().replace(" ", "_")
        pos_tag = task_config["position_tag"]  
        
        
        try:
            pred_position = result_dict[task_id]["final_positions"][obj_name][0][0][:3]
        except:
            pred_position = [0,0,0]
        try:
            pred_quaternion = result_dict[task_id]["final_positions"][obj_name][-1][0][-4:]
        except:
            pred_rotation  = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            pred_quaternion = R.from_matrix(pred_rotation).as_quat()
            pred_quaternion = pred_quaternion.tolist()
             
        if np.isnan(pred_quaternion).any():
            pred_quaternion = np.array([1, 0, 0, 0])
            pred_quaternion = pred_quaternion.tolist()
        deviation= evaluator.evaluate_rot(config_file_path, pred_quaternion)
        eval_dict[task_id] = { "deviation": deviation, "proposal": pred_quaternion}
        
        
        sel_pos_list = []
        for obj in result_dict[task_id]["final_positions"]:
            sel_pos_list.append(result_dict[task_id]["final_positions"][obj][0][0][:3])
        
        # First evaluation the position
        if sel_pos_list == []:
            continue
        if pos_tag in ["left", "right", "front", "behind", "top"]:
            sel_pos = sel_pos_list[0]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos)
            pos.append(success)
        elif pos_tag == "between":
            sel_pos_1 = sel_pos_list[0]
            sel_pos_2 = sel_pos_list[1]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
            pos.append(success)
        elif pos_tag == "center":
            sel_pos_all = sel_pos_list[:-1]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos_all=sel_pos_all)  
            pos.append(success)
        else:
            sel_pos = sel_pos_list[0]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos)
            pos.append(success)
        pos.append(success)
        success = 1 if success else 0
        
        detail_acc[pos_tag].append(success)

        pred_quaternion = result_dict[task_id]["final_positions"][obj_name][-1][0][-4:]
        pred_position = result_dict[task_id]["final_positions"][obj_name][-1][0][:3]
        
        deviation= evaluator.evaluate_rot(config_file_path, pred_quaternion)
        
        # pred_quaternion = pred_quaternion.tolist()
        eval_dict[task_id] = {"pos_success": success, "pred_position": pred_position, "deviation": deviation,
                           "pred_quaternion": pred_quaternion}

        if deviation == "No annotation found" or deviation == "Annotation stage 2":
            continue
        else:
            int(deviation)

        if int(deviation) <= 45:
            rot.append(1)
        else:
            rot.append(0)

        if success and int(deviation) <= 45:
            all.append(1)
        else:
            all.append(0)
    
    for pos_tag, pos_acc in detail_acc.items():
        print(pos_tag, sum(pos_acc) / (len(pos_acc) +1e-5))

    print(len(all))
    print("6-dof pos acc:", sum(pos) / len(pos))
    print("6-dof rot acc:", sum(rot) / len(rot))
    print("6-dof all acc:", sum(all) / len(all))            

    with open(eval_file, "w") as f:
        json.dump(eval_dict, f, indent=4)
    return eval_dict


def evaluate_exec_rot_only(result_path, eval_file):
    eval_dict = {}
    level1 = []
    level2 = []
    level3 = []
    all = []
    result_dict = {}

    with open(result_path, 'r') as f:
        result_dict = json.load(f)
        
    

    for task_id in result_dict:

        if task_id not in json.load(open("/data/workspace/LIBERO/open6dor_list.json", 'r')):
            continue
        
        config_file_path = result_dict[task_id]["task_json_path"]
        task_config = json.load(open(config_file_path, 'r'))
        obj_name = task_config["selected_obj_names"][-1].lower().replace(" ", "_")
        rot_tag_level = task_config["rot_tag_level"]    
        try:
            pred_quaternion = result_dict[task_id]["final_positions"][obj_name][-1][0][-4:]
        except:
            pred_rotation  = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            pred_quaternion = R.from_matrix(pred_rotation).as_quat()
            pred_quaternion = pred_quaternion.tolist()
            
        
        
        if np.isnan(pred_quaternion).any():
            pred_quaternion = np.array([1, 0, 0, 0])
            pred_quaternion = pred_quaternion.tolist()
        deviation= evaluator.evaluate_rot(config_file_path, pred_quaternion)
        eval_dict[task_id] = { "deviation": deviation, "proposal": pred_quaternion}
        
        if deviation == "No annotation found" or deviation == "Annotation stage 2":
            continue
        else:
            int(deviation)
        
        # pred_quaternion = pred_quaternion.tolist()
        if int(deviation) <= 45:
                if rot_tag_level == 0:
                    level1.append(1)
                if rot_tag_level == 1:
                    level2.append(1)
                if rot_tag_level == 2:
                    level3.append(1)
                all.append(1)
        else:
            if rot_tag_level == 0:
                level1.append(0)
            if rot_tag_level == 1:
                level2.append(0)
            if rot_tag_level == 2:
                level3.append(0)
            all.append(0)
    print("len level1", len(level1))
    print("level1 acc:", sum(level1) / len(level1))
    print("len level2", len(level2))
    print("level2 acc:", sum(level2) / len(level2))
    print("len level3", len(level3))
    print("level3 acc:", sum(level3) / len(level3))
    print("all acc:", sum(all) / len(all))
    
    print("all", len(all))
    with open(eval_file, "w") as f:
        json.dump(eval_dict, f, indent=4)
    return eval_dict  


def evaluate_exec_pos(result_path,eval_file):
    
    
    eval_dict = {}
    level1 = []
    level2 = []
    level3 = []
    level4 = []
    all = []
    

    result_dict = {}
    success = 0

    with open(result_path, 'r') as f:
        result_dict = json.load(f)

    total_success = 0
    detail_acc = {
        "left": [],
        "right": [],
        "front": [],
        "behind": [],
        "top": [],
        "between": [],
        "center": []
    }

    for task_id in result_dict:
        if task_id not in json.load(open("/data/workspace/LIBERO/open6dor_list.json", 'r')):
            continue
        config_file_path = result_dict[task_id]["task_json_path"]
        task_config = json.load(open(config_file_path, 'r'))
        obj_name = task_config["selected_obj_names"][-1].lower().replace(" ", "_")


        
        try:
            pred_position = result_dict[task_id]["final_positions"][obj_name][0][0][:3]
        except:
            pred_position = [0, 0, 0]
            
        pos_tag = task_config["position_tag"]
      
        

        sel_pos_list = []
        for obj in result_dict[task_id]["final_positions"]:
            sel_pos_list.append(result_dict[task_id]["final_positions"][obj][0][0][:3])
        
        # First evaluation the position
        if pos_tag in ["left", "right", "front", "behind", "top"]:
            sel_pos = sel_pos_list[0]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos)
            level1.append(success)
        elif pos_tag == "between":
            sel_pos_1 = sel_pos_list[0]
            sel_pos_2 = sel_pos_list[1]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos_1=sel_pos_1, sel_pos_2=sel_pos_2)
            level2.append(success)
        elif pos_tag == "center":
            sel_pos_all = sel_pos_list[:-1]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos_all=sel_pos_all)
            level2.append(success)
        else:
            sel_pos_all = sel_pos_list[0]
            success = evaluator.evaluate_posi(pred_position, pos_tag, sel_pos_all=sel_pos_all)  
            level3.append(success)

        detail_acc[pos_tag].append(success)

        all.append(success)
        success = 1 if success else 0
        
        total_success += success
        eval_dict[task_id] = {"success": int(success), "proposal": pred_position}
        
    for pos_tag, pos_acc in detail_acc.items():
        print(pos_tag, sum(pos_acc) / len(pos_acc))

    print("len level1", len(level1))
    print("level1 acc:", sum(level1) / len(level1))
    print("len level2", len(level2))
    print("level2 acc:", sum(level2) / len(level2))
    print("len level3", len(level3))
    print("level3 acc:", sum(level3) / (len(level3) + 1e-5))
    # print("level4 acc:", sum(level4) / (len(level4) + 1e-5))
    print("all acc:", sum(all) / len(all))
    print("all", len(all))
   
    with open(eval_file, "w") as f:
        json.dump(eval_dict, f, indent=4)
    return eval_dict




if __name__ == "__main__":
    
    
    
    result_path = '/data/workspace/LIBERO/sofar_output/merged_open6dor_pos_exec_dict_pos_0218.json'
    eval_file = "/data/workspace/LIBERO/exp_log/sofar/merged_open6dor_pos_exec_dict_0218.json"
    evaluate_exec_pos(result_path, eval_file)
    
    # # rot_only
    # print("*"*20)
    result_path = "/data/workspace/LIBERO/sofar_output/open6dor_exec_rotonly_rotonly_0218.json"
    eval_file = "/data/workspace/LIBERO/exp_log/sofar/open6dor_exec_dict_rot_only_0218.json"
    evaluate_exec_rot_only(result_path, eval_file)
    
    
    print("*"*20)
    # rot 
    # result_path = "/data/workspace/LIBERO/spatial/task_refine_rot/task_dict_static.json"
    result_path = "/data/workspace/LIBERO/open6dor_exec_dict_rot_0210_merged.json"
    eval_file = "/data/workspace/LIBERO/exp_log/sofar/eval_rot_0210.json"
    evaluate_exec_rot(result_path, eval_file)
    
    

                    
