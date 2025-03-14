
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
print(BASE_DIR)
import json
import warnings
import numpy as np
from PIL import Image
from SoFar.depth.utils import depth2pcd, transform_obj_pts
from SoFar.segmentation import sam, florence
from SoFar.serve.scene_graph import open6dor_scene_graph
from SoFar.serve.utils import generate_rotation_matrix
from SoFar.serve.chatgpt import open6dor_spatial_reasoning, open6dor_parsing
warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)



def sofar_libero(detection_model, sam_model, orientation_model, image, depth, intrinsic_matrix, extrinsic_matrix, prompt):

    output_folder = "/mnt/afs/zhangwenyao/LIBERO/SpatialAgent/output"
    image = Image.fromarray(image)
    image.save("./output/img_simpler.png")
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    intrinsic = [fx, fy, cx, cy]
    pcd_camera, pcd_base = depth2pcd(depth, intrinsic, extrinsic_matrix)

    print("\nStart object parsing...")
    info = open6dor_parsing(prompt, image)
    print(json.dumps(info, indent=2))
    object_list = [info['picked_object']] + info['related_objects']
    print("Start Segment Anything...")
    # detections = grounding_dino.get_detections(image, object_list, detection_model, output_folder=output_folder, box_threshold=0.15, text_threshold=0.15, single=True)
    # mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)

    picked_object_name = info["picked_object"]
    detections = florence.get_detections(image, object_list, detection_model, output_folder=output_folder, single=True)
    mask, ann_img, object_names = sam.get_mask(image, object_list, sam_model, detections, output_folder=output_folder)
    index = object_names.index(picked_object_name)
    object_mask = mask[index]
    segmented_object = pcd_camera[object_mask]
    segmented_image = np.array(image)[object_mask]
    
    obj_pts_base = transform_obj_pts(segmented_object,extrinsic_matrix)
    
    colored_object_pcd = np.concatenate((segmented_object.reshape(-1, 3), segmented_image.reshape(-1, 3)), axis=-1)
    np.save(os.path.join(output_folder, f"picked_obj.npy"), colored_object_pcd)

    print("Generate scene graph...")
    picked_object_info, other_objects_info, picked_object_dict = open6dor_scene_graph(image, pcd_base, mask, info, object_names, orientation_model, output_folder=output_folder)
    
    print("picked info:", picked_object_info)
    print("other info:")
    for node in other_objects_info:
        print(node)
    
    
    print("Start spatial reasoning...")
    response = open6dor_spatial_reasoning(image, prompt, picked_object_info, other_objects_info)
    print(response)
    

    init_position = picked_object_dict["center"]
    target_position = response["target_position"]
    init_orientation = picked_object_dict["orientation"]
    target_orientation = info["target_orientation"]

    if len(target_orientation) > 0 and target_orientation.keys() == init_orientation.keys():
        direction_attributes = target_orientation.keys()
        init_directions = [init_orientation[direction] for direction in direction_attributes]
        target_directions = [target_orientation[direction] for direction in direction_attributes]
        transform_matrix = generate_rotation_matrix(np.array(init_directions), np.array(target_directions)).tolist()
    else:
        transform_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    print("Result:")
    # if target_position is None:
    #     result = {
    #         'init_position': init_position,
    #         'target_position': None,
    #         'delta_position': None,
    #         'init_orientation': init_orientation,
    #         'target_orientation': target_orientation,
    #         'transform_matrix': transform_matrix
    #     }
    #     return pcd_camera.reshape(-1,3), colored_object_pcd[:,:3], pcd_base.reshape(-1,3), None, transform_matrix
    
    result = {
        'init_position': init_position,
        'target_position': target_position,
        'delta_position': [round(target_position[i] - init_position[i], 2) for i in range(3)],
        'init_orientation': init_orientation,
        'target_orientation': target_orientation,
        'transform_matrix': transform_matrix
    }
    print(result)
    return pcd_camera.reshape(-1,3), pcd_base.reshape(-1,3),  colored_object_pcd[:,:3], obj_pts_base, object_mask, [round(target_position[i] - init_position[i], 2) for i in range(3)], transform_matrix
