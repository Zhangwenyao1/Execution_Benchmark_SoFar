import os
# if os.environ.get('DEBUG') == '1':
#     import debugpy
#     debugpy.listen(5678)
#     print(f'waiting for debugger to attach...')
#     debugpy.wait_for_client()

import sys
# sys.path.append('libs/octo')

import torch
torch.autograd.set_grad_enabled(False)

import PIL
import io
import zmq
import numpy as np
import logging
from zmq import Poller
from pathlib import Path
import json
import os
import pickle
logging.getLogger().setLevel(logging.INFO)
import time
import debugpy

# if os.environ.get('DEBUG') == '1':
#     debugpy.listen(5681)
#     print(f'waiting for debugger to attach...')
#     debugpy.wait_for_client()



import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str, default=6666)
arg_parser.add_argument("--batching-delay", type=float, default=80)
arg_parser.add_argument("--batch-size", type=int, default=1)
arg_parser.add_argument("--dataset-statistics", type=str, required=False)
arg_parser.add_argument("--path", type=str, required=False)
arg_parser.add_argument("--openloop", action="store_true")
arg_parser.add_argument("--multiview", action="store_true")
arg_parser.add_argument("--top_k", type=int, default=None)
arg_parser.add_argument("--known_proprio", action="store_true")
arg_parser.add_argument("--fast", action="store_true", help="only generate necessary outputs, for speed")


# def normalize(x, stats, normalization_type):
#     assert normalization_type == NormalizationType.BOUNDS_Q99
#     low = np.array(stats["p01"])
#     high = np.array(stats["p99"])
#     mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
#     ret = np.around(np.where(mask, np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1), x), 4)
#     return ret

def sample_transform(sample_batch, train=False):
    return sample_batch
    

def test_batch_process():
    sample_batch = [
        {
            'text': 'Pick up toy large elephant.',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.zeros((7,), dtype=np.float32)],
            'traj_metadata': None,
            'env_id': 1,
        },
        {
            'text': 'pick up the toy rhino and the apple',
            'image_array': [np.ones((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.zeros((7,), dtype=np.float32)],
            'traj_metadata': None,
            'env_id': 2,
        },
        {
            'text': 'pick up the toy rhino and the apple and the car',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.ones((7,), dtype=np.float32)],
            'traj_metadata': None,
            'env_id': 3,
        }
    ]
    batch_process(sample_batch)


import numpy as np
from scipy.spatial.transform import Rotation as R


def compose_poses(a_to_b, b_to_c):
    """
    input: a_to_b([x,y,z,rx,ry,rz,rw]): a在b坐标系下的位姿
           b_to_c([x,y,z,rx,ry,rz,rw]): b在c坐标系下的位姿
    output: a_to_c([x,y,z,rx,ry,rz,rw]): a在c坐标系下的位姿
    """
    # get transition and rotation
    t_ab = np.array(a_to_b[0:3])
    q_ab = np.array(a_to_b[3:7])
    t_bc = np.array(b_to_c[0:3])
    q_bc = np.array(b_to_c[3:7])

    rot_ab = R.from_quat(q_ab)
    rot_bc = R.from_quat(q_bc)

    # rotation
    rot_ac = rot_ab * rot_bc
    q_ac = rot_ac.as_quat()

    # transition
    t_ac = t_ab + rot_ab.apply(t_bc)

    a_to_c = np.concatenate((t_ac, q_ac))
    return a_to_c


def batch_process(data_batch):
    # TODO: openvla does not support batch processing for now
    samples = []
    for data in data_batch:
        if data.get('compressed', False):
            for key in ['image_array', 'image_wrist_array']:
                decompressed_image_array = []
                for compressed_image in data[key]:
                    decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))
                data[key] = decompressed_image_array
            decompressed_depth_array = []
            for compressed_depth in data['depth_array']:
                decompressed_depth_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_depth))).view(np.float32).squeeze(axis=-1))
            data['depth_array'] = decompressed_depth_array

        assert len(data['proprio_array']) == 1, "only support 4 frame of proprio array for now"

        sample = {
            'dataset_name': 'grasp_sim',
            'task': {'language_instruction': bytes(data["text"], 'utf-8')},
            'observation': {
                'image_primary': [data['image_array'][-1].astype(np.uint8)],  # (window, h, w, 3)
                # 'image_wrist': [data['image_wrist_array'][-1].astype(np.uint8)],  # (window, h, w, 3)
                'depth_primary': np.expand_dims([data['depth_array'][-1]], axis=-1),  # (window, h, w, 1)
            },
            'traj_metadata': data['traj_metadata'],
        }
        samples.append(sample_transform(sample, train=False))

    generation_kwargs = dict(do_sample=False) if args.top_k is None else dict(do_sample=True, top_k=args.top_k)
    # 
    # with torch.no_grad():
        # transformation_from_depth_to_plc
        # plc = depth_to_plc(samples[0]['observation']['depth_primary'])
        
        # grasp pose prediction
        # grasp_pose = gsnet(plc)
        
        # goal pose prediction
        # goal_pose = our_model(sample['task']['language_instruction'], color_img, plc)
        
        # prevent the coliision
        # pre_grasp_pose = 
        # pre_goal_pose =
        
        # pre_goal_pose = 
    grasp_pose = [[-0.2, -0.3, 0.5, 1.0, 0, 0, 0]]
    goal_pose = [[-0.5, -0.6, 1.0, 1.0, 0, 0, 0]]
    # for pred_pose, pred_action, pred_proprio, data in zip(pred_poses, pred_actions, pred_proprios, data_batch):
    #     pred_action = interpolate_delta_actions(pred_action, 3)
    #     debugs.append({
    #         'pose': (pred_pose[:3], pred_pose[3:6]),
    #         'proprio': (pred_proprio[:3], pred_proprio[3:6]),
    #         'proprio_gt': (data["proprio_array"][-1][:3], data["proprio_array"][-1][3:6]),
    #     })

    return grasp_pose, goal_pose


if __name__ == "__main__":
    args = arg_parser.parse_args()
    print(args)

    # assert args.known_proprio, "let's keep things simple, please add --known_proprio"

    # a training checkpoint
    # if os.path.isfile(args.path):
        # open6dor = load_vla(args.path, hf_token=None, load_for_training=False).to('cuda').type(torch.bfloat16)
        # sample_transform = img_transformation()
 
    if os.environ.get('DEBUG') == '1':
        test_batch_process()

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")

    poller = Poller()
    poller.register(socket, zmq.POLLIN)

    requests = []
    logging.info('start serving')

    first_arrive_time = None

    while True:
        while True:
            try:
                client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)
                data = pickle.loads(data)

                if len(requests) == 0:
                    first_arrive_time = time.time() * 1000

                requests.append((client_id, data))
                if len(requests) >= args.batch_size:
                    break
            except zmq.Again:
                time.sleep(0.01)
                break
        if len(requests) == 0:
            continue
        current_time = time.time() * 1000
        if len(requests) >= args.batch_size or ((first_arrive_time is not None) and (current_time-first_arrive_time>args.batching_delay) and len(requests)>0):
            data_num = min(args.batch_size, len(requests))
            client_ids, data_batch = zip(*requests[:data_num])

            logging.info(f'processing {len(requests)} requests')
            tbegin = time.time()

            # pad to batch size to avoid jax recompiling
            padding_num = args.batch_size - data_num
            data_batch = list(data_batch)
            data_batch += [data_batch[0]] * padding_num
            grasp_poses, goal_poses = batch_process(data_batch) # for rotation
            
            tend = time.time()
            logging.info(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')

            # remove padding
            grasp_pose = grasp_poses[:data_num]
            goal_pose = goal_poses[:data_num]

            for client_id, grasp_pose, goal_pose, in zip(client_ids, grasp_pose, goal_pose):
                response = {
                    'info': 'success',
                    'grasp_pose': grasp_pose,
                    'goal_pose': goal_pose,
                    
                }
                socket.send_multipart([client_id, b'', pickle.dumps(response)])
            requests = requests
