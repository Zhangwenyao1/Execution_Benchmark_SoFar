import os
if os.environ.get('DEBUG') == '1':
    import debugpy
    debugpy.listen(5678)
    print(f'waiting for debugger to attach...')
    debugpy.wait_for_client()

import sys
sys.path.append('libs/octo')

import torch
torch.autograd.set_grad_enabled(False)
from transformers import AutoProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.datasets.graspsim_dataset import RLDSBatchTransform
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.models import load_vla
from prismatic.util.data_utils import PaddedCollatorForActionPrediction

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
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str)
arg_parser.add_argument("--batching-delay", type=float, default=80)
arg_parser.add_argument("--batch-size", type=int, default=1)
arg_parser.add_argument("--dataset-statistics", type=str, required=True)
arg_parser.add_argument("--path", type=str, required=True)
arg_parser.add_argument("--openloop", action="store_true")
arg_parser.add_argument("--multiview", action="store_true")
arg_parser.add_argument("--top_k", type=int, default=None)
arg_parser.add_argument("--known_proprio", action="store_true")
arg_parser.add_argument("--fast", action="store_true", help="only generate necessary outputs, for speed")


def normalize(x, stats, normalization_type):
    assert normalization_type == NormalizationType.BOUNDS_Q99
    low = np.array(stats["p01"])
    high = np.array(stats["p99"])
    mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
    ret = np.around(np.where(mask, np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1), x), 4)
    return ret


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


def interpolate_delta_actions(delta_actions, n):
    """
    Interpolate m delta_actions to m*n delta_actions.

    actions: list of actions, each action is (delta x, delta y, delta z, delta roll, delta pitch, delta yaw, gripper open/close).
    """
    import transforms3d as t3d
    ret = []
    for delta_action in delta_actions:
        xyzs = 1 / n * np.array([delta_action[:3]]*n)
        axangle_ax, axangle_angle = t3d.euler.euler2axangle(*delta_action[3:6])
        eulers = [t3d.euler.axangle2euler(axangle_ax, axangle_angle / n)]*n
        grippers = np.array([[0.]] * (n-1) + [[delta_action[-1]]])  # 0 for no change of gripper state
        ret.extend(np.concatenate([xyzs, eulers, grippers], axis=-1))
    return ret


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

        assert len(data['proprio_array']) == 4, "only support 4 frame of proprio array for now"

        sample = {
            'dataset_name': 'grasp_sim',
            'task': {'language_instruction': bytes(data["text"], 'utf-8')},
            'observation': {
                'image_primary': [data['image_array'][-1].astype(np.uint8)],  # (window, h, w, 3)
                'image_wrist': [data['image_wrist_array'][-1].astype(np.uint8)],  # (window, h, w, 3)
                'depth_primary': np.expand_dims([data['depth_array'][-1]], axis=-1),  # (window, h, w, 1)
                'proprio': [normalize(d, vla.get_pose_stats(args.dataset_statistics), NormalizationType.BOUNDS_Q99) for d in data["proprio_array"]], # (window, dim)
            },
            'traj_metadata': data['traj_metadata'],
        }
        samples.append(sample_transform(sample, train=False))

    generation_kwargs = dict(do_sample=False) if args.top_k is None else dict(do_sample=True, top_k=args.top_k)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_poses, pred_actions, pred_proprios = vla.predict_action(
            input_ids=[sample['input_ids'] for sample in samples],
            pixel_values=[sample['pixel_values'] for sample in samples],
            proprio_ids_with_special_token=[sample.get('proprio_ids_with_special_token') for sample in samples],
            unnorm_key=args.dataset_statistics, fast=args.fast, **generation_kwargs)

    actions = []
    debugs = []
    for pred_pose, pred_action, pred_proprio, data in zip(pred_poses, pred_actions, pred_proprios, data_batch):
        pred_action = interpolate_delta_actions(pred_action, 3)
        actions.append(pred_pose if args.openloop else pred_action)
        debugs.append({
            'pose': (pred_pose[:3], pred_pose[3:6]),
            'proprio': (pred_proprio[:3], pred_proprio[3:6]),
            'proprio_gt': (data["proprio_array"][-1][:3], data["proprio_array"][-1][3:6]),
        })

    return actions, debugs, [data['env_id'] for data in data_batch]


if __name__ == "__main__":
    args = arg_parser.parse_args()
    print(args)

    assert args.known_proprio, "let's keep things simple, please add --known_proprio"

    # a training checkpoint
    if os.path.isfile(args.path):
        vla = load_vla(args.path, hf_token=None, load_for_training=False).to('cuda').type(torch.bfloat16)
        sample_transform = RLDSBatchTransform(
            ActionTokenizer(vla.llm_backbone.get_tokenizer()),
            vla.llm_backbone.get_tokenizer(),
            image_transform=vla.vision_backbone.get_image_transform().__call__,
            prompt_builder_fn=PurePromptBuilder,
            train=False,
            multiview=args.multiview,
            known_proprio=args.known_proprio,
        )
    # or a huggingface exported model (a directory for the step)
    else:
        processor = AutoProcessor.from_pretrained(args.path, trust_remote_code=True)
        sample_transform = RLDSBatchTransform(
            ActionTokenizer(processor.tokenizer),
            processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=PurePromptBuilder,
            train=False,
            multiview=args.multiview,
            known_proprio=args.known_proprio,
        )
        vla = OpenVLAForActionPrediction.from_pretrained(
            args.path,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        # FIXME: a dirty fix
        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="left",
        )
        vla.collator = collator
        parent_path = os.path.dirname(os.path.dirname(args.path))
        if os.path.isdir(parent_path) and os.path.isfile(f"{parent_path}/dataset_statistics.json") :
            with open(Path(parent_path) / "dataset_statistics.json", "r") as f:
                vla.norm_stats.update(json.load(f))

    vla = torch.compile(vla)
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
                break

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
            actions, debugs, env_ids = batch_process(data_batch)
            tend = time.time()
            logging.info(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')

            # remove padding
            actions = actions[:data_num]
            env_ids = env_ids[:data_num]

            for client_id, action, debug, env_id in zip(client_ids, actions, debugs, env_ids):
                response = {
                    'info': 'success',
                    'env_id': env_id,
                    'action': action,
                    'debug': debug,
                }
                socket.send_multipart([client_id, b'', pickle.dumps(response)])
            requests = requests[data_num:]
