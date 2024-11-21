import argparse
import sys
import os
import sys
import os
sys.path.append('/data/workspace/LIBERO')
sys.path
# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import datetime
import torch

import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)


from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

from datetime import datetime

import time
import debugpy
debugpy.listen(5679)
debugpy.wait_for_client()
benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
    "6dor_task": "6DOR_TASK", #  add our task
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        # required=True,
        default= "libero_spatial",
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", default=0, type=int) #, required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        # required=True,
        default="base",
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        # required=True,
        default="bc_rnn_policy",
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, default= 100) #, required=True)
    parser.add_argument("--ep", default=50, type=int)
    parser.add_argument("--load_task", default=0, type=int)
    parser.add_argument("--device_id", default=0, type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    cur_time = datetime.now()
    time_str = cur_time.strftime("%Y%m%d%H%M%S")
    args.save_dir = f"experiment_output/{args.experiment_dir}_saved_{time_str}"


    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


def main():
    args = parse_args()
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    experiment_dir = os.path.join(
        args.experiment_dir,
        f"{benchmark_map[args.benchmark]}/"
        + f"{algo_map[args.algo]}/"
        + f"{policy_map[args.policy]}_seed{args.seed}",
    )
    # find the checkpoint
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    if experiment_id == 0:
        print(f"[error] cannot find the checkpoint under {experiment_dir}")
        sys.exit(0)

    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    try:
        if args.algo == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
        else:
            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
    except:
        print(f"[error] cannot find the checkpoint at {str(model_path)}")
        sys.exit(0)
# TODO: 
# cfg 
# {'seed': 100, 'use_wandb': False, 'wandb_project': 'lifelong learning', 'folder': '/data/workspace/LIBERO/libero/libero/../datasets', 'bddl_folder': '/data/workspace/LIBERO/libero/libero/./bddl_files', 'init_states_folder': '/data/workspace/LIBERO/libero/libero/./init_files', 'load_previous_model': False, 'device': 'cuda:0', 'task_embedding_format': 'bert', 'task_embedding_one_hot_offset': 1, 'pretrain': False, 'pretrain_model_path': '', 'benchmark_name': 'libero_spatial', 'data': {'data_modality': [...], 'seq_len': 10, 'frame_stack': 1, 'use_eye_in_hand': True, 'use_gripper': True, 'use_joint': True, 'use_ee': False, 'max_word_len': 25, 'state_dim': None, 'num_kp': 64, 'img_h': 128, 'img_w': 128, 'task_group_size': 1, 'task_order_index': 0, 'shuffle_task': False, 'obs': {...}, 'obs_key_mapping': {...}, 'affine_translate': 4, 'action_scale': 1.0, ...}, 'policy': {'color_aug': {...}, 'translation_aug': {...}, 'image_encoder': {...}, 'language_encoder': {...}, 'policy_head': {...}, 'policy_type': 'BCRNNPolicy', 'image_embed_size': 64, 'text_embed_size': 32, 'rnn_hidden_size': 1024, 'rnn_num_layers': 2, 'rnn_dropout': 0.0, 'rnn_bidirectional': False}, 'train': {'optimizer': {...}, 'scheduler': {...}, 'n_epochs': 50, 'batch_size': 32, 'num_workers': 4, 'grad_clip': 100.0, 'loss_scale': 1.0, 'resume': False, 'resume_path': '', 'debug': False, 'use_augmentation': True}, 'eval': {'load_path': '', 'eval': True, 'batch_size': 64, 'num_workers': 4, 'n_eval': 20, 'eval_every': 5, 'max_steps': 600, 'use_mp': True, 'num_procs': 20, 'save_sim_states': False}, 'lifelong': {'algo': 'Sequential'}, 'experiment_dir': './experiments/libero_spatial/Sequential/BCRNNPolicy_seed100/run_031', 'experiment_name': 'libero_spatial_Sequential_BCRNNPolicy_seed100_run_031', 'shape_meta': {'ac_dim': 7, 'all_shapes': {...}, 'all_obs_keys': [...], 'use_images': True}}

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
                # we never train norm layers
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index) # 'libero_spatial'
    # TODO: add personalized task
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    # get task_embs from language encoder
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        # this returns a SequenceDataset-'dataset' and a dictionary-'shape_meta'
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ), # libero_spatial
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len, # 10
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[args.task_id : args.task_id + 1]
        )
    except:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else: # args.algo : 'base'
        save_folder = os.path.join(
            args.save_dir, # experiment_output/xxxxx
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
    )

    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file 
            ), # '/data/workspace/LIBERO/libero/libero/./bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl'
            "camera_depths": True,
            "camera_heights": cfg.data.img_h, # 128
            "camera_widths": cfg.data.img_w, # 128
            # "camera_segmentations": True#"",
            "controller": "OSC_POSE",
            "controller_config_file": "/data/workspace/LIBERO/workspace/controller/no_delta.json"
        }
        
        env_num = 1#20
        # env = SubprocVectorEnv(
        #     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        # )
        env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)]) #OffScreenRenderEnv(**env_args)
        # TODO: change the function to load our configuration file
        env.reset()

        from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
        # intrinstic_matrix = get_camera_intrinsic_matrix(env.sim,'agentview',128, 128)
        # extrinsic_matrix = get_camera_extrinsic_matrix(env.sim, 'agentview')


        env.seed(cfg.seed)
        algo.reset()
        
        # import pdb; pdb.set_trace()
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        ) # '/data/workspace/LIBERO/libero/libero/./init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init'
        # wenyao: here we should load the init states for each env
        
        # eg:
        # problem_folder: 'libero_spatial', 
        # init_states_file: 'pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.pruned_init'
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices] # an array
        # env.workers[0].sim.data
        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        # env.workers[0].sim.data
        task_emb = benchmark.get_task_emb(args.task_id)

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            env.step(np.zeros((env_num, 7)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                # actions = algo.policy.get_action(data)
                # TODO: change policy forwarding -> our execution
                if (steps // 10) % 2:
                    actions = np.array([[0, 0.5, 1,0,0,0,-1]])
                else:
                    actions = np.array([[0, 0, 1.5,0,0,0,1]])
                # actions = np.array([[0.1, 0, 0,0,0,0,0]])
                # actions = np.array([[0, 0.1, 0,0,0,0,0]])
                # actions = np.array([[0, 0, 0.1,0,0,0,0]])
                # actions = np.array([[0, 0.5, 1,0,0,0,0]])
                # for observable in env._observables.values():
                #     pass
                # import pdb; pdb.set_trace()
                obs, reward, done, info = env.step(actions)

                depth = obs[0]["agentview_depth"]
                rgb = obs[0]["agentview_image"]
                if steps == 0:
                    #np.save('#TODO', depth)
                    pass
            
                
                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print(test_loss, success_rate)

if __name__ == "__main__":
    main()


'''
python libero/lifelong/test.py --benchmark libero_spatial --task_id 0 --algo base --seed 100 --ep 50 --load_task 0 --device_id 0 --save-videos

# --policy bc_rnn_policy

'''