import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import json
from libero.libero import benchmark

import wandb
from PIL import Image
# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

@dataclass
class GenerateConfig:
    # fmt: off
    root_dir = "/data/benchmark/inputs"
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b-finetuned-libero-spatial"     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1#50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 0                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    storage_dir = Path(os.path.join(cfg.root_dir, cfg.task_suite_name))


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")

# task id, task_description, depth, rgb, bddl

    # Start evaluation
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # if task_id < 2: continue
        # Get task
        os.makedirs(os.path.join(storage_dir, str(task_id)), exist_ok=True)
        task = task_suite.get_task(task_id)
        task_description = task.language
        print(f"\nTask: {task_description}")

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        resolution = 300
        
        
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env_args = {"bddl_file_name": task_bddl_file, 
                    "camera_heights": resolution, 
                    "camera_widths": resolution,
                    "camera_depths": True,
                    "camera_names": ["agentview","frontview"],
                    "controller": "OSC_POSE",
                    "controller_config_file": "/data/workspace/LIBERO/workspace/controller/no_delta.json"
                    }

        env = OffScreenRenderEnv(**env_args)
        env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
        
        
        # Start episodes
        task_episodes = 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            

            # Reset environment
            env.reset()
            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
           
            # while t<cfg.num_steps_wait+1:
            try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                for _ in range(10):
                    env.step([-3,-3,3,0,0,0,-1])
                for _ in range(30):env.step([-4,0,3,0,0,0,-1])    
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step([-3,-3,3,0,0,0,-1])
                    t += 1
                    img = obs["agentview_image"][::-1]
                    img1 = obs["frontview_image"][::-1]
                    img = Image.fromarray(img)
                    img1 = Image.fromarray(img1)
                    img.save(f'/data/benchmark/test/eval_ours_final.png')
                    img1.save(f'/data/benchmark/test/eval_front.png')

                    # img.save(f'./images/test_script_o_{t}.png')
                    
                # Get preprocessed image
                from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
                sim = env.workers[0].env.sim
                intrinstic_matrix = get_camera_intrinsic_matrix(sim,'agentview',128, 128)
                extrinsic_matrix = get_camera_extrinsic_matrix(sim, 'agentview')

                
                img = obs["agentview_image"][::-1]
                depth = obs['agentview_depth'][::-1]

                # img = obs["birdview_image"][::-1]
                # depth = obs['birdview_depth'][::-1]
                # np.save(f'/data/benchmark/test/depth_{task_id}_{episode_idx}_ours.npy', depth)


                # np.save(f'/data/benchmark/test/rgb_{task_id}_{episode_idx}_ours.npy', img)
           

                # import pdb; pdb.set_trace()
                # img = Image.fromarray(img)
                # img.save(f'/data/benchmark/test/rgb_{task_id}_{episode_idx}_ours.png')
                # img.save('./images/test_script.png')
                # exit(0)
            except Exception as e:
                print(f"Caught exception: {e}")

            
            # import pdb; pdb.set_trace()
            input_data = {
                "task_id": task_id,
                "task_description": task_description,
                "bddl": task_bddl_file
            }

            # store depth and rgb images:
            rgb_img = Image.fromarray(img)
            rgb_img.save(os.path.join(storage_dir, str(task_id), "rgb.png"))
            # store depth as npy:
            np.save(os.path.join(storage_dir, str(task_id), "depth.npy"), depth)
            # store rbg as npy:
            np.save(os.path.join(storage_dir, str(task_id), "rgb.npy"), img)
          
            data_path = os.path.join(storage_dir, str(task_id), "info.json")
            with open (data_path, 'w') as f:
                json.dump(input_data, f)
            task_episodes += 1
            print(task_episodes)
   




    # Save local log file


    # Push total metrics and local log file to wandb



if __name__ == "__main__":
    eval_libero()
