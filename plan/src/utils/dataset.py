import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

from PIL import Image
import scipy.io as scio
import random
import collections.abc as container_abcs
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
try:
    import MinkowskiEngine as ME
except:
    pass
from pytorch3d import transforms as pttf
from transforms3d.axangles import mat2axangle
from transforms3d.quaternions import quat2mat
from typing import Optional
from pprint import pprint

from src.utils.config import DotDict, to_dot_dict
from src.utils.vis_plotly import Vis
from src.utils.data import get_traj_list, get_traj, read_traj, get_mesh_file, State, get_mesh_name
from src.utils.constants import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_INTRINSICS, MAX_OBJ_NUM
from src.utils.render import Renderer

def get_sparse_tensor(pc: torch.tensor, voxel_size: float, labels: Optional[torch.tensor] = None):
    """
        pc: (B, N, 3)
        return dict(point_clouds, coors, feats, quantize2original)
    """
    coors = pc / voxel_size
    feats = np.concatenate([np.ones_like(pc), pc], axis=-1)
    coordinates_batch, features_batch = ME.utils.sparse_collate([coor for coor in coors], [feat for feat in feats])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch, return_index=True, return_inverse=True)
    if labels is not None:
        features_batch[:] = 0
        labels = labels.reshape(-1, labels.shape[-1])
        for idx in range(labels.shape[-1]):
            features_batch[quantize2original[labels[:, idx] != 0], idx] = 1
    return dict(point_clouds=pc, coors=coordinates_batch, feats=features_batch, quantize2original=quantize2original)

class Loader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            data = next(self.iter)
        return data

class MixDataset(Dataset):
    def __init__(self, datasets, weights):
        for i, dataset in enumerate(datasets):
            if dataset.__len__() == 0:
                weights[i] = 0
        self.datasets = datasets
        self.weights = weights
    
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # Randomly choose a dataset based on the weights
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights, k=1)[0]
        # Randomly select an item from the chosen dataset
        item_idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
        return self.datasets[dataset_idx].__getitem__(item_idx)

class TrajDataset(Dataset):
    def __init__(self, config, split: str):
        self.traj_list = get_traj_list(root=config.root, mode=split)
        # self.traj_list = ["0019/9032513d-27a9-4320-ae85-5d02a97e4791.npz"]#get_traj_list(mode=split)
        self.config = config
        self.split = split
        # self.split = ''
        self.vis = Vis()
    
    def __len__(self):
        return len(self.traj_list)
    
    def __getitem__(self, idx):
        traj = get_traj(self.traj_list[idx % len(self.traj_list)], root=self.config.root, mode=self.split)
        state = read_traj(traj, 0)
        state['eef'] = np.eye(4).astype(np.float32)
        state['eef'][:3, 3] = state['trans']
        state['eef'][:3, :3] = state['rot']
        state['history'] = np.concatenate([state['trans'], state['rot'].reshape(-1), state['qpos']])[None].repeat(self.config.common.len_history + 1, 0)
        pad_obj_num = MAX_OBJ_NUM-len(state['obj_idx'])

        cur_state = dict(
            cur_qpos=state['qpos'].astype(np.float32),
            obj_num=len(state['obj_idx']),
            obj_idx=np.concatenate([state['obj_idx'], np.full((pad_obj_num,), -1)]),
            obj_pose=np.concatenate([state['obj_pose'], np.full((pad_obj_num, 4, 4), -1)]).astype(np.float32),
            extrinsics=state['extrinsics'].astype(np.float32),
            cur_eef=state['eef'],
            final_trans=state['final_trans'].astype(np.float32),
            final_rot=state['final_rot'].astype(np.float32),
            final_width=state['final_width'].astype(np.float32),
            history=state['history'].astype(np.float32),
            grasp_obj_id=state['grasp_obj_id'],
        )
        return cur_state

class CloseLoopDataset(Dataset):
    def __init__(self, config, split: str, direct = None):
        self.direct = not direct is None
        if self.direct:
            self.traj_list = direct
        else:
            self.traj_list = get_traj_list(root=config.root, mode=split)
        random.shuffle(self.traj_list)
        self.traj_list = self.traj_list[::config.frac]
        self.config = config
        self.split = split
        self.vis = Vis()

    def __len__(self):
        length = len(self.traj_list) 
        if length < 1000 and not self.direct:
            length *= int(100000 / length)
        return length
    
    def __getitem__(self, idx: Optional[int] = None, frame: Optional[int] = None):
        try:
            if idx is None:
                idx = random.randint(0, len(self.traj_list) - 1)
            if self.direct:
                traj = self.traj_list[idx % len(self.traj_list)]
            else:
                traj = get_traj(self.traj_list[idx % len(self.traj_list)], root=self.config.root, mode=self.split)
            state = traj['state']
            if frame is None:
                # frame = random.randint(1, (state<=2).sum() - 1)
                frame = random.randint(0, len(state) - 1) 
            cur_state = read_traj(traj, frame, binary=self.config.binary)

            cur_eef = np.eye(4)
            cur_eef[:3, :3] = cur_state['rot']
            cur_eef[:3, 3] = cur_state['trans']

            # next_qpos = []
            # for f in range(frame, frame + self.config.common.len_pred):
            #     next_state = read_traj(traj, min(f, len(state) - 1), binary=self.config.binary)
            #     next_qpos.append(next_state['action'])
            # next_qpos = np.stack(next_qpos)

            next_states = [read_traj(traj, min(f, len(state) - 1), binary=self.config.binary) for f in range(frame, frame + self.config.common.len_pred + 1)]
            delta_trans, delta_rot, next_qpos = [], [], []
            if self.config.common.delta_type == 'robot':
                for i in range(self.config.common.len_pred):
                    next_qpos.append(next_states[i]['action'])
                    next_eef = np.eye(4)
                    next_eef[:3, :3] = next_states[i+1]['rot']
                    next_eef[:3, 3] = next_states[i+1]['trans']
                    delta_eef = np.linalg.inv(cur_eef) @ next_eef
                    delta_trans.append(delta_eef[:3, 3])
                    delta_rot.append(delta_eef[:3, :3])
            else:
                for i in range(self.config.common.len_pred):
                    next_qpos.append(next_states[i]['action'])
                    delta_trans.append(next_states[i+1]['trans']-cur_state['trans'])
                    delta_rot.append(next_states[i+1]['rot'] @ cur_state['rot'].T)
            delta_trans, delta_rot, next_qpos = np.stack(delta_trans), np.stack(delta_rot), np.stack(next_qpos)
            delta_qpos = (next_qpos - cur_state['qpos']) / (np.arange(self.config.common.len_pred) + 1)[:, None]

            history = []
            for i in range(self.config.common.len_history):
                his_state = read_traj(traj, max(0, frame-i-1), binary=self.config.binary)
                history.append(np.concatenate([his_state[k].reshape(-1) for k in ['trans', 'rot', 'qpos']]))
            history = np.stack(history) if self.config.common.len_history else np.zeros((0, 3+9+9))


            pad_obj_num = MAX_OBJ_NUM-len(cur_state['obj_idx'])
            assert pad_obj_num >= 0

            result = dict(
                traj_idx=idx,
                frame_idx=frame,
                extrinsics=cur_state['extrinsics'],
                obj_num=len(cur_state['obj_idx']),
                obj_idx=np.concatenate([cur_state['obj_idx'], np.full((pad_obj_num,), -1)]),
                obj_pose=np.concatenate([cur_state['obj_pose'], np.full((pad_obj_num, 4, 4), -1)]),
                cur_qpos=cur_state['qpos'].astype(np.float32),
                next_qpos=next_qpos.astype(np.float32),
                delta_qpos=delta_qpos.astype(np.float32),
                final_trans=cur_state['final_trans'],
                final_rot=cur_state['final_rot'],
                final_width=cur_state['final_width'],
                cur_eef=cur_eef.astype(np.float32),
                history=history.astype(np.float32),
                delta_trans=delta_trans.astype(np.float32),
                delta_rot=delta_rot.astype(np.float32),
            )
            return result
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.traj_list) - 1))
    
    def visualize(self, idx: Optional[int] = None, frame: Optional[dict] = None):
        data = self.__getitem__(idx=idx, frame=frame)
        pprint(data)
        plotly_list = []
        plotly_list += self.vis.pose_plotly(data['cur_eef'][:3, 3], data['cur_eef'][:3, :3])
        plotly_list += self.vis.pose_plotly(data['next_eef'][:3, 3], data['next_eef'][:3, :3], width=2, length=0.15)
        plotly_list += self.vis.pose_plotly(data['final_trans'], data['final_rot'], width=10)
        plotly_list += self.vis.plane_plotly(np.array([0, 0, 1, 0.]), color='orange')
        plotly_list += self.vis.robot_plotly(qpos=torch.from_numpy(data['cur_qpos'][None]), mesh_type='collision', color='violet', opacity=0.5)
        plotly_list += self.vis.robot_plotly(qpos=torch.from_numpy(data['next_qpos'][None]), mesh_type='collision', color='brown', opacity=0.5)
        for obj_idx, pose in zip(data['obj_idx'], data['obj_pose']):
            if obj_idx != -1:
                plotly_list += self.vis.mesh_plotly(path=get_mesh_file(obj_idx), rot=pose[:3, :3], trans=pose[:3, 3])
        # self.vis.show(plotly_list)
        return data, plotly_list

if __name__ == "__main__":
    config = to_dot_dict({"data": {"point_num": 20000}, "common": {"len_history": 1}})
    dataset = CloseLoopDataset(config, 'train')
    renderer = Renderer(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_INTRINSICS, 0)
    data, p = dataset.visualize()
    pc = renderer.render_full(torch.from_numpy(data['cur_qpos'][None]), [data['obj_idx']], [data['obj_pose']], [data['extrinsics']], config.data.point_num, [data['obj_num']])
    pcp = Vis.pc_plotly(pc[0, :, :3])
    Vis.show(p + pcp)
    # datas = []
    # for _ in range(100):
    #     datas.append(dataset.__getitem__())
    # datas = {k: np.array([data[k] for data in datas]) for k in datas[0].keys()}
    # print(datas['final_width'])