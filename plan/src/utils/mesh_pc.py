import os
import numpy as np
import torch
import trimesh
import trimesh.sample
from typing import List, Tuple
from scipy.spatial import cKDTree

from src.utils.vis_plotly import Vis
# from src.utils.data import get_mesh_name
from src.utils.utils import to_torch

class MeshPC:
    """
        Refine gripper pose by closing it and fix depth
    """
    def __init__(self, N=10000, dist=None, full=10**5):
        assert not (N is None and dist is None)
        assert not (N is not None and dist is not None)
        mesh_path = os.path.join('data', 'meshes')
        self.obj_ids = [id for id in os.listdir(mesh_path) if id.isnumeric()]
        self.fps = dict()
        self.vis = Vis()
        
        pc_dir = os.path.join('data', 'fps', str(N) if dist is None else f'{dist*1000:.1f}mm')
        os.makedirs(pc_dir, exist_ok=True)
        for id in self.obj_ids:
            if os.path.exists(os.path.join(pc_dir, id + '.npy')):
                continue
            mesh = trimesh.load(os.path.join(mesh_path, id, 'nontextured_simplified.ply'))
            if not N is None:
                samples = trimesh.sample.sample_surface_even(mesh, N)[0]
                if len(samples) < N:
                    idxs = np.random.randint(0, len(samples), N - len(samples))
                    samples = np.concatenate([samples, samples[idxs]])
            else:
                K = 1
                full_samples = trimesh.sample.sample_surface_even(mesh, full)[0]
                while len(full_samples) < full:
                    add_samples = trimesh.sample.sample_surface_even(mesh, int(1.3 * (full - len(full_samples))))[0]
                    full_samples = np.concatenate([full_samples, add_samples])[:full]
                
                while True:
                    for _ in range(3 if K < 2048 else 1):
                        samples = trimesh.sample.sample_surface_even(mesh, K)[0]
                        tree = cKDTree(samples)
                        distances, _ = tree.query(full_samples)
                        if distances.max() < dist:
                            break
                    if distances.max() < dist:
                        break
                    K *= 2
            np.save(os.path.join(pc_dir, id + '.npy'), samples.astype(np.float32))

        for id in self.obj_ids:
            self.fps[id] = torch.from_numpy(np.load(os.path.join(pc_dir, id + '.npy'))).float()
    
    def __call__(self, idx) -> torch.Tensor:
        return self.fps[get_mesh_name(idx)]
    
    def get_full_pc(self, objs, obj_num=9999, pc_num=None, with_table=False, separate=False) -> torch.Tensor:
        pcs = dict()
        for i in range(min(len(objs), obj_num)):
            if objs[i]['type'] == 'mesh':
                assert objs[i]['mesh_type'] == 'graspnet'
                obj_idx = objs[i]['graspnet_id']
                if obj_idx == -1:
                    continue
                fps = self.__call__(obj_idx)
                pose = to_torch(objs[i]['pose']).float()
                pcs[objs[i]['name']] = torch.einsum('ab,nb->na', pose[:3, :3], fps) + pose[:3, 3]
            elif objs[i]['type'] == 'box':
                if objs[i]['name'] == 'table' and not with_table:
                    continue
                raise NotImplementedError
        if separate:
            return pcs
        all_pcs = torch.cat(list(pcs.values()), dim=0)
        if not pc_num is None:
            all_pcs = all_pcs[torch.randint(0, len(all_pcs), (pc_num,))]
        return all_pcs