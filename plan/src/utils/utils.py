import sys
import os
import torch
import numpy as np
import random
from typing import Union, Optional
from datetime import datetime
import uuid
from multiprocessing import Pool
from pytorch3d import transforms as pttf
import scipy.spatial.transform as sst
from scipy.spatial import cKDTree
from transforms3d.quaternions import quat2mat, mat2quat
# from src.utils.vis_plotly import Vis

from src.utils.constants import GRIPPER_HALF_WIDTH, FRANKA_NEUTRAL_QPOS, FRANKA_JOINT_LIMITS, ROBOT_JOINTS

def to_list(x: Union[torch.Tensor, np.ndarray, list], spec='cpu') -> list:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    elif isinstance(x, np.ndarray):
        return x.tolist()
    elif isinstance(x, list):
        return x
    else:
        raise ValueError(f'Unsupported type {type(x)}')

def to_torch(x: Union[torch.Tensor, np.ndarray, list], spec='cpu') -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(spec)
    elif isinstance(x, torch.Tensor):
        return x.to(spec)
    elif isinstance(x, list):
        return torch.tensor(x).to(spec)
    else:
        raise ValueError(f'Unsupported type {type(x)}')

def to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.array(x)
    else:
        raise ValueError(f'Unsupported type {type(x)}')

def to_number(x: Union[torch.Tensor, np.ndarray, float, int]) -> float:
    if isinstance(x, (torch.Tensor, np.ndarray)):
        return x.item()
    return x

def torch_dict_to_device(dic, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dic.items()}

def transform_pc(pose: Union[torch.Tensor, np.ndarray], pc: Union[torch.Tensor, np.ndarray], inv: bool = False):
    # pose: ([B]*, 4, 4), pc: ([B]*, N, 3) -> result ([B]*, N, 3)
    orig_shape = pose.shape[:-2]
    pose, pc = pose.reshape(-1, 4, 4), pc.reshape(-1, pc.shape[-2], 3)
    einsum = np.einsum if isinstance(pose, np.ndarray) else torch.einsum
    if inv:
        result = einsum('bji,bni->bnj', pose[:, :3, :3], pc - pose[:, :3, 3][:, None])
    else:
        result = einsum('bij,bnj->bni', pose[:, :3, :3], pc) + pose[:, :3, 3][:, None]
    return result.reshape(*orig_shape, -1, 3)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def to_voxel_center(pc: torch.Tensor, voxel_size: float):
    """calculate the center of voxel corresponding to each point

    Args:
        pc (torch.Tensor): (..., 3)
    returns:
        voxel_center (torch.Tensor): (..., 3)
    """
    return torch.div(pc, voxel_size, rounding_mode='floor') * voxel_size + voxel_size / 2

def proper_svd(rot: torch.Tensor):
    """
    compute proper svd of rotation matrix
    rot: (B, 3, 3)
    return rotation matrix (B, 3, 3) with det = 1
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum('bij,bkj->bik', u, v)))
        diag = torch.stack([torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1)
        diag = torch.diag_embed(diag)
    return torch.einsum('bij,bjk,blk->bil', u, diag, v).to(rot.dtype)

def pack_17dgrasp(rot: torch.Tensor, # (N, 3, 3)
                trans: torch.Tensor, # (N, 3)
                width: torch.Tensor, # (N,)
                depth: torch.Tensor, # (N,)
                score: Optional[torch.Tensor] = None # (N,)
                ) -> np.ndarray:
    if score is None:
        score = torch.zeros_like(width)
    return torch.cat([score[:, None], width[:, None], torch.full_like(width[:, None], GRIPPER_HEIGHT), depth[:, None], rot.reshape(-1, 9), trans, torch.full_like(width[:, None], -1)], dim=-1).cpu().numpy()

def unpack_17dgrasp(grasp: np.ndarray):
    grasp = torch.from_numpy(grasp)
    return grasp[:, -13:-4].reshape(-1, 3, 3), grasp[:, -4:-1], grasp[:, 1], grasp[:, 3], grasp[:, 0]

def width_to_gripper_qpos(width: Union[np.ndarray, torch.Tensor]):
    return (GRIPPER_HALF_WIDTH - (width / 2)).clip(min=0.0, max=0.04)

def gripper_qpos_to_width(qpos: Union[np.ndarray, torch.Tensor]):
    return (GRIPPER_HALF_WIDTH - qpos) * 2

def to_urdf_qpos(x: np.ndarray):
    return x
    return np.concatenate([x[..., :-2], GRIPPER_HALF_WIDTH - x[..., -2:]], -1)

def from_urdf_qpos(x: np.ndarray):
    return to_urdf_qpos(x)

def silent_call(func, *args, **kwargs):
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    try:

        result = func(*args, **kwargs)
    except Exception as e:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(e)
        raise e
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    return result

def depth_img_to_xyz(depth_img: np.ndarray, intrinsics: np.ndarray):
    """
    depth_img: (H, W)
    intrinsics: (3, 3)
    return: (H, W, 3)
    """
    H, W = depth_img.shape
    x = np.arange(W)
    y = np.arange(H)
    x, y = np.meshgrid(x, y)
    x = (x - intrinsics[0, 2]) * depth_img / intrinsics[0, 0]
    y = (y - intrinsics[1, 2]) * depth_img / intrinsics[1, 1]
    return np.stack([x, y, depth_img], -1)

def get_time_str():
    return datetime.now().strftime('%Y%m%d%H%M%S')

def sample_sphere(center: np.ndarray, radius: float, n: int = 1):
    theta = np.random.rand(n) * np.pi * 2
    r = radius * np.sqrt(np.random.rand(n))
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1) * r[:, None] + center

def gen_uuid():
    return str(uuid.uuid4())

def inf_generator():
    while True:
        yield

def pool_process(num, func, args):
    pool = Pool(processes=num)
    results = pool.map(func, args)
    pool.close()
    pool.join()
    # results = pool.starmap(func, args)
    return results

def matrix_to_axis_angle(rot: torch.Tensor):
    aa = pttf.matrix_to_axis_angle(rot.reshape(-1, 3, 3))
    aa_norm = aa.norm(dim=-1, keepdim=True)
    aa = torch.where(aa_norm < np.pi, aa, aa / aa_norm * (aa_norm - 2 * np.pi))
    return aa.reshape(*rot.shape[:-2], 3)


def get_random_init_eef(num=1, r=0.25, h=0.2):
    """
    Sample in a gaussian cylinder.
    """
    # EEF_OFFSET = 0.120
    thetas = np.random.uniform(0, 2*np.pi, num)
    rs = np.abs(np.random.normal(0, r, num))
    heights = np.abs(np.random.normal(0, h, num))
    eef_xyzs = np.stack([np.cos(thetas), np.sin(thetas), np.zeros(num)], axis=1) * rs[:, None] + [0.5, 0., 0.]
    eef_xyzs[:, 2] += heights + 0.01 # leave some space from the table
    hand_xyzs = eef_xyzs #+ [0., 0., EEF_OFFSET]  # roughly
    roll = np.repeat(np.pi, num) + np.random.uniform(-np.pi/2, np.pi/2, num)  # should be pi if using omniverse franka model, and 0 if using shenzhen's franka model
    pitch = np.repeat(0, num) + np.random.uniform(-np.pi/2, np.pi/2, num)
    yaw = np.random.uniform(0, 2*np.pi, num)
    quats = sst.Rotation.from_euler('xyz', np.stack([roll, pitch, yaw], axis=1)).as_quat()
    # change x y z w to w x y z
    quats = np.roll(quats, 1, axis=1)
    return hand_xyzs, quats

def get_random_init_qpos(ik, r=0.25, h=0.2):
    while True:
        trans, quat = get_random_init_eef(1, r, h)
        try:
            qpos = ik.ik(trans[0], quat2mat(quat[0]), silent=True)
            return qpos
        except:
            continue

def get_random_init_qpos_neutral(std=0.25):
    new_pose = FRANKA_NEUTRAL_QPOS[:7] + np.random.normal(0, std, (7,))
    new_pose = np.clip(new_pose, FRANKA_JOINT_LIMITS[:7, 0], FRANKA_JOINT_LIMITS[:7, 1])
    return new_pose

def coll(robot_model, mesh_pc, state, table_thresh=0.0025, obj_thresh=0.0025, robot_thresh=0.01):
    raise DeprecationWarning('This function is deprecated. Use Scene.check_coll instead.')
    obj_pc = mesh_pc.get_full_pc(state['objs'], to_torch(state['obj_pose']).float())
    qpos = to_torch(state['qpos'][None]).float()
    qpos = {k: qpos[:, i] for i, k in enumerate(ROBOT_JOINTS)}
    robot_pc, link_trans, link_rot = robot_model.sample_surface_points_full(qpos, n_points_each_link=2**15, with_fk=True)
    robot_pc = robot_pc[0]
    robot_pc = robot_pc[~((robot_pc[:, 2] < table_thresh * 2) & (robot_pc[:, 0] < 0.1) & (robot_pc[:, 0] > -0.2) & (robot_pc[:, 1] < 0.1) & (robot_pc[:, 1] > -0.1))]

    # Vis.show(Vis.pc_plotly(obj_pc, color='blue') + Vis.pc_plotly(robot_pc, color='red'))

    if robot_pc[:, 2].min() < table_thresh: 
        return True
    if not obj_thresh is None:
        for k, v in FRANKA_ADDITIONAL_BOXES.items():
            box = to_torch(v)
            link_trans_k, link_rot_k = link_trans[k][0], link_rot[k][0]
            obj_pc_in_link = torch.einsum('ba,nb->na', link_rot_k, obj_pc - link_trans_k)
            clamped = torch.clamp(obj_pc_in_link, min=box[0], max=box[1])
            dist = torch.norm(obj_pc_in_link - clamped, dim=-1)
            if dist.min() < obj_thresh:
                return True
    if not robot_thresh is None:
        for i in range(3):
            obj_min, obj_max = obj_pc[:, i].min(), obj_pc[:, i].max()
            robot_pc = robot_pc[(robot_pc[:, i] > obj_min - 1.5 * robot_thresh) & (robot_pc[:, i] < obj_max + 1.5 * robot_thresh)]
            if len(robot_pc) == 0:
                return False
        cdist = torch.cdist(robot_pc, obj_pc)
        if cdist.min() < robot_thresh:
            return True
    return False

def cdist_test(pc1, pc2, thresh, filter=True):
    if filter:
        pc1_min, pc1_max = pc1.min(dim=0).values, pc1.max(dim=0).values
        for i in range(3):
            pc2 = pc2[(pc2[:, i] > pc1_min[i] - 1.2 * thresh) & (pc2[:, i] < pc1_max[i] + 1.2 * thresh)]
            if len(pc2) == 0:
                return False
        pc2_min, pc2_max = pc2.min(dim=0).values, pc2.max(dim=0).values
        for i in range(3):
            pc1 = pc1[(pc1[:, i] > pc2_min[i] - 1.2 * thresh) & (pc1[:, i] < pc2_max[i] + 1.2 * thresh)]
            if len(pc1) == 0:
                return False
    # cdist = torch.cdist(pc1, pc2)
    # return cdist.min() < thresh
    if len(pc2) > len(pc1):
        pc1, pc2 = pc2, pc1
    tree = cKDTree(pc2)
    distances, _ = tree.query(pc1, k=1)  # Find the nearest point in pc2 for each point in pc1
    return np.min(distances) < thresh