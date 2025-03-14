import torch

def convert_gripper(trans: torch.Tensor, rot: torch.Tensor, depth_from: float, depth_to: float):
    # trans: (..., 3), rot: (..., 3, 3)
    return trans + rot[..., 0] * (depth_from - depth_to), rot

def goal_rot_to_gripper_rot(goal_rot: torch.Tensor):
    # goal_rot: (..., 3, 3)
    return torch.stack([goal_rot[..., 2], -goal_rot[..., 1], goal_rot[..., 0]], -1)