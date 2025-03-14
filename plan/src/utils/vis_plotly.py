import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# sys.path.append(os.path.realpath('.'))

from typing import Optional, Union, Dict, List
import trimesh as tm
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.io as scio
import cv2
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
# from graspnetAPI.utils.xmlhandler import xmlReader
# from graspnetAPI.utils.utils import parse_posevector
import plotly.express as px
import random

from src.utils.robot_model import RobotModel
# from src.utils.data import get_mesh_file, get_traj, read_traj
from src.utils.constants import ROBOT_URDF, ROBOT_JOINTS_NOFINGER
from src.utils.utils import to_numpy, to_torch, to_urdf_qpos, from_urdf_qpos, to_number
# from src.utils.pin_model import PinRobotModel

class Vis:
    def __init__(self, urdf_path: str = ROBOT_URDF):
        self.robot = RobotModel(urdf_path)
        # self.robot = PinRobotModel(urdf_path)
        self.robot_joints = self.robot.joint_names

    @staticmethod
    def save_video(array, filename, fps=12, silent=False):
        # Check the shape of the array
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        T, H, W, C = array.shape
        assert C == 3, "The last dimension must be 3 (RGB channels)"

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4
        out = cv2.VideoWriter(filename, fourcc, fps, (W, H))

        for i in range(T):
            # Write the frame to the video
            frame = array[i]
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR

        # Release the VideoWriter object
        out.release()
        if not silent:
            print('video saved')

    @staticmethod
    def plot_fig(path: str = 'tmp/vis.png'):
        plt.legend()
        plt.show()
        plt.savefig(path)
        plt.clf()
        print('saved')

    @staticmethod
    def plot_curve(data: Union[np.ndarray, torch.tensor], # ([n], d)
                   norm: bool = False,
    ):
        data = to_numpy(data).reshape(-1, data.shape[-1])
        if norm:
            data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        for i in range(data.shape[-1]):
            plt.plot(data[:, i], label=f'Series {i+1}')

    @staticmethod
    def rand_color():
        return random.choice(px.colors.sequential.Plasma)
    
    @staticmethod
    def sphere_plotly(trans: Optional[Union[np.ndarray, torch.tensor]] = None,
                      radius: float = None,
                      opacity: float = None,
                      color: str = None,
        ) -> list:
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        radius = 0.1 if radius is None else to_number(radius)
        
        # From https://nbviewer.org/gist/theengineear/a423ab25eb1b22367111

        theta = np.linspace(0,2*np.pi,100)
        phi = np.linspace(0,np.pi,100)
        x = np.outer(np.cos(theta),np.sin(phi)) * radius + trans[0]
        y = np.outer(np.sin(theta),np.sin(phi)) * radius + trans[1]
        z = np.outer(np.ones(100),np.cos(phi)) * radius + trans[2]

        return [
            go.Surface(x=x, y=y, z=z, surfacecolor=x*0, colorscale=[[0, color], [1, color]], opacity=opacity)
        ]
    
    @staticmethod
    def box_plotly(scale: Union[np.ndarray, torch.tensor], # (3, )
                   trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                   rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                   opacity: Optional[float] = None,
                   color: Optional[str] = None,
        ) -> list:

        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        trans = np.zeros(3) if trans is None else trans
        rot = np.eye(3) if rot is None else rot

        # 8 vertices of a cube
        corner = np.array([[0, 0, 1, 1, 0, 0, 1, 1],
                           [0, 1, 1, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 1, 1, 1, 1]]).T - 0.5
        corner *= to_numpy(scale)
        corner = np.einsum('ij,kj->ki', to_numpy(rot), corner) + to_numpy(trans)
        
        return [go.Mesh3d(
            x = corner[:, 0],
            y = corner[:, 1],
            z = corner[:, 2],
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color=color,
            opacity=opacity,
        )]
    
    @staticmethod
    def pose_plotly(trans: Union[np.ndarray, torch.tensor], # (3, )
                    rot: Union[np.ndarray, torch.tensor], # (3, 3)
                    width: int = 5,
                    length: float = 0.1,
    ) -> list:
        result = []
        for i, color in zip(range(3), ['red', 'green', 'blue']):
            result += Vis.line_plotly(trans, trans + rot[:, i] * length, width=width, color=color)
        return result

    @staticmethod
    def plane_plotly(plane_vec: Union[np.ndarray, torch.tensor], # (4, )
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
    ) -> list:

        plane_vec = to_torch(plane_vec)
        color = 'blue' if color is None else color
        opacity = 1.0 if opacity is None else opacity

        dir = plane_vec[:3]
        assert (torch.linalg.norm(dir) - 1).abs() < 1e-4
        center = dir * -plane_vec[3]
        z_axis = dir
        x_axis = torch.zeros(3)
        x_axis[dir.abs().argmin()] = 1
        x_axis = x_axis - (x_axis * z_axis).sum() * z_axis
        x_axis = x_axis / torch.linalg.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)

        return Vis.box_plotly(np.array([2,2,0]), center, rot, color=color)
    
    def robot_plotly(self,
                     trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3)
                     rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                     qpos: Optional[Union[Union[np.ndarray, torch.tensor]]] = None, # (n)
                     opacity: Optional[float] = None,
                     color: Optional[str] = None,
                     mesh_type: str = 'collision',
    ) -> list:
        trans = np.zeros((3,)) if trans is None else to_numpy(trans).reshape(3)
        rot = np.eye(3) if rot is None else to_numpy(rot).reshape(3, 3)
        qpos = torch.zeros((len(self.robot_joints),)) if qpos is None else to_torch(qpos).float().reshape(-1)
        color = 'violet' if color is None else color
        opacity = 1.0 if opacity is None else opacity
        print(qpos.shape)
        print(self.robot_joints)
        print(len(self.robot_joints))
        if type(qpos) == torch.Tensor:
            qpos = qpos[None]
            qpos = {joint:qpos[:, i] for i, joint in enumerate(self.robot.movable_joint_names)}
        link_trans, link_rot = self.robot.forward_kinematics(qpos)
        plotly_data = []
        for n in link_trans.keys():
        # for mesh, (mesh_trans, mesh_rot) in zip(self.robot.meshes[mesh_type], poses):
            # if not hasattr(mesh, 'vertices'):
                # mesh = mesh.to_mesh()
            # vertices, faces = mesh.vertices, mesh.faces
            mesh_trans, mesh_rot = link_trans[n].numpy()[0], link_rot[n].numpy()[0]
            vertices, faces = self.robot.get_link_mesh(n, mesh_type=mesh_type)
            if vertices is None:
                continue
            plotly_data += self.mesh_plotly(vertices=vertices, faces=faces, trans=rot@mesh_trans+trans, rot=rot@mesh_rot, opacity=opacity, color=color)
        return plotly_data
        
    @staticmethod
    def gripper_plotly(trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (1, 3)
            rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (1, 3, 3)
            qpos: Optional[Union[Union[np.ndarray, torch.tensor]]] = None, # (1, n) or Dict[str, (1,)]
            opacity: Optional[float] = None,
            color: Optional[str] = None,
    ) -> list:
        raise NotImplementedError
        trans = np.zeros((1, 3)) if trans is None else trans
        rot = np.eye(3)[None] if rot is None else rot
        qpos = np.array(GRIPPER_MAX_WIDTH / 2, GRIPPER_NEW_DEPTH) if qpos is None else qpos

        # changed for better visualization
        height = 0.002 # GRIPPER_HEIGHT
        finger_width = 0.002 # GRIPPER_FINGER_WIDTH
        tail_length = GRIPPER_TAIL_LENGTH
        depth_base = GRIPPER_DEPTH_BASE
        width, depth = qpos[0]
        """
        4 boxes: 
                    2|------ 1
            --------|  . O
                3   |------ 0

                                    y
                                    | 
                                    O--x
                                    /
                                    z
        """
        centers = torch.tensor([[(depth - finger_width - depth_base)/2, (width + finger_width)/2, 0],
                                [(depth - finger_width - depth_base)/2, -(width + finger_width)/2, 0],
                                [-depth_base-finger_width/2, 0, 0],
                                [-depth_base-finger_width-tail_length/2, 0, 0]])
        scales = torch.tensor([[finger_width+depth_base+depth, finger_width, height],
                                [finger_width+depth_base+depth, finger_width, height],
                                [finger_width, width, height],
                                [tail_length, finger_width, height]])
        centers = torch.einsum('ij,kj->ki', rot[0], centers) + trans[0]
        box_plotly_list = []
        for i in range(4):
            box_plotly_list += Vis.box_plotly(scales[i].numpy(), centers[i].numpy(), rot[0].numpy(), opacity, color)
        return box_plotly_list
    
    @staticmethod
    def pc_plotly(pc: Union[np.ndarray, torch.tensor], # (n, 3)
                  value: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, )
                  size: int = 1,
                  color: Union[str, Union[np.ndarray, torch.tensor]] = 'red', # (n, 3)
                  color_map: str = 'Viridis',
    ) -> list:
        if value is None: 
            if not isinstance(color, str):
                color = to_numpy(color)
                color = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for c in color]
            marker = dict(size=size, color=color)
        else:
            marker=dict(size=size, color=to_numpy(value), colorscale=color_map, showscale=True)
        pc = to_numpy(pc)
        pc_plotly = go.Scatter3d(
            x=pc[:, 0],
            y=pc[:, 1],
            z=pc[:, 2],
            mode='markers',
            marker=marker, 
        )
        return [pc_plotly]
    
    @staticmethod
    def line_plotly(p1: Union[np.ndarray, torch.tensor], # (3)
                    p2: Union[np.ndarray, torch.tensor], # (3)
                    width: int = None,
                    color: str = None,
    ) -> list:
        color = 'green' if color is None else color
        width = 1 if width is None else width

        p1, p2 = to_numpy(p1), to_numpy(p2)
        pc = np.stack([p1, p2])
        x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
        return [go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(width=width, color=color),
        )]

    @staticmethod
    def to_color(array: Union[np.ndarray, torch.tensor]): # (3)
        array = to_numpy(array)
        return f'rgb({int(array[0]*255)},{int(array[1]*255)},{int(array[2]*255)})'

    
    @staticmethod
    def mesh_plotly(path: str = None,
                    scale: float = 1.0,
                    trans: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, )
                    rot: Optional[Union[np.ndarray, torch.tensor]] = None, # (3, 3)
                    opacity: float = 1.0,
                    color: str = 'lightgreen',
                    vertices: Optional[Union[np.ndarray, torch.tensor]] = None, # (n, 3)
                    faces: Optional[Union[np.ndarray, torch.tensor]] = None, # (m, 3)
    ) -> list:
        
        trans = np.zeros(3) if trans is None else to_numpy(trans)
        rot = np.eye(3) if rot is None else to_numpy(rot)
        
        if path is not None:
            mesh = tm.load(path).apply_scale(scale)
            vertices, faces = mesh.vertices, mesh.faces
        vertices = to_numpy(vertices) * scale
        faces = to_numpy(faces)

        v = np.einsum('ij,kj->ki', rot, vertices) + trans
        f = faces
        mesh_plotly = go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            color=color,
            opacity=opacity,
        )
        return [mesh_plotly]
    
    def traj_plotly(self,
                    idx: Optional[Union[int, str]] = None,
                    frame: int = 0,
                    root: Optional[str] = None,
                    traj: Optional[Dict[str, np.ndarray]] = None,
                    mode: str = 'train',
                    vis_objs: List[str] = ['obj', 'robot'],
    ) -> list:
        result = []
        traj = get_traj(idx, root, mode) if traj is None else traj
        state = read_traj(traj, frame)

        if 'obj' in vis_objs:
            for objs, obj_pose in zip(state['objs'], state['obj_pose']):
                assert objs['type'] == 'graspnet'
                obj_idx = objs['id']
                if obj_idx != -1:
                    result += self.mesh_plotly(path=get_mesh_file(obj_idx), rot=obj_pose[:3, :3], trans=obj_pose[:3, 3])
        if 'robot' in vis_objs:
            result += self.robot_plotly(qpos=to_torch(state['qpos'][None]), mesh_type='collision')
        
        return result

    
    # def acronym_scene_plotly(self,
    #                          scene: str,
    #                          view: str,
    #                          camera: str = 'realsense',
    #                          mode: str = 'model',
    #                          opacity: Optional[float] = None,):
    #     path = f'data/scenes/scene_0000/{camera}'
    #     align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
    #     camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
    #     camera_pose = np.matmul(align_mat, camera_poses[int(view)])
    #     if mode == 'model':
    #         poses = np.load(os.path.join('data_acronym', 'scenes', scene+'.npz'))
    #         result = []
    #         for obj_idx, pose in poses.items():
    #             pose = torch.from_numpy(np.linalg.inv(camera_pose) @ pose).float()
    #             mesh_path = os.path.join('data_acronym', 'meshdata', obj_idx, 'scaled.obj')
    #             result += self.mesh_plotly(mesh_path, trans=pose[:3, 3], rot=pose[:3, :3], opacity=opacity)
            
    #         table_mat = np.linalg.inv(camera_pose)
    #         table_trans = table_mat[:3, 3] - 0.05 * table_mat[:3, 2]
    #         result += self.box_plotly(np.array([1,1,0.1]), table_trans, table_mat[:3,:3], color='blue')
    #         return result
    
    # def scene_plotly(self,
    #                  scene: str,
    #                  view: str,
    #                  camera: str,
    #                  mode: str = 'model',
    #                  num_points: int = 40000,
    #                  with_pc: bool = False,
    #                  with_extrinsics: bool = False,
    #                  graspness_path: Optional[str] = None,
    #                  opacity: Optional[float] = None,
    #                  gt: int = 1,
    # ):
    #     """
    #         if with_pc is True, return both pc and plotly
    #         pc: torch.tensor (N, 5) with (x, y, z, seg, graspness)
    #     """
    #     path = os.path.join('data', 'scenes', scene, camera)
    #     gt_str = "_gt" * gt
    #     if mode == 'pc':
    #         # loading
    #         depth = np.array(Image.open(os.path.join(path, 'depth'+gt_str, str(view).zfill(4) + '.png')))
    #         edge_path = os.path.join(path, 'edge'+gt_str, str(view).zfill(4) + '.png')
    #         if os.path.exists(edge_path):
    #             edge = np.array(Image.open(edge_path))
    #         else:
    #             print(f'edge not found: {edge_path}')
    #             edge = np.zeros_like(depth)
    #         seg = np.array(Image.open(os.path.join(path, 'label'+gt_str, str(view).zfill(4) + '.png')))
    #         meta = scio.loadmat(os.path.join(path, 'meta', str(view).zfill(4) + '.mat'))
    #         camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
    #         align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
    #         instrincs = meta['intrinsic_matrix']
    #         factor_depth = meta['factor_depth']

    #         # mask pc
    #         cloud = depth_image_to_point_cloud(depth, instrincs, factor_depth)
    #         depth_mask = (depth > 0)
    #         trans = np.dot(align_mat, camera_poses[int(view)])
    #         workspace_mask = get_workspace_mask(cloud, seg, trans)
    #         mask = (depth_mask & workspace_mask)
    #         cloud = cloud[mask]
    #         seg = seg[mask]
    #         edge = edge[mask]
    #         # idxs = np.arange(len(cloud))
    #         idxs = np.random.choice(len(cloud), num_points, replace=True)
    #         cloud = cloud[idxs]
    #         seg = seg[idxs]
    #         edge = edge[idxs]
    #         if graspness_path is not None:
    #             graspness = np.load(os.path.join('data', graspness_path, scene, camera, str(view).zfill(4) + '.npy'))
    #             graspness = graspness.reshape(-1)
    #             graspness = graspness[idxs]

    #         result = []
    #         for i, idx in enumerate(np.unique(seg)):
    #             result += self.pc_plotly(torch.from_numpy(cloud[seg == idx]), size=1, color=px.colors.qualitative.Dark24[i])
    #         if with_pc:
    #             list = [cloud, seg[:, None]]
    #             if graspness_path is not None:
    #                 list.append(graspness[:, None])
    #             list.append(edge[:, None])
    #             if with_extrinsics:
    #                 return result, torch.from_numpy(np.concatenate(list, axis=-1)).float(), trans
    #             else:
    #                 return result, torch.from_numpy(np.concatenate(list, axis=-1)).float()
    #         return result, None

    #     elif mode == 'model':
    #         scene_reader = xmlReader(os.path.join(path, 'annotations', str(view).zfill(4) + '.xml'))
    #         align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
    #         camera_poses = np.load(os.path.join(path, 'camera_poses.npy'))
    #         posevectors = scene_reader.getposevectorlist()

    #         result = []
    #         for posevector in tqdm(posevectors, desc='loading scene objects'):
    #             obj_idx, pose = parse_posevector(posevector)
    #             pose = torch.from_numpy(pose).float()
    #             mesh_path = os.path.join('data', 'meshdata', str(obj_idx).zfill(3), 'coacd', 'decomposed.obj')
    #             result += self.mesh_plotly(mesh_path, trans=pose[:3, 3], rot=pose[:3, :3], opacity=opacity)
            
    #         table_mat = np.linalg.inv(np.matmul(align_mat, camera_poses[int(view)]))
    #         table_trans = table_mat[:3, 3] - 0.05 * table_mat[:3, 2]
    #         result += self.box_plotly(np.array([1,1,0.1]), table_trans, table_mat[:3,:3], color='blue')
    #         return result, None
    #     else:
    #         raise ValueError('mode should be either pc or model')
    
    # def get_scene_view_camera(
    #     self,
    #     scene: str,
    #     view: str,
    #     camera: str = 'kinect',
    # ):
    #     path = os.path.join('data', 'scenes', scene, camera)
    #     camera_pose = np.load(os.path.join(path, 'camera_poses.npy'))[int(view)]
    #     align_mat = np.load(os.path.join(path, 'cam0_wrt_table.npy'))
    #     pose = np.linalg.inv(np.matmul(align_mat, camera_pose))
    #     x = -0.15
    #     y = -0.25
    #     z = 1.5
    #     eye = np.array([x, y, z, 1])
    #     center = np.array([x, y, 0, 1])
    #     eye = np.dot(pose, eye)[:3]
    #     center = np.dot(pose, center)[:3]
    #     names = ('x', 'y', 'z')
    #     return dict(
    #         up={k: v for k, v in zip(names, -pose[:3, 1])},
    #         center={k: v for k, v in zip(names, center)},
    #         eye={k: v for k, v in zip(names, eye)},
    #     )

    @staticmethod
    def show(plotly_list: list,
             path: Optional[str] = None,
            #  scene: Optional[str] = None,
            #  view: Optional[str] = None,
            #  camera: Optional[str] = 'realsense',
    ) -> None:
        fig = go.Figure(data=plotly_list, layout=go.Layout(scene=dict(aspectmode='data')))
        # if scene is not None and view is not None:
        #     camera = self.get_scene_view_camera(scene, view, camera)
        #     fig.update_layout(scene_camera=camera)
        if path is None:
            fig.show()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.write_html(path)
        print('saved')

    @staticmethod
    def show_series(plotly_list: list,
             path: Optional[str] = None,
            #  scene: Optional[str] = None,
            #  view: Optional[str] = None,
            #  camera: Optional[str] = 'realsense',
    ) -> None:
        fig = go.Figure(layout=go.Layout(scene=dict(aspectmode='data')))
        for i, d in enumerate(plotly_list):
            for scatter in d:
                scatter.visible = (i == 0)
                fig.add_trace(scatter)
        
        # Create the slider steps
        steps = []
        num_scatter_each = [len(d) for d in plotly_list]
        for i, d in enumerate(plotly_list):
            step = dict(
                method='update',
                args=[{'visible': [False] * sum(num_scatter_each[:i]) + [True] * num_scatter_each[i] + [False] * sum(num_scatter_each[i+1:])},
                    {'title': f"Time: {i}"}],  # Update the title
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            pad={"t": 100},
            steps=steps
        )]

        # Update the layout to include the slider
        fig.update_layout(
            sliders=sliders,
            title="Time: Time 1"
        )

        if path is None:
            fig.show()
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.write_html(path)
        print('saved')
