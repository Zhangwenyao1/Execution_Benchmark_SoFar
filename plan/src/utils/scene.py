import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
import numpy as np
import torch
import yaml

from src.utils.vis_plotly import Vis
from src.utils.robot_model import RobotModel
# from src.utils.pin_model import PinRobotModel
from src.utils.mesh_pc import MeshPC
from src.utils.utils import to_torch, cdist_test
# from src.utils.data import get_mesh_file
from src.utils.constants import ROBOT_JOINTS, ROBOT_URDF, FRANKA_COLLISION_FILE, FRANKA_CUROBO_FILE
from src.utils.config import DotDict
import open3d as o3d

def box_dist_test(pc, box, thresh):
    box = box.clone()
    box[0] -= thresh
    box[1] += thresh
    # return ((pc[:, 0] > box[0][0]) & (pc[:, 0] < box[1][0]) & (pc[:, 1] > box[0][1]) & (pc[:, 1] < box[1][1]) & (pc[:, 2] > box[0][2]) & (pc[:, 2] < box[1][2])).any()
    for i in range(3):
        pc = pc[(pc[:, i] > box[0][i]) & (pc[:, i] < box[1][i])]
        if len(pc) == 0:
            return False
    return True

class Scene:
    def __init__(self, table_thresh=0.0005, obj_thresh=0.0001, self_thresh=0.0001, obj_thresh_sparse=0.0001):
        self.table_thresh, self.obj_thresh, self.self_thresh, self.obj_thresh_sparse = table_thresh, obj_thresh, self_thresh, obj_thresh_sparse
        self.vis = Vis()
        self.robot = RobotModel(ROBOT_URDF)
        # self.pin_robot = PinRobotModel(ROBOT_URDF)
        # self.mesh_pc_sparse = MeshPC(N=None, dist=self.obj_thresh_sparse)
        # self.mesh_pc = MeshPC(N=None, dist=self.obj_thresh)
        # self.mesh_pc = MeshPC(N=4096)
        # with open(FRANKA_COLLISION_FILE, 'r') as f:
        #     self.collision = {k: {k2: torch.tensor([v2[k2] for v2 in v], dtype=torch.float32) for k2 in ['center', 'radius']} 
        #                       for k, v in yaml.load(f, Loader=yaml.FullLoader)['collision_spheres'].items()}
        with open(FRANKA_CUROBO_FILE, 'r') as f:
            self.collision_ignore = yaml.load(f, Loader=yaml.FullLoader)['robot_cfg']['kinematics']['self_collision_ignore']
        self.adj_list = []
        adj = self.robot.adjacency_mask
        # print(self.collision_ignore)
        # print(adj)
        for i in range(len(adj)):
            for j in range(i):
                link_i, link_j = self.robot.link_names[i], self.robot.link_names[j]
                if adj[i, j] or self.check_collision_ignore(link_i, link_j):
                    continue
                self.adj_list.append((link_i, link_j))
    
    def check_collision_ignore(self, link_i, link_j):
        if link_i in self.collision_ignore[link_j]:
            return True
        if link_j in self.collision_ignore[link_i]:
            return True
        return False
    
    def vis_robot_collision(self, state):
        scene_ply = self.to_plotly(state, opacity=0.5, robot_mesh_type='visual')
        qpos = to_torch(state['qpos'][None]).float()
        qpos = {k: qpos[:, i] for i, k in enumerate(ROBOT_JOINTS)}
        coll_ply = []
        link_trans, link_rot = self.robot.forward_kinematics(qpos)
        for k in self.collision.keys():
            # if k in FRANKA_ADDITIONAL_BOXES:
                # continue
            for i in range(len(self.collision[k]['center'])):
                center = torch.einsum('ab,b->a', link_rot[k][0], self.collision[k]['center'][i]) + link_trans[k][0]
                radius = self.collision[k]['radius'][i]
                coll_ply += self.vis.sphere_plotly(center, radius, opacity=0.8)
        
        # for k, vs in FRANKA_ADDITIONAL_BOXES.items():
        #     for v in vs:
        #         box = to_torch(v).float()
        #         scale = box[1] - box[0]
        #         center = box[0] + scale / 2
        #         trans = torch.einsum('ab,b->a', link_rot[k][0], center) + link_trans[k][0]
        #         coll_ply += self.vis.box_plotly(scale, trans, link_rot[k][0], color='blue', opacity=0.8)
        return scene_ply + coll_ply
        
    def get_obj_pc(self, state, sparse=False):
        mesh_pc = self.mesh_pc_sparse if sparse else self.mesh_pc
        return mesh_pc.get_full_pc(state['obj'], with_table=False, separate=True)
    
    def to_pc(self, state, with_robot_pc=True) -> DotDict:
        
        result = DotDict()
        qpos = to_torch(state['qpos'][None]).float()
        qpos = {k: qpos[:, i] for i, k in enumerate(ROBOT_JOINTS)}
        if with_robot_pc:
            robot_pc, link_trans, link_rot, link_pc = self.robot.sample_surface_points_full(qpos, n_points_each_link=2**11, with_fk=True)
            robot_pc = robot_pc[0]
            link_trans, link_rot, link_pc = ({k: v[0] for k, v in x.items()} for x in [link_trans, link_rot, link_pc])
            result['link_pc'] = link_pc
            result['robot_pc'] = robot_pc
        else:
            link_trans, link_rot = self.robot.forward_kinematics(qpos)
            link_trans, link_rot = ({k: v[0] for k, v in x.items()} for x in [link_trans, link_rot])
            # lk, lr = self.pin_robot.forward_kinematics(state['qpos'], mode='link')
            # for k in link_trans:
            #     if (not torch.allclose(link_trans[k], lk[k])) or (not torch.allclose(link_rot[k], lr[k])):
            #         print(k)
            #print(state['qpos'])
            #print(type(state['qpos']))
            #link_trans, link_rot = self.pin_robot.forward_kinematics(state['qpos'], mode='link')
        result['link_trans'] = link_trans
        result['link_rot'] = link_rot
        return result
    
    def get_robot_sphere(self, qpos, link_trans=None, link_rot=None):
        if link_trans is None:
            qpos = to_torch(qpos[None]).float()
            qpos = {k: qpos[:, i] for i, k in enumerate(ROBOT_JOINTS)}
            link_trans, link_rot = ({k: v[0] for k, v in x.items()} for x in self.robot.forward_kinematics(qpos))
        link_spheres = {}
        for k in self.collision.keys():
            center = torch.einsum('ab,nb->na', link_rot[k], self.collision[k]['center']) + link_trans[k]
            link_spheres[k] = dict(center=center, radius=self.collision[k]['radius'])
        spheres = dict(center=torch.cat([v['center'] for v in link_spheres.values()], dim=0), radius=torch.cat([v['radius'] for v in link_spheres.values()], dim=0))
        return spheres, link_spheres
    
    # @profile
    def check_coll(self, state):
        # self.vis_robot_collision(state)
        result = self.to_pc(state, with_robot_pc=True)
        # Vis.show(Vis.pc_plotly(result['robot_pc']))


        # # 将点云转换为 open3d 格式
        # state_pc = o3d.geometry.PointCloud()
        # state_pc.points = o3d.utility.Vector3dVector(state['pc'])

        # robot_pc = o3d.geometry.PointCloud()
        # robot_pc.points = o3d.utility.Vector3dVector(result['robot_pc'])

        # # 使用 KDTree 查找与 robot_pc 重叠的点
        # kd_tree = o3d.geometry.KDTreeFlann(state_pc)
        # indices_to_remove = []
        # for point in robot_pc.points:
        #     [_, idx, _] = kd_tree.search_radius_vector_3d(point, radius=0.05)  # 设置合适的半径
        #     indices_to_remove.extend(idx)

        # # 移除重复点
        # state_pc = state_pc.select_by_index(indices_to_remove, invert=True)

        # # 更新 state['pc']
        # state['pc'] = torch.tensor(np.asarray(state_pc.points))
        
        
        #spheres, link_spheres = self.get_robot_sphere(state['qpos'], link_trans=result.link_trans, link_rot=result.link_rot)
        #if result['robot_pc'][:, 2] < self.table_thresh:
        #    return True, 'table'
        #if (spheres['center'][:, 2] - spheres['radius']).min() < self.table_thresh: 
        #    return True, 'table'
        # if not obj_thresh is None:
        #     for k, vs in FRANKA_ADDITIONAL_BOXES.items():
        #         link_trans_k, link_rot_k = result.link_trans[k], result.link_rot[k]
        #         obj_pc_in_link = torch.einsum('ba,nb->na', link_rot_k, result.obj_pc - link_trans_k)
        #         for v in vs:
        #             box = to_torch(v)
        #             if box_dist_test(obj_pc_in_link, box, obj_thresh):
        #                 return True, f'robot-obj {k} {v}'
        #                 # box2 = box.clone()
        #                 # box2[0] -= obj_thresh
        #                 # box2[1] += obj_thresh
        #                 # for i in range(3):
        #                 #     obj_pc_in_link = obj_pc_in_link[(obj_pc_in_link[:, i] > box2[0][i]) & (obj_pc_in_link[:, i] < box2[1][i])]
        #                 # Vis.show(Vis.pc_plotly(torch.einsum('ab,nb->na',link_rot_k,obj_pc_in_link)+link_trans_k,color='red',size=5)+self.to_plotly(state))
        if not self.obj_thresh is None:
                # cdist = cdist(state['pc'], result['robot_pc'])
                # if torch.min(cdist) < self.obj_thresh:
                if cdist_test(state['pc'], result['robot_pc'], self.obj_thresh):
                    return True, 'robot-obj'
                
        if not self.self_thresh is None:
            for link_i, link_j in self.adj_list:
                # cdist = torch.cdist(link_spheres[link_i]['center'], link_spheres[link_j]['center']) - link_spheres[link_j]['radius'] - link_spheres[link_i]['radius'][:, None]
                # if torch.min(cdist) < self.self_thresh:
                if cdist_test(result['link_pc'][link_i], result['link_pc'][link_j], self.self_thresh):
                    return True, f'self {link_i} {link_j}'
        # Vis.show(Vis.pc_plotly(result['robot_pc'], color='red')+Vis.pc_plotly(state['pc'], color='blue'))
        return False, None
    
    def to_plotly(self, state, opacity=1, robot_mesh_type='collision'):
        robot_ply = self.vis.robot_plotly(qpos=state['qpos'], opacity=opacity, mesh_type=robot_mesh_type)
        pc_ply = self.vis.pc_plotly(state['pc'], color='blue', size=1)
        scene_ply = robot_ply + pc_ply
        return scene_ply
    
    def to_plotly_traj(self, qpos_list, pc):
        plys = []
        for q in qpos_list:
            plys.append(self.to_plotly(state=dict(qpos=q, pc=pc), opacity=1))
        self.vis.show_series(plys[-1])