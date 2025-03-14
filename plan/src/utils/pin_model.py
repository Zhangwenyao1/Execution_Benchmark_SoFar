from os.path import dirname
import numpy as np
import trimesh as tm
import pinocchio

from src.utils.utils import to_torch

class PinRobotModel:
    def __init__(self, urdf_path):
        self.model, self.collision_model, self.visual_model = pinocchio.buildModelsFromUrdf(
            urdf_path, dirname(urdf_path)
        )

        # Create data required by the algorithms
        self.data, self.collision_data, self.visual_data = pinocchio.createDatas(
            self.model, self.collision_model, self.visual_model
        )

        self.setup_mesh()
        self.joint_names = [n for n in self.model.names[1:]]

    def setup_mesh(self):
        self.meshes = dict(collision=[], visual=[])
        for mode, model in [('collision', self.collision_model), ('visual', self.visual_model)]:
            for obj in model.geometryObjects:
                self.meshes[mode].append(tm.load_mesh(obj.meshPath))

    def forward_kinematics(self, q: np.ndarray, mode: str):
        pinocchio.forwardKinematics(self.model, self.data, q)
        if mode == 'link':
            pinocchio.updateFramePlacements(self.model, self.data)
            link_trans = {self.model.frames[i].name: to_torch(self.data.oMf[i].translation).float() for i in range(len(self.model.frames))}
            link_rot = {self.model.frames[i].name: to_torch(self.data.oMf[i].rotation).float() for i in range(len(self.model.frames))}
            return link_trans, link_rot
        elif mode == 'visual':
            pinocchio.updateGeometryPlacements(self.model, self.data, self.visual_model, self.visual_data)
            d = self.visual_data
        elif mode == 'collision':
            pinocchio.updateGeometryPlacements(self.model, self.data, self.collision_model, self.collision_data)
            d = self.collision_data
        else:
            raise ValueError(f'mode must be visual/collision. current mode: {mode}')
        return [(oMg.translation, oMg.rotation) for oMg in d.oMg]