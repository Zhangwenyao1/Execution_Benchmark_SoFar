import warnings
import numpy as np
import ikpy.chain
from src.utils.constants import ARM_URDF, ROBOT_JOINTS

# ROBOT_BASE = dict(franka = 'link_base')
# ROBOT_ARM_JOINTS = dict(google_robot=ROBOT_JOINTS[:7])
# INIT_QPOS = dict(google_robot=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
ROBOT_BASE = dict(franka='panda_link0')
ROBOT_ARM_JOINTS = dict(franka=ROBOT_JOINTS[:7])
class IK:
    def __init__(self, robot='franka') -> None:
        self.robot = robot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.joints = [l.name for l in ikpy.chain.Chain.from_urdf_file(ARM_URDF, base_elements=[ROBOT_BASE[robot]]).links]
        
        self.movable_joints = ROBOT_ARM_JOINTS[robot]
        self.robot_joints = ROBOT_JOINTS
        self.arm = ikpy.chain.Chain.from_urdf_file(ARM_URDF, base_elements=[ROBOT_BASE[robot]], active_links_mask=[l in self.movable_joints for l in self.joints])
        # self.joints = ['joint_torso', 'joint_shoulder', 'joint_bicep', 'joint_elbow', 'joint_forearm', 'joint_wrist', 'joint_gripper']
        #self.self_test()
    
    def pad_joints(self, joints: np.array):
        # arm joints -> full joints
        full_joints = np.zeros(len(self.joints))
        for i, name in enumerate(self.movable_joints):
            full_joints[self.joints.index(name)] = joints[i]
        return full_joints

    def unpad_joints(self, joints: np.array):
        # full joints -> arm joints
        return np.array([joints[self.joints.index(name)] for name in self.movable_joints])
    
    def robot_to_arm_joints(self, joints: np.array):
        # robot joints -> arm joints
        arm_qpos = np.zeros(len(self.movable_joints))
        for i, name in enumerate(self.movable_joints):
            arm_qpos[i] = joints[self.robot_joints.index(name)]
        return arm_qpos
    
    def arm_to_robot_joints(self, joints: np.array, ref: np.array = None):
        # arm joints -> robot joints
        if ref is None:
            ref = INIT_QPOS[self.robot]
        robot_qpos = ref.copy()
        for i, name in enumerate(self.movable_joints):
            robot_qpos[self.robot_joints.index(name)] = joints[i]
        return robot_qpos
    
    def get_default_joints(self):
        init_full_qpos = INIT_QPOS[self.robot]
        return self.robot_to_arm_joints(init_full_qpos)
   
    def fk(self, joints: np.array = None):
        """
            Forward kinematics function
            joints: joint angles, (7,)
            
            return rotation matrix and translation vector, (3, 3), (3,)
        """
        if joints is None:
            joints = self.get_default_joints()
        fk = self.arm.forward_kinematics(self.pad_joints(joints).tolist())
        fk = np.array(fk).reshape(4, 4)
        return fk[:3, :3], fk[:3, 3]

    def ik(self, trans: np.array, rot: np.array, joints: np.ndarray = None, allow_fail: bool = False, silent: bool = False):
        """
            Inverse kinematics function
            trans: translation in robot frame, (3,)
            rot: rotation in robot frame, (3, 3)
            joints: current joints, (7,)
            
            return joint angles, (7,)
        """
        if joints is None:
            joints = self.get_default_joints()
        try:
            ik = self.arm.inverse_kinematics(target_position=trans, 
                                            target_orientation=rot, 
                                            orientation_mode='all',
                                            initial_position=self.pad_joints(joints).tolist())
        except Exception as e:
            if allow_fail:
                return joints
            else:
                raise e
        
        fk2 = self.arm.forward_kinematics(ik)
        ik_arm = self.unpad_joints(ik)
        if not ((np.diag(fk2[:3, :3]@rot.T).sum()-1)/2 > 0.99 and np.linalg.norm(fk2[:3, 3] - trans) < 0.005):
            if allow_fail:
                return ik_arm
            if not silent:
                if not (np.diag(fk2[:3, :3]@rot.T).sum()-1)/2 > 0.99:
                    print(f'invalid rot {rot}')
                elif not np.linalg.norm(fk2[:3, 3] - trans) < 0.005:
                    print(f'invalid trans {trans}')
                # print(f'unreachable trans {trans}, rot {rot}')
                return None
            # raise ValueError
        
        return ik_arm
    
    def self_test(self, vis=False):
        import torch
        from src.utils.vis_plotly import Vis
        from plan.src.utils.robot_model import RobotModel
        robot_model = RobotModel(ARM_URDF)
        arm_qpos = self.get_default_joints()
        rot, trans = self.fk(arm_qpos)
        link_trans, link_rot = (x[self.joints[-1]].numpy() for x in robot_model.forward_kinematics(arm_qpos))
        assert np.allclose(trans, link_trans) and np.allclose(rot, link_rot)
        ik = self.ik(trans, rot)
        if vis:
            vis = Vis(urdf=ARM_URDF[self.robot])
            ply = vis.robot_plotly(qpos=arm_qpos) + vis.pose_plotly(trans, rot)
            vis.show(ply)