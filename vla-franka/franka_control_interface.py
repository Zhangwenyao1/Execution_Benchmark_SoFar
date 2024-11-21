import panda_py
import panda_py.controllers
from panda_py import libfranka
import numpy as np
from copy import deepcopy


class FrankaPyController:
    REAL_EEF_TO_SIM_EEF = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ])

    def __init__(self, robot_host):
        self.panda = panda_py.Panda(robot_host)
        self.gripper = libfranka.Gripper(robot_host)
        self.gripper_status = None

        # start_joint_pose = [0, -0.5585, 0, -2.3038, 0, 1.6580, 0]
        # self.panda.move_to_joint_position(start_joint_pose)

    def set_waypoints(self, waypoints, use_ik=True):
        pending_waypoints = []
        for idx, (pos, ori_mat, gripper_action) in enumerate(waypoints):
            mat = np.eye(4)
            mat[:3, 3] = pos
            mat[2, 3] = max(mat[2, 3], 0.05)
            mat[:3, :3] = ori_mat
            # from sim pose mat to real pose mat
            mat = mat @ np.linalg.inv(self.REAL_EEF_TO_SIM_EEF)
            if use_ik:
                current_q = self.panda.get_state().q if len(
                    pending_waypoints) == 0 else pending_waypoints[-1]
                # candidate_q7s should be close to current_q[6], exponentially, starting from 0.001
                candidate_q7s = np.array([0.01 * i for i in range(290)] +
                                         [-0.01 * i for i in range(290)] + [0])
                candidate_q7s = np.clip(candidate_q7s, -2.8973, 2.8973)
                candidate_qs = []
                for candidate_q7 in candidate_q7s:
                    q = panda_py.ik(mat, q_init=current_q, q_7=candidate_q7)
                    if not np.isnan(q).any():
                        candidate_qs.append(q)
                if len(candidate_qs) == 0:
                    print(f'no ik solution found, return')
                    return
                candidate_qs = np.array(candidate_qs)
                waypoint = candidate_qs[np.argmin(
                    np.linalg.norm(candidate_qs - current_q, ord=1, axis=1))]
            else:
                waypoint = mat
            pending_waypoints.append(waypoint)

            desired_gripper_status = None
            if gripper_action <= -0.9 and self.gripper_status == 'open':
                desired_gripper_status = 'close'
            elif gripper_action > 0.9 and self.gripper_status == 'close':
                desired_gripper_status = 'open'

            if desired_gripper_status is not None or idx == len(waypoints) - 1:
                try:
                    if use_ik:
                        self.panda.move_to_joint_position(
                            pending_waypoints, speed_factor=args.speed_factor)
                    else:
                        self.panda.move_to_pose(
                            pending_waypoints,
                            success_threshold=0.002,
                            speed_factor=args.speed_factor,
                            impedance=[[900., 0., 0., 0., 0., 0.],
                                       [0., 900., 0., 0., 0., 0.],
                                       [0., 0., 900., 0., 0., 0.],
                                       [0., 0., 0., 40., 0., 0.],
                                       [0., 0., 0., 0., 40., 0.],
                                       [0., 0., 0., 0., 0., 40.]])
                        for _ in range(2):
                            self.panda.move_to_pose(
                                pending_waypoints[-1],
                                success_threshold=0.002,
                                speed_factor=args.speed_factor,
                                impedance=[[900., 0., 0., 0., 0., 0.],
                                           [0., 900., 0., 0., 0., 0.],
                                           [0., 0., 900., 0., 0., 0.],
                                           [0., 0., 0., 40., 0., 0.],
                                           [0., 0., 0., 0., 40., 0.],
                                           [0., 0., 0., 0., 0., 40.]])
                except Exception as e:
                    print(
                        f'failed to move to waypoint {waypoint} due to {repr(e)}'
                    )
                pending_waypoints.clear()
            if desired_gripper_status is not None:
                self.move_gripper(desired_gripper_status)
                self.gripper_status = desired_gripper_status

    def get_current_pose(self):
        real_pos = self.panda.get_position()
        real_ori_quat_sxyz = self.panda.get_orientation(scalar_first=True)
        real_pose_mat = np.eye(4)
        real_pose_mat[:3, 3] = real_pos
        real_pose_mat[:3, :3] = t3d.quaternions.quat2mat(real_ori_quat_sxyz)
        sim_pose_mat = real_pose_mat @ self.REAL_EEF_TO_SIM_EEF
        sim_pos = sim_pose_mat[:3, 3]
        sim_ori_rpy = t3d.euler.mat2euler(sim_pose_mat[:3, :3])
        return np.array([
            *sim_pos, *sim_ori_rpy, 1 if self.gripper_status == 'open' else -1
        ])

    def get_current_qpos(self):
        return self.panda.get_state().q

    def move_gripper(self, action):
        if action == 'close':
            self.gripper.grasp(0.02, 0.2, 50, 0.04, 0.04)
        if action == 'open':
            self.gripper.move(0.08, 0.2)

    def wait(self):
        """set_waypoints() is blocking for now."""
        return

    def modify_grasp(self, grasp: Grasp):
        """Modify the grasp pose and pre-grasp pose
            Args:
                grasp (Grasp): the grasp object
            Returns:
                modify_grasp (Grasp): the modified grasp object
                modify_grasp_pre (Grasp): the modified pre-grasp object
                """
        tmp_grasp = deepcopy(grasp)
        # modify orientation
        grasp_z_p = tmp_grasp.rotation_matrix[:, 2]
        robot_x_p = np.array([1, 0, 0])
        dot = np.dot(grasp_z_p,
                     robot_x_p)  # make sure the z-axis is pointing up
        if dot < 0:
            tmp_grasp.rotation_matrix[:, 2] = -tmp_grasp.rotation_matrix[:, 2]
            tmp_grasp.rotation_matrix[:, 1] = -tmp_grasp.rotation_matrix[:, 1]
        # modify position in object frame
        modify_grasp = deepcopy(tmp_grasp)
        # modify_grasp.translation += 0.028 * modify_grasp.rotation_matrix[:,0] # tcp
        modify_grasp.translation -= 0.017 * modify_grasp.rotation_matrix[:,
                                                                         0]  # hitbot tcp
        modify_grasp_pre = deepcopy(modify_grasp)
        modify_grasp_pre.translation -= 0.15 * modify_grasp_pre.rotation_matrix[:,
                                                                                0]  # tcp
        # modify position in global frame
        t = np.array([0.013, 0, 0.0])
        r = np.eye(3)
        modify_grasp.translation = r @ modify_grasp.translation + t
        modify_grasp_pre.translation = r @ modify_grasp_pre.translation + t
        if (modify_grasp.translation[2] - self.grp_roi['z_min']) < 0.02:
            modify_grasp.translation[2] += 0.007

        return modify_grasp, modify_grasp_pre
