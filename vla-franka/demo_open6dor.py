      
"""
File: demo_placement.py
Author: Xiaomeng Fang
Description: demo script for testing placement policy

Dependencies:
    - 
    
History:
    - Version 0.0 (2024-01-28): careated
    
Run:
    - bash argrasp_ros_node.sh
"""

import time
import os
import rospkg
import rospy
import tf
import imageio
import numpy as np
import open3d as o3d
import geometry_msgs.msg
import tf2_geometry_msgs
import transforms3d as t3d
import matplotlib.pyplot as plt
import sys
ROS_PKG_NAME = "argrasp"
rospack = rospkg.RosPack()
__PACK_PATH__ = rospack.get_path(ROS_PKG_NAME)
sys.path.append(os.path.join(__PACK_PATH__, "lib"))


from copy import deepcopy
from loguru import logger
# from observation.realsense_camera_sdk import RealsenseCamera
from camera import RealsenseCamera
# from robot.panda_arm_ros import PandaArm
# from robot.panda_arm import PandaArm
from robot import execute_grasp, reset_arm, PandaArm, PandaGripper
# from robot.panda_gripper import PandaGripper
# from placement.inference_place_pose import PlacePolicy
# from observation.ASGrasp import MVSGSNetEval
from ASGrasp import MVSGSNetEval
# from grasping.gsnet import inference_rt
from gsnet import inference_rt
from graspnetAPI import Grasp, GraspGroup
# from graspnetAPI import Grasp, GraspGroup
from scipy.spatial.transform import Rotation as R

# Global variables
INIT_JNT = np.array([0, 0, 0, -90, 0, 90, 45]) # default home position
# INIT_JNT = np.array([6, -32, -8, -86, -1, 70, 41]) # high home position
MID_JNT = np.array([0, 0, 0, -90, 0, 90, 45])

import tf2_ros


# Global variables
INIT_JNT = np.array([0, 0, 0, -90, 0, 90, 45]) # default home position
# INIT_JNT = np.array([6, -32, -8, -86, -1, 70, 41]) # high home position
MID_JNT = np.array([0, 0, 0, -90, 0, 90, 45])

import tf2_ros

# def broadcast_transform():
#     rospy.init_node('my_tf_broadcaster')

#     br = tf2_ros.TransformBroadcaster()

#     t = geometry_msgs.msg.TransformStamped()

#     t.header.stamp = rospy.Time.now()
#     t.header.frame_id = "base_link"
#     t.child_frame_id = "camera_link"
#     t.transform.translation.x = 0.0
#     t.transform.translation.y = 0.0
#     t.transform.translation.z = 1.0
#     t.transform.rotation.x = 0.0
#     t.transform.rotation.y = 0.0
#     t.transform.rotation.z = 0.0
#     t.transform.rotation.w = 1.0

#     br.sendTransform(t)
    
def get_camera_pose(pose_type = 'tf'):
    tf_base_cam = geometry_msgs.msg.TransformStamped()
    listener = tf.TransformListener()
    try:
        listener.waitForTransform('panda_link0', 'camera_link', rospy.Time(0), rospy.Duration(1.0))
        trans, rot = listener.lookupTransform('panda_link0', 'camera_link', rospy.Time(0))
        tf_base_cam.header.seq = 0
        tf_base_cam.header.stamp = rospy.Time.now()
        tf_base_cam.header.frame_id = "panda_link0"
        tf_base_cam.child_frame_id = "camera_link"
        tf_base_cam.transform.translation.x = trans[0]
        tf_base_cam.transform.translation.y = trans[1]
        tf_base_cam.transform.translation.z = trans[2]
        tf_base_cam.transform.rotation.x = rot[0]
        tf_base_cam.transform.rotation.y = rot[1]
        tf_base_cam.transform.rotation.z = rot[2]
        tf_base_cam.transform.rotation.w = rot[3]
        logger.info("Received transform panda_link0->camera_link")
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        logger.warning("TF lookup failed")
    if pose_type == 'matrix': # 'matrix'
        camera_pose = tf.transformations.quaternion_matrix([tf_base_cam.transform.rotation.x,
                                                            tf_base_cam.transform.rotation.y,
                                                            tf_base_cam.transform.rotation.z,
                                                            tf_base_cam.transform.rotation.w])
        camera_pose[0, 3] = tf_base_cam.transform.translation.x
        camera_pose[1, 3] = tf_base_cam.transform.translation.y
        camera_pose[2, 3] = tf_base_cam.transform.translation.z
    else: # 'tf'
        camera_pose = tf_base_cam
    return camera_pose

def get_pcd():
    realsense_camera = RealsenseCamera(is_reset=True)
    depth = realsense_camera.get_aligned_depth_image()
    depth = np.clip(depth, 0, 800)
    depth = depth/1000.0
    color = realsense_camera.get_color_image()
    intrinsics = realsense_camera.get_color_intrinsics()
    camera_pose = get_camera_pose('matrix')
    pcd, points, colors = realsense_camera.create_point_cloud(depth, color, intrinsics, camera_pose)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return pcd, depth, color, points, colors 

def get_observations(args):
    
    realsense_camera = RealsenseCamera(is_reset=False)

    rgb = realsense_camera.get_color_image()
    raw_rgb = deepcopy(rgb)
    irl = realsense_camera.get_ir_image(1)
    raw_irl = deepcopy(irl)
    irr = realsense_camera.get_ir_image(2)
    raw_irr = deepcopy(irr)
    
    mvsnet = MVSGSNetEval(args)
    depth1, _ = mvsnet.infer_depth(rgb, irl, irr)
    depth = np.clip(depth1, 0, 0.8)
    raw_depth = deepcopy(depth)

    intrinsics = realsense_camera.get_color_intrinsics()
    camera_pose = get_camera_pose('matrix')
    pcd, points, colors = realsense_camera.create_point_cloud(depth, rgb, intrinsics, camera_pose)

    return pcd, points, colors, raw_depth, raw_rgb, raw_irl, raw_irr     

def to_pose_stamped(grasp: Grasp, tf_base_cam: geometry_msgs.msg.TransformStamped):

    grasp_pose = geometry_msgs.msg.PoseStamped()
    grasp_pose.header.seq = 1
    grasp_pose.header.frame_id = "camera_link"
    grasp_pose.pose.position.x = grasp.translation[0]
    grasp_pose.pose.position.y = grasp.translation[1]
    grasp_pose.pose.position.z = grasp.translation[2]

    quat = t3d.quaternions.mat2quat(grasp.rotation_matrix)
    grasp_pose.pose.orientation.x = quat[1]
    grasp_pose.pose.orientation.y = quat[2]
    grasp_pose.pose.orientation.z = quat[3]
    grasp_pose.pose.orientation.w = quat[0]

    grasp_pose_base = tf2_geometry_msgs.do_transform_pose(grasp_pose, tf_base_cam)
    return grasp_pose_base


def get_grasp_pose(args, rgb, irl, irr, log_dirs=None, target_mask=None):

    mvsnet = MVSGSNetEval(args)
    pts, clr = mvsnet.infer_rt(rgb, irl, irr)

    grasp_group = inference_rt(pts, clr, args, log_dirs=log_dirs)
    logger.info(f"Number of ram grasps: {len(grasp_group)}")
    
    tf_base_cam = get_camera_pose()
    up = np.array([0, 0, 1])     
    gg_valid = GraspGroup()
    grasp = Grasp()
    
    for i in range(len(grasp_group)):
        grasp = grasp_group[i]
        
        if target_mask is not None:
            realsense_camera = RealsenseCamera(is_reset=False)
            pnt = np.asarray(grasp.translation)
            if not realsense_camera.is_point_in_mask(pnt, target_mask):
                continue
        
        grasp_pose_base = to_pose_stamped(grasp, tf_base_cam)
        quat_base = np.array([grasp_pose_base.pose.orientation.w,
                              grasp_pose_base.pose.orientation.x,
                              grasp_pose_base.pose.orientation.y,
                              grasp_pose_base.pose.orientation.z])
        mat = t3d.quaternions.quat2mat(quat_base)
        z_dir = - mat[:, 0] # mat[:, 2] # 
        angle = np.arccos(np.clip(np.dot(z_dir, up), -1.0, 1.0)) 
        if angle <= np.deg2rad(20): 
            gg_valid.add(grasp)

    assert len(gg_valid) > 0
    logger.info(f"Number of valid grasps: {len(gg_valid)}")
    gg_valid.sort_by_score()

    best_grasp = gg_valid[0]
    best_grasp_pre = deepcopy(best_grasp)
    best_grasp.translation -= 0.11 * best_grasp.rotation_matrix[:,0]
    best_grasp_pre.translation -= 0.22 * best_grasp_pre.rotation_matrix[:,0]

    if args.grasp_visualize:
        gg = GraspGroup()
        gg.add(best_grasp)
        gg.add(best_grasp_pre)
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])
   
    if args.save_logs and log_dirs is not None:
        gg = GraspGroup()
        gg.add(best_grasp)
        gg.add(best_grasp_pre)
        gg.save_npy(f"{log_dirs}/gsnet_output_slected_grasps.npy")
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(clr.astype(np.float32))
        o3d.io.write_point_cloud(f"{log_dirs}/asgrasp_output_point_cloud.ply", cloud)
        
    best_grasp_pose = to_pose_stamped(best_grasp, tf_base_cam)
    best_grasp_pose_pre = to_pose_stamped(best_grasp_pre, tf_base_cam)
    
    # grasp list
    grasp_pose_list = []
    pre_grasp_pose_list = [] 
    for i in range(len(gg_valid)):
        grasp = gg_valid[i]
        pre_grasp = deepcopy(grasp)
        grasp.translation -= 0.11 * grasp.rotation_matrix[:,0]
        pre_grasp.translation -= 0.22 * pre_grasp.rotation_matrix[:,0]
        grasp_pose = to_pose_stamped(grasp, tf_base_cam)
        pre_grasp_pose = to_pose_stamped(pre_grasp, tf_base_cam)
        grasp_pose_list.append(grasp_pose)
        pre_grasp_pose_list.append(pre_grasp_pose)

    return best_grasp_pose, best_grasp_pose_pre, grasp_pose_list, pre_grasp_pose_list



def grasp_transfer(grasp_group, log_dirs, target_mask = None):

    logger.info(f"Number of ram grasps: {len(grasp_group)}")
    
    tf_base_cam = get_camera_pose()
    up = np.array([0, 0, 1])     
    gg_valid = GraspGroup()
    grasp = Grasp()
    
    for i in range(len(grasp_group)):
        grasp = grasp_group[i]
        
        if target_mask is not None:
            realsense_camera = RealsenseCamera(is_reset=False)
            pnt = np.asarray(grasp.translation)
            if not realsense_camera.is_point_in_mask(pnt, target_mask):
                continue
        
        grasp_pose_base = to_pose_stamped(grasp, tf_base_cam)
        quat_base = np.array([grasp_pose_base.pose.orientation.w,
                              grasp_pose_base.pose.orientation.x,
                              grasp_pose_base.pose.orientation.y,
                              grasp_pose_base.pose.orientation.z])
        mat = t3d.quaternions.quat2mat(quat_base)
        z_dir = - mat[:, 0] # mat[:, 2] # 
        angle = np.arccos(np.clip(np.dot(z_dir, up), -1.0, 1.0)) 
        if angle <= np.deg2rad(20): 
            gg_valid.add(grasp)

    assert len(gg_valid) > 0
    logger.info(f"Number of valid grasps: {len(gg_valid)}")
    gg_valid.sort_by_score()

    best_grasp = gg_valid[0]
    best_grasp_pre = deepcopy(best_grasp)
    best_grasp.translation -= 0.11 * best_grasp.rotation_matrix[:,0]
    best_grasp_pre.translation -= 0.22 * best_grasp_pre.rotation_matrix[:,0]

    if args.grasp_visualize:
        gg = GraspGroup()
        gg.add(best_grasp)
        gg.add(best_grasp_pre)
        grippers = gg.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])
   
    if args.save_logs and log_dirs is not None:
        gg = GraspGroup()
        gg.add(best_grasp)
        gg.add(best_grasp_pre)
        gg.save_npy(f"{log_dirs}/gsnet_output_slected_grasps.npy")
        
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(clr.astype(np.float32))
        o3d.io.write_point_cloud(f"{log_dirs}/asgrasp_output_point_cloud.ply", cloud)
        
    best_grasp_pose = to_pose_stamped(best_grasp, tf_base_cam)
    best_grasp_pose_pre = to_pose_stamped(best_grasp_pre, tf_base_cam)
    
    # grasp list
    grasp_pose_list = []
    pre_grasp_pose_list = [] 
    for i in range(len(gg_valid)):
        grasp = gg_valid[i]
        pre_grasp = deepcopy(grasp)
        grasp.translation -= 0.11 * grasp.rotation_matrix[:,0]
        pre_grasp.translation -= 0.22 * pre_grasp.rotation_matrix[:,0]
        grasp_pose = to_pose_stamped(grasp, tf_base_cam)
        pre_grasp_pose = to_pose_stamped(pre_grasp, tf_base_cam)
        grasp_pose_list.append(grasp_pose)
        pre_grasp_pose_list.append(pre_grasp_pose)

    return best_grasp_pose, best_grasp_pose_pre, grasp_pose_list, pre_grasp_pose_list

def modify_grasp_pose(grasp, pre_grasp, ref_vec):
    """Modify the grasp pose."""
    # rotate 90 degree around y-axis
    R = t3d.quaternions.quat2mat(np.array([
        grasp.pose.orientation.w, 
        grasp.pose.orientation.x, 
        grasp.pose.orientation.y, 
        grasp.pose.orientation.z, 
    ]))
    angle_y = np.deg2rad(90)
    rot_mat_y = np.array([[ np.cos(angle_y), 0, np.sin(angle_y)],
                            [ 0            ,   1, 0],
                            [-np.sin(angle_y), 0, np.cos(angle_y)]])
    R = R @ rot_mat_y
    
    # rotate 180 degree around z-axis to avoid much rotation
    # ref_vec = np.array([1, 0, 0])
    # norm = np.linalg.norm(ref_vec)
    # ref_vec_norm = deepcopy(ref_vec)
    # if norm == 0:
    #     ref_vec_norm = ref_vec
    # else:
    #     ref_vec_norm = ref_vec / norm

    # x = ref_vec_norm[0]
    # y = ref_vec_norm[1]
    # # z = ref_vec_norm[2]
    # th = np.deg2rad(0)
    # x_r = np.cos(th)*x -  np.sin(th)*y
    # y_r = np.sin(th)*x +  np.cos(th)*y
    # ref_vec_norm[0] = x_r
    # ref_vec_norm[1] = y_r
    # ref_vec_norm[2] = 0.0

    # dot = np.dot(R[:,0], ref_vec_norm)
    # if dot < 0:
    #     angle_z = np.deg2rad(180)
    #     rot_mat_z = np.array([[ np.cos(angle_z), -np.sin(angle_z), 0],
    #                           [ np.sin(angle_z),  np.cos(angle_z), 0],
    #                           [ 0              ,  0            ,   1]])  
    #     R = R @ rot_mat_z

    q = t3d.quaternions.mat2quat(R)

    grasp_pose = np.array([
        grasp.pose.position.x, 
        grasp.pose.position.y,
        grasp.pose.position.z,
        q[1], q[2], q[3], q[0]
    ]) # x, y, z, qx, qy, qz, qw

    pre_grasp_pose = np.array([
        pre_grasp.pose.position.x, 
        pre_grasp.pose.position.y,
        pre_grasp.pose.position.z,
        q[1], q[2], q[3], q[0]
    ]) # x, y, z, qx, qy, qz, qw

    return grasp_pose, pre_grasp_pose

def pick_and_place(pick_pose=None, place_pose=None, pre_pick_pose=None, pre_place_pose=None, mid_pose=None):
    """Pick and place the object.

    Args:
        pick_pose (x y z qx qy qz qw): The pick pose.
        place_pose (x y z qx qy qz qw): The place pose.
        pre_pick_pose (x y z qx qy qz qw, optional): The pre pick pose. Defaults to None.
        pre_place_pose (x y z qx qy qz qw, optional): The pre place pose. Defaults to None.
    """
    arm = PandaArm()
    hand = PandaGripper()
    
    # Move to the init point
    logger.info("Moving to the start joint")
    arm.go_to_joint_state(np.radians(INIT_JNT))
    hand.open_gripper()
        
    # Move to the pre-pick point
    logger.info("Move to the pre-pick point")
    if pre_pick_pose is None:
        pre_dist = 0.1
        pre_pick_pose = deepcopy(pick_pose)
        rot_mat = R.from_quat(pick_pose[3:]).as_matrix()
        pre_pick_pose[:3] -= pre_dist * rot_mat[:, 2]
    arm.go_to_pose_goal(pre_pick_pose)
    hand.open_gripper()

    # Move to the pick point 
    logger.info("Move to the pick point")
    plan = arm.movel(pick_pose)
    arm.execute_plan(plan)
    hand.close_gripper()
    rospy.sleep(1.0) # wait for the gripper to close

    # Move to the lift point
    logger.info("Move to the lift point")
    reversed_plan = arm.reverse_traj(plan)
    arm.execute_plan(reversed_plan)
    
    if mid_pose is not None:
        logger.info("Moving to the mid pose")
        # arm.go_to_joint_state(np.radians(MID_JNT))
        arm.go_to_pose_goal(mid_pose)

    # Move to the pre-place point
    logger.info("Move to the pre-place point")
    if pre_place_pose is None:
        pre_place_pose = deepcopy(place_pose)
        pre_place_pose[2] += 0.2
    arm.go_to_pose_goal(pre_place_pose)
    hand.close_gripper()
    
    # Move to the place point
    logger.info("Move to the place point")
    arm.go_to_pose_goal(place_pose)
    hand.open_gripper()
    rospy.sleep(1) # wait for the gripper to open
    
    # Move to the init point
    logger.info("Moving to the start pose")
    arm.go_to_joint_state(np.radians(INIT_JNT))
    hand.open_gripper()
    
def main(args=None):
    
    arm = PandaArm(enable_constraints=False)
    hand = PandaGripper()
    
    if args.go_home:
        logger.info("Moving to the home position")
        arm.go_to_joint_state(np.radians(INIT_JNT))
        return 0
    
    wpts_list = []
    # Move to the start pose
    logger.info("Moving to the start pose")
    arm.go_to_joint_state(np.radians(INIT_JNT))
    hand.open_gripper()
    pose = arm.get_current_pose()
    init_pose = [pose.pose.position.x,
                 pose.pose.position.y,
                 pose.pose.position.z,
                 pose.pose.orientation.x,
                 pose.pose.orientation.y,
                 pose.pose.orientation.z,
                 pose.pose.orientation.w]
    wpts_list.append(init_pose)
    
    # level1 or level2 demo
    if args.demo_level == 'p1' or args.demo_level == 'p2':
        
        if args.save_logs:
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
            instruction_name = args.instruction.replace(' ', '_')
            logs_dir = os.path.abspath(f'{args.logs_dir}/{instruction_name}/{time_str}')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)
            logger.info(f'Log directory: {logs_dir}')
        
        # Get the rgb image, points and colors
        logger.info("Getting the observations from the camera")
        pcd, points, colors, depth, rgb, irl, irr = get_observations(args)

        # inference the place position
        plc_trans = None
        plc_rot = None
        
        if args.save_logs:
            imageio.imwrite(f'{logs_dir}/gpt_input_rgb.png', rgb)
            o3d.io.write_point_cloud(f'{logs_dir}/gpt_input_pcd.ply', pcd)
        
        # Infer the place position
        logger.info("Inferring the place position") 
        # plc = PlacePolicy(args, logs_dir=logs_dir) 
        # plc_trans, mask = plc.run_placement_pipeline(instruction=args.instruction, 
        #                                              target_obj_path_relative=None, 
        #                                              rgb_img=rgb, 
        #                                              points=points, 
        #                                              colors=colors)
        logger.info(f'The place translation is: {plc_trans}') 
        plt.imshow(mask.reshape(360, 640))
        plt.show()
        
        # Infer the place rotation 
        plc_rot = None
        logger.info(f'The place rotation is: {plc_rot}')

        # Infer the grasp pose
        logger.info("Inferring the grasp pose")
        grasp, pre_grasp = get_grasp_pose(args, 
                                            rgb, 
                                            irl, 
                                            irr,
                                            log_dirs=logs_dir, 
                                            target_mask=mask)

        # Modify the grasp pose
        logger.info("Modify the grasp pose")
        ref_vec = np.array([init_pose.pose.position.x, init_pose.pose.position.y, 0.0])
        grasp_pose, pre_grasp_pose = modify_grasp_pose(grasp, pre_grasp, ref_vec)
        logger.info(f'Grasp pose:\n {grasp_pose}')

        # Calculate the place pose
        place_pose = deepcopy(pre_grasp_pose)
        place_pose[2] += 0.1
        if plc_trans is not None:
            place_pose[:3] = plc_trans
        if plc_rot is not None:
            place_pose[3:] = plc_rot
        logger.info(f'Place pose:\n {place_pose}')
        z_low = 0.65
        if place_pose[2] < z_low:
            place_pose[2] = z_low
            logger.warning(f"The place pose is too low, raise it to {z_low}m")
        
        # execute the pick and place
        mid_pose = deepcopy(place_pose)
        mid_pose[1] += 0.25
        pick_and_place(pick_pose=grasp_pose, 
                       place_pose =place_pose, 
                       pre_pick_pose=pre_grasp_pose, 
                       mid_pose=mid_pose)
    
    # level3 demo
    if args.demo_level == 'p3':
        instruction_list = [
            'place the pot lid to the right of the orange',
            'place the orange into the black pot',
            'place the round handle on top of the black pot'
                            ]
        
        for i, instruction in enumerate(instruction_list):
            if args.save_logs:
                time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
                instruction_name = instruction.replace(' ', '_')
                logs_dir = os.path.abspath(f'{args.logs_dir}/{instruction_name}/{time_str}')
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir, exist_ok=True)
                logger.info(f'Log directory: {logs_dir}')  
                          
            # Get the rgb image, points and colors
            logger.info("Getting the observations from the camera")
            pcd, points, colors, depth, rgb, irl, irr = get_observations(args)

            # inference the place position
            plc_trans = None
            plc_rot = None
            
            if args.save_logs:
                imageio.imwrite(f'{logs_dir}/gpt_input_rgb.png', rgb)
                o3d.io.write_point_cloud(f'{logs_dir}/gpt_input_pcd.ply', pcd)
            
            # Infer the place position
            logger.info("Inferring the place position") 
            plc = PlacePolicy(args, logs_dir=logs_dir) 
            plc_trans, mask = plc.run_placement_pipeline(instruction=instruction, 
                                                         target_obj_path_relative=None, 
                                                         rgb_img=rgb, 
                                                         points=points, 
                                                         colors=colors)
            logger.info(f'The place translation is: {plc_trans}') 
            plt.imshow(mask.reshape(360, 640))
            plt.show()
            
            # Infer the place rotation 
            plc_rot = None
            logger.info(f'The place rotation is: {plc_rot}')

            # Infer the grasp pose
            logger.info("Inferring the grasp pose")
            
            mvsnet = MVSGSNetEval(args)
            pts, clr = mvsnet.infer_rt(rgb, irl, irr)
            grasp_group = inference_rt(pts, clr, args, log_dirs=logs_dir)
            
            grasp, pre_grasp = grasp_transfer(grasp_group, log_dirs=logs_dir)
            
            
            # grasp, pre_grasp = get_grasp_pose(args, 
            #                                   rgb, 
            #                                   irl, 
            #                                   irr,
            #                                   log_dirs=logs_dir, 
            #                                   target_mask=mask)

            # Modify the grasp pose
            logger.info("Modify the grasp pose")
            ref_vec = np.array([init_pose.pose.position.x, init_pose.pose.position.y, 0.0])
            grasp_pose, pre_grasp_pose = modify_grasp_pose(grasp, pre_grasp, ref_vec)
            logger.info(f'Grasp pose:\n {grasp_pose}')

            # Calculate the place pose
            place_pose = deepcopy(pre_grasp_pose)
            place_pose[2] += 0.15
            if plc_trans is not None:
                place_pose[:3] = plc_trans
            if plc_rot is not None:
                place_pose[3:] = plc_rot
            logger.info(f'Place pose:\n {place_pose}')
            z_low = 0.35
            if place_pose[2] < z_low:
                place_pose[2] = z_low
                logger.warning(f"The place pose is too low, raise it to {z_low}m")
            
            # execute the pick and place
            pick_and_place(grasp_pose, place_pose, pre_grasp_pose)
    
    if args.demo_level == 'o1':
        if args.save_logs:
            time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
            instruction_name = args.instruction.replace(' ', '_')
            logs_dir = os.path.abspath(f'{args.logs_dir}/{instruction_name}/{time_str}')
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)
            logger.info(f'Log directory: {logs_dir}')
        
        # Get the rgb image, points and colors
        logger.info("Getting the observations from the camera")
        pcd, points, colors, depth, rgb, irl, irr = get_observations(args)

        # inference the place position
        plc_trans = None
        plc_rot = None
        
        if args.save_logs:
            imageio.imwrite(f'{logs_dir}/gpt_input_rgb.png', rgb)
            o3d.io.write_point_cloud(f'{logs_dir}/gpt_input_pcd.ply', pcd)
        
        # Infer the place position
        logger.info("Inferring the place position") 
        # plc = PlacePolicy(args, logs_dir=logs_dir) 
        # plc_trans, mask = plc.run_placement_pipeline(instruction=args.instruction, 
        #                                              target_obj_path_relative=None, 
        #                                              rgb_img=rgb, 
        #                                              points=points, 
        #                                              colors=colors)
        # logger.info(f'The place translation is: {plc_trans}') 
        # plt.imshow(mask.reshape(360, 640))
        # plt.show()
        
        # Infer the place rotation 
        # rotate -90 degree around y-axis


 

        # rot_matrix = R.from_euler('y', -90, degrees=True).as_matrix()
        # rot_matrix = R.from_euler('x', 30, degrees=True).as_matrix()
        # rot_matrix = R.from_euler('z', -90, degrees=True).as_matrix()


        rot_quat = [0, 0, 0.3, 0, 0, 0.7, 0.7] # this is our tra
        rot_matrix = R.from_quat(rot_quat[3:]).as_matrix()


        rot_matrix = np.linalg.inv(rot_matrix)
        plc_rot = rot_matrix
        logger.info(f'The place rotation matrix is:\n {plc_rot}')

        # Infer the grasp pose
        logger.info("Inferring the grasp pose")
        grasp, pre_grasp, grasp_list, pre_grasp_list = get_grasp_pose(args, 
                                                                      rgb, 
                                                                      irl, 
                                                                      irr,
                                                                      log_dirs=logs_dir, 
                                                                      target_mask = None)

        grasp_pose_list = []    
        pre_grasp_pose_list = []
        for i in range(len(grasp_list)):
            grasp_pose, pre_grasp_pose = modify_grasp_pose(grasp_list[i], pre_grasp_list[i], np.array([1, 0, 0]))
            grasp_pose_list.append(grasp_pose)
            pre_grasp_pose_list.append(pre_grasp_pose)
        

        T_b2e = np.eye(4)
        T_b2e[:3, :3] = R.from_quat(grasp_pose[3:]).as_matrix()
        T_b2e[:3, 3] = grasp_pose[:3]

        # Modify the grasp pose
        logger.info("Modify the grasp pose")
        # ref_vec = np.array([init_pose.pose.position.x, init_pose.pose.position.y, 0.0])
        ref_vec = np.array([1.0, 0.0, 0.0])
        grasp_pose, pre_grasp_pose = modify_grasp_pose(grasp, pre_grasp, ref_vec)
        logger.info(f'Grasp pose:\n {grasp_pose}')
        T_b2e = np.eye(4)
        T_b2e[:3, :3] = R.from_quat(grasp_pose[3:]).as_matrix()
        T_b2e[:3, 3] = grasp_pose[:3]
        logger.info(f'Grasp pose transformation matrix:\n {T_b2e}')
        
        # Calculate the place pose
        T_e2o = np.eye(4)
        T_e2o[:3, 3] = np.array([0.0, 0.0, 0.1])
        T_new2old = np.eye(4)
        T_new2old[:3, :3] = plc_rot
        place_poseT = T_b2e @ T_e2o @ T_new2old @ np.linalg.inv(T_e2o)
        place_poseT[:3, 3] = plc_trans
        place_poseT = place_poseT @ np.linalg.inv(T_e2o)
        
        place_pose = np.zeros(7)
        place_pose[:3] = place_poseT[:3, 3]
        place_pose[3:] = R.from_matrix(place_poseT[:3, :3]).as_quat()
        logger.info(f'Place pose:\n {place_pose}')
        z_low = 0.20
        if place_pose[2] < z_low:
            place_pose[2] = z_low
            logger.warning(f"The place pose is too low, raise it to {z_low}m")

        # Move to the pre-pick point
        logger.info("Move to the pre-pick point")
        # arm.go_to_pose_goal(pre_grasp_pose)
        sts, traj = arm.plan_pose_target(pre_grasp_pose)
        arm.execute_plan(traj)
        hand.open_gripper()
        wpts_list.append(pre_grasp_pose)

        # Move to the pick point 
        logger.info("Move to the pick point")
        plan = arm.movel(grasp_pose)
        arm.execute_plan(plan)
        hand.close_gripper()
        rospy.sleep(1.0) # wait for the gripper to close
        wpts_list.append(grasp_pose)

        # Move to the lift point
        logger.info("Move to the lift point")
        reversed_plan = arm.reverse_traj(plan)
        arm.execute_plan(reversed_plan)
        wpts_list.append(pre_grasp_pose)
        
        # logger.info("Moving to the mid pose")
        # mid_pose = deepcopy(place_pose)
        # mid_pose[1] += 0.25
        arm.go_to_joint_state(np.radians(INIT_JNT))
        wpts_list.append(init_pose)

        # Move to the pre-place point
        # place_pose = [0.5, -0.18, 0.30, -0.3487, -0.3619, -0.5799, 0.6412]
        # plc_jnt = np.array([-18, 33, -19, -106, 74, 47, -97])
        logger.info("Move to the pre-place point")
        pre_place_pose = deepcopy(place_pose)
        pre_place_pose[2] += 0.1
        arm.go_to_pose_goal(pre_place_pose)
        hand.close_gripper()
        wpts_list.append(pre_place_pose)
        
        # Move to the place point
        logger.info("Move to the place point")
        # arm.go_to_joint_state(np.radians(plc_jnt))
        arm.go_to_pose_goal(place_pose)
        rospy.sleep(0.5) # wait for the gripper to open
        hand.open_gripper()
        rospy.sleep(1) # wait for the gripper to open
        wpts_list.append(place_pose)
        
        # Move to the init point
        logger.info("Moving to the start pose")
        arm.go_to_joint_state(np.radians(INIT_JNT))
        hand.open_gripper()
        wpts_list.append(init_pose)
        
    if args.save_logs:
        # wpts = np.array([wpts_list])
        # np.save(f'{logs_dir}/waypoints.npy', wpts)
        with open(f'{logs_dir}/path(x-y-z-qx-qy-qz-qw).txt', 'w') as f:
            for wpt in wpts_list:
                f.write(f'{wpt}\n')
        with open(f'{logs_dir}/candidate_path(x-y-z-qx-qy-qz-qw).txt', 'w') as f:
            for i in range(len(grasp_pose_list)):
                f.write(f'traj{i}:\n')
                f.write(f'{init_pose}\n')
                f.write(f'{pre_grasp_pose_list[i]}\n')
                f.write(f'{grasp_pose_list[i]}\n')
                f.write(f'{pre_grasp_pose_list[i]}\n')
                f.write(f'{init_pose}\n')
                f.write(f'{pre_place_pose}\n')
                f.write(f'{place_pose}\n')
                f.write(f'{init_pose}\n')
                f.write(f'\n')
        
    logger.success("Demo finished")

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    # ASGrasp
    parser.add_argument('--robot', type=str, default="panda", help="robot name")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
                        help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true',
                        help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels',
                        type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'],
                        help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--num_sample', type=int, default=96, help="number of depth levels")
    parser.add_argument('--depth_min', type=float, default=0.2, help="number of levels in the correlation pyramid")
    parser.add_argument('--depth_max', type=float, default=1.5, help="width of the correlation pyramid")
    parser.add_argument('--train_2layer', default=True, help="dmtet file path")
    parser.add_argument('--train_ir1_depth', default=False, help="dmtet file path")
    parser.add_argument('--train_dreds', default=False, help="dmtet file path")
    parser.add_argument('--restore_ckpt',
                        default=f'/workspace/src/argrasp/lib/ASGrasp/checkpoints/e2e_s5_raft-stereo.pth',
                        help="restore checkpoint")
    
    # Demo
    parser.add_argument('--save_ply', type=bool, default=False, help="save the point clouds")        
    parser.add_argument('--grasp_visualize', type=bool, default=True, help="visualize the grasp pose")    
    parser.add_argument('--go_home', type=bool, default=False, help="go to the home position")
    parser.add_argument('--save_logs', type=bool, default=True, help="whether to save the logs")
    parser.add_argument('--logs_dir', type=str, default=f'../logs', help="path to the logs directory")
    parser.add_argument('--demo_level', type=str, default='o1', 
                        help="set demo level: p1 for single position task, p2 for single position-relationship task, p3 for sequence position task, o1 for orientation task, po1 for combined position-orientation task")   
    
    
    # p1
    # parser.add_argument('--instruction', type=str, default='move the ball directly in front of the metal mug', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='move the capybara to the right of the mug', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='move the juice box into the red bowl', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='move the flashlight on top of the book', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='place a piece of tissue paper on top of the screwdriver', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='place the capybara on top of the metal mug', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='place the toy building blocks into the red bowl', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='place the base ball into the drawer cabinet', help="text instruction") 

    # p2
    # parser.add_argument('--instruction', type=str, default='place the base ball piece between the spoon and the screwdriver', help="text instruction") 
    # parser.add_argument('--instruction', type=str, default='place the green ball piece in the center of the flashlight, the capybara and the screwdriver', help="text instruction") 
    
    # p3
    
    # o1
    # parser.add_argument('--instruction', type=str, default='place the bottle upright on top of the red bowl', help="text instruction")
    # parser.add_argument('--instruction', type=str, default='Place the mug on the book and turn its handle to the right', help="text instruction")
    # parser.add_argument('--instruction', type=str, default='Place the bottle on the desk and turn its bottleneck to the right', help="text instruction")
    parser.add_argument('--instruction', type=str, default='Place the bottle upside down on table top', help="text instruction")
    
    # op
    # parser.add_argument('--instruction', type=str, default='place the can upright on the book', help="text instruction")
    
    args = parser.parse_args()
    
    try:
        rospy.init_node("demo_placement", anonymous=True)
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e

    