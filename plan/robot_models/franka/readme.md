
1. Copy franka_with_gripper_extensions_240510 to curobo/src/curobo/content/assets/robot
2. Copy franka_with_gripper_extensions_240510/franka_with_gripper_extensions_robot_config.yml to curobo/src/curobo/content/configs/robot
3. Copy franka_with_gripper_extensions_240510/franka_with_gripper_extensions_world_config.yml to curobo/src/curobo/content/configs/world
4. open  franka_with_gripper_extensions_240510/examples/demo_franka_with_gripper_extensions_robot_config_franka_with_gripper_extensions_robot_config.usd in isaac sim and click play
4. install galbot_motion_planning [optional] git clone ssh://git@git.galbot.com:6043/robot_skill/galbot_motion_planning.git -b stable
5. python franka_with_gripper_extensions_240510/examples/forward_kinematics.py [optional]
6. python franka_with_gripper_extensions_240510/examples/inverse_kinematics.py [optional]
7. python franka_with_gripper_extensions_240510/examples/motion_generate.py [optional]

