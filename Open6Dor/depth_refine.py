import numpy as np
import open3d as o3d
import plotly
def backproject_depth_to_pointcloud(depth_img, fx, fy, cx, cy, near, far):
    # Convert normalized depth to actual depth
    # actual_depth = near + depth_img * (far - near)
    actual_depth = near / (1 - depth_img * (1 - near / far))


    # Get image dimensions
    height, width = depth_img.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Backproject to 3D coordinates
    z = actual_depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into a (N, 3) point cloud
    point_cloud = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    return point_cloud

# Example usage with camera parameters
# intrinsic_mat = np.array([[154.50966799,   0.        ,  64.        ],
#        [  0.        , 154.50966799,  64.        ],
#        [  0.        ,   0.        ,   1.        ]])
intrinsic_mat_open6dor = np.array([[309.01933598,   0.        , 128.        ],
       [  0.        , 309.01933598, 128.        ],
       [  0.        ,   0.        ,   1.        ]])
extrinsic_mat_agent = np.array([[-7.26818450e-08,  6.28266450e-01, -7.77998244e-01,
         6.58613175e-01],
       [ 1.00000000e+00, -7.26818450e-08, -1.52115266e-07,
         0.00000000e+00],
       [-1.52115266e-07, -7.77998244e-01, -6.28266450e-01,
         1.61035002e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]) #agentview
extrinsic_mat_front = np.array([[-5.55111512e-17,  2.58174524e-01, -9.66098295e-01,  1.00000000e+00],
 [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
 [ 0.00000000e+00, -9.66098295e-01, -2.58174524e-01,  1.48000000e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]) #frontview

extrinsic_mat_ours = np.array([[ 0.        , -0.76604444,  0.64278761, -0.5       ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.64278761, -0.76604444,  1.45      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_oursyf = np.array([[ 0.        , -0.76604444,  0.64278761, -0.5       ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.64278761, -0.76604444,  1.26      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_cal = np.array([[ 0.        , -0.76604444,  0.64278761, -0.5       ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.64278761, -0.76604444,  1.26      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_cal1 = np.array([[ 0.76604444,  0.        , 0.64278761, -0.5       ],
       [ 0.64278761,  0.        ,  -0.76604444, 0     ],
       [ 0.        , 1.        ,  0.        ,  1.26      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_test = np.array([[ 0.        , -1.        ,  0.     ,-0.5   ],
    [ 0.76604444,  0.        , -0.64278761, 0.     ],
       [ 0.64278761,  0.        ,  0.76604444, 1.26      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_test1 = np.array([[ 0.        ,  0.76604444, -0.64278761, -0.5       ],
       [-1.        ,  0.        ,  0.   ,0     ],
       [ 0.        ,  0.64278761,  0.76604444,  1.26      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_final = np.array([[ 0.        , -0.42261825,  0.90630779, -0.72      ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.90630779, -0.42261825,  1.15      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_final = np.array([[ 0.        , -0.57357643,  0.81915205, -0.73      ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.81915205, -0.57357643,  1.5       ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_final1 = np.array([[ 0.        , -0.57357643,  0.81915205, -0.65      ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.81915205, -0.57357643,  1.16      ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
extrinsic_mat_6dor = np.array([[-7.26818450e-08,  6.28266450e-01, -7.77998244e-01,
         1.45861317e+00],
       [ 1.00000000e+00, -7.26818450e-08, -1.52115266e-07,
         0.00000000e+00],
       [-1.52115266e-07, -7.77998244e-01, -6.28266450e-01,
         1.71035002e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]]) #agent

extrinsic_mat_6dor_front_final = np.array([[-5.55111512e-17,  2.58174524e-01, -9.66098295e-01,
         1.60000000e+00],
       [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00],
       [ 0.00000000e+00, -9.66098295e-01, -2.58174524e-01,
         1.30000000e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])

extrinsic_mat_test = np.array([[ 0.        , -0.57357643,  0.81915205,  0.1       ],
       [-1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        , -0.81915205, -0.57357643,  1.2       ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])


# near = 0.01060981611847304  # Near clipping plane
# far = 530.490780726692  # Far clipping plane

# 6dor
near = 0.01183098675640314
far = 591.5493097230725
fx = intrinsic_mat_open6dor[0, 0]
fy = intrinsic_mat_open6dor[1, 1]
cx = intrinsic_mat_open6dor[0, 2]
cy = intrinsic_mat_open6dor[1, 2]
extrinsic_mat = extrinsic_mat_6dor_front_final#extrinsic_mat_6dor_front_final # extrinsic_mat_test
depth_np = np.load("/data/benchmark/test/depth_open6dor_ours_test6.npy").squeeze()
rgb_np = np.load('/data/benchmark/test/rgb_open6dor_ours_test6.npy').squeeze()

depth_np = np.flip(depth_np,  axis=0)
rgb_np = np.flip(rgb_np, axis=0)
# depth_img should be your depth image from mjr_readPixels
point_cloud_np = backproject_depth_to_pointcloud(depth_np, fx, fy, cx, cy, near, far)
# LIBERO MASK
# mask = point_cloud_np[:,2] < 1.6
# Open6DOR MASK
mask = point_cloud_np[:,2] < 2
mask &= point_cloud_np[:,1] < 2
rgb_img = rgb_np.reshape(-1, 3)  # Flatten the RGB image
rgb_img = rgb_img[mask]  # Apply the mask to RGB colors

# mask out z>= 2

point_cloud_np = point_cloud_np[mask]

# apply extrinsic transformation
# points_homogeneous = np.hstack((point_cloud_np, np.ones((point_cloud_np.shape[0], 1))))  # Add homogeneous coordinate
# transformed_points = (extrinsic_mat @ points_homogeneous.T).T[:, :3]  # Apply transformation
# point_cloud_np = transformed_points

point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud_np)
point_cloud_o3d.colors = o3d.utility.Vector3dVector(rgb_img / 255.0) 
# output = 'data/workspace/LIBERO/experiments/pc/bird_glob.ply'

output = '/data/workspace/LIBERO/experiments/pc/depth_open6dor_ours_agent_test6_cam.ply'
print(f"Saving point cloud to {output}")
o3d.io.write_point_cloud(output, point_cloud_o3d)
# viusulize the point cloud with go
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter3d(x=point_cloud_np[:,0], y=point_cloud_np[:,1], z=point_cloud_np[:,2], mode='markers', marker=dict(size=2, color=rgb_img, opacity=0.8))])
fig.show()