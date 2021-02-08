import open3d as o3d
import numpy as np


raw_file = "D:/thantham/Project/data/raw/180_single_1.ply"

pcd_file = "D:/thantham/Project/data/database/180_plane.pcd"
des_file = "D:/thantham/Project/data/database/180_plane_des.des"

# %% Read
pcd = o3d.io.read_point_cloud(raw_file)


# %% Vox

voxel_size = 0.0025
pcd_vox = pcd.voxel_down_sample(voxel_size=voxel_size)

# %% outlier removal

cl, ind = pcd_vox.remove_radius_outlier(nb_points=50, radius=0.05)

pcd_vox_out = pcd_vox.select_by_index(ind)

# %% remove plane

plane , inliers = pcd_vox_out.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

pcd_vox_out_pln = pcd_vox_out.select_by_index(inliers, invert=True)

# %% compute fpfh

radius_feature = voxel_size * 5

pcd_vox_out_pln_des = o3d.pipelines.registration.compute_fpfh_feature(pcd_vox_out_pln, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# %% visualize

o3d.visualization.draw([pcd_vox_out_pln])


# %% save

o3d.io.write_point_cloud(pcd_file, pcd_vox_out_pln, write_ascii=True)
o3d.io.write_feature(des_file, pcd_vox_out_pln_des)
