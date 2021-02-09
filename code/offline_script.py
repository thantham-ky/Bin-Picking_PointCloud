import open3d as o3d
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan


raw_file = "D:/PointCloud/Project/data/raw/offline/90_down_h_pre.ply"

pcd_file = "D:/PointCloud/Project/data/database/90_down_h_pre.pcd"
des_file = "D:/PointCloud/Project/data/database/90_down_h_pre_des.des"

# %% Read
pcd = o3d.io.read_point_cloud(raw_file)


# %% Vox

voxel_size = 0.0025
pcd_vox = pcd.voxel_down_sample(voxel_size=voxel_size)

# %% outlier removal

# cl, ind = pcd_vox.remove_radius_outlier(nb_points=30, radius=0.05)

# pcd_vox_out = pcd_vox.select_by_index(ind)

# # %% remove plane

# plane , inliers = pcd_vox_out.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=1000)

# pcd_vox_out_pln = pcd_vox_out.select_by_index(inliers, invert=True)

# # %% cluster

# object_np = np.asarray(pcd_vox_out_pln.points)

# cluster = OPTICS(min_samples=30, xi=.04, min_cluster_size=.04)

# cluster.fit(object_np)

# labels = cluster.labels_

# max_label = labels.max()

# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd_vox_out_pln.colors = o3d.utility.Vector3dVector(colors[:, :3])

# pcd_vox_out_pln_slt = pcd_vox_out_pln.select_by_index(np.transpose(np.where(labels==2)))

# o3d.visualization.draw([pcd_vox])


# %% compute fpfh

radius_normal = voxel_size * 2

radius_feature = voxel_size * 5

pcd_vox.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

pcd_vox_out_pln_des = o3d.pipelines.registration.compute_fpfh_feature(pcd_vox, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# %% visualize

o3d.visualization.draw([pcd_vox])


# %% save

o3d.io.write_point_cloud(pcd_file, pcd_vox, write_ascii=True)
o3d.io.write_feature(des_file, pcd_vox_out_pln_des)
