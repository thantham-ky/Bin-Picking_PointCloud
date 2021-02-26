import open3d as o3d
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, cluster_optics_dbscan

import os


partials_dir = "/data/partial_views/partial_views_dodecahedron/"

partialview_des = "/data/database/partial_views/dodecahedron/"
descriptor_des = "/data/database/descriptors/dodecahedron/"

voxel_size = 0.005

# this is for rescale because CAD model created by mm. scale while camera read depth as m.
rescale_factor = 0.001

def rescale_mm_to_m(pcd, factor):
    
    points = np.asarray(pcd.points)

    points = points*factor

    new_pcd = o3d.geometry.PointCloud()

    new_pcd.points = o3d.utility.Vector3dVector(points)
    
    return new_pcd

# %% Read

partials_list = os.listdir(os.getcwd()+partials_dir)

print("[INFO] partial views dir: ", os.getcwd()+partials_dir)
print("[INFO] number of partial views: ", len(partials_list))
print("[INFO] voxel size as ", voxel_size, "\n")

for partial in partials_list:
    
    print("[PROCESS] process file: ", partial)

    pcd = o3d.io.read_point_cloud(os.getcwd()+partials_dir+partial)

    pcd = rescale_mm_to_m(pcd, 0.001)

# %% Vox

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
    
    pcd_vox_des = o3d.pipelines.registration.compute_fpfh_feature(pcd_vox, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
# %% visualize

    # o3d.visualization.draw([pcd_vox])


# %% save

    o3d.io.write_point_cloud(os.getcwd()+partialview_des+os.path.splitext(partial)[0]+"_part.pcd", pcd_vox, write_ascii=True)
    o3d.io.write_feature(os.getcwd()+descriptor_des+os.path.splitext(partial)[0]+"_fpfh.des", pcd_vox_des)
    
print("[INFO] create partial views database completed")
