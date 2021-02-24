import open3d as o3d
import pickle

object_file = 'D:/PointCloud/Project/data/raw/offline/90_down_45.ply'

object_db = 'D:/PointCloud/Project/data/database/120_single_1_pre_db'


# %% Read Object as pcd
pcd = o3d.io.read_point_cloud(object_file)

# %% Find Keypoint
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

pcd.paint_uniform_color([1,0,0])
keypoints.paint_uniform_color([0,0,1])

# o3d.visualization.draw([pcd, keypoints])

# %% conpute descriptor

voxel_size = 0.0025

radius_normal = voxel_size *2

keypoints.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30))

radius_feature = voxel_size * 5

pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
# pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(keypoints, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

# %% save pcd, keypoints and fpfh

o3d.io.write_point_cloud(object_db+'.pcd', pcd ,write_ascii=True)
o3d.io.write_point_cloud(object_db+'_kp.pcd', keypoints, write_ascii=True)
o3d.io.write_feature(object_db+'_des.des', pcd_fpfh)

