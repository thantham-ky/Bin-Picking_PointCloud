import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import OPTICS, cluster_optics_dbscan

global_pcd_file = "D:/PointCloud/Project/data/raw/online/90_dou_1.ply"

db_pcd_file = "D:/PointCloud/Project/data/database/90_plane.pcd"
db_des_file = "D:/PointCloud/Project/data/database/90_plane_des.des"

# %% 1 Read pcd

print("[PROCESS] Read Point Cloud ", global_pcd_file)

global_pcd = o3d.io.read_point_cloud(global_pcd_file)


# %% 2 Voxel downasmpling

voxel_size = 0.0025

print("[PROCESS] Voxel Downsampling with size ", voxel_size)
global_pcd_vox = global_pcd.voxel_down_sample(voxel_size = voxel_size)



# %% 3 Remove outlier

print("[PROCESS] Remove Outlier Using Radius Method")
# cl, ind = global_pcd_vox.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
cl, ind = global_pcd_vox.remove_radius_outlier(nb_points=40, radius=0.05)

# o3d.visualization.draw([cl])

# %% 4 Remove plane

global_pcd_vox_plane = global_pcd_vox.select_by_index(ind)

plane_model, inliers = global_pcd_vox_plane.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

print("[PROCESS] Plane Removal")
#o3d.visualization.draw([preprocessed_ply])

[a, b, c, d] = plane_model
print(f"[INFO] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

object_cloud = global_pcd_vox_plane.select_by_index(inliers, invert=True)
# object_cloud.paint_uniform_color([1.0, 0, 0])
plane_cloud = global_pcd_vox_plane.select_by_index(inliers)
         
# o3d.visualization.draw([object_cloud])  

# %% Clustering

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#        object_cloud.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

# max_label = labels.max()

# print("[PROCESS] Point Cloud Clustering by DBSCAN")

# print(f"[INFO] point cloud has {max_label + 1} clusters")


print("[PROCESS] OPTICS Clustering")

object_np = np.asarray(object_cloud.points)

cluster = OPTICS(min_samples=30, xi=.04, min_cluster_size=.04)

cluster.fit(object_np)

labels = cluster.labels_

max_label = labels.max()

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
object_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw([object_cloud])

print("[INFO] ", max_label+1, " cluster found")

# %%% define registration object

print("[PROCESS] Read DB pcd")

db_pcd = o3d.io.read_point_cloud(db_pcd_file)
db_des = o3d.io.read_feature(db_des_file)


# %%% Matching and registration
def execute_global_registration_refine(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    
    distance_threshol_regis = voxel_size * 1.6
    
    # print("\n:: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f.\n" % distance_threshol_regis)
    
    regis_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshol_regis,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshol_regis)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000000, 0.9999))
    
    print(regis_result,"\n\n")
    
    distance_threshold_refine = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f.\n" % distance_threshold_refine)
    refine_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold_refine, regis_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    print("\n[INFO] fitness -> ", refine_result.fitness)
    print("\n[INFO] inlier RMSE -> ", refine_result.inlier_rmse)
    
    print("\n",refine_result,"\n")
    
    return refine_result


# %% For each cluster

print("[PROCESS] Registration for each cluster")

object_list = []
unobject_list = []

for i in range(max_label+1):
    
# limit at max labels

    model_temp = copy.deepcopy(db_pcd)

    object_i = object_cloud.select_by_index(np.transpose(np.where(labels==i)))

    print("[INFO] Object processing: ", i)
    # o3d.visualization.draw([object_i])

# %% Registration

# %%% compute fpfh

    radius_feature = voxel_size * 5

    radius_normal = voxel_size *2

# object_i.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30))

    object_fpfh = o3d.pipelines.registration.compute_fpfh_feature(object_i, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    regis_result = execute_global_registration_refine(db_pcd, object_i, db_des, object_fpfh ,voxel_size)


    print("\nRegistration transformation: \n", regis_result.transformation)
    
    if regis_result.fitness >= 0.2:
        
        object_list.append(model_temp.transform(regis_result.transformation))
    else:
        unobject_list.append(model_temp.transform(regis_result.transformation))

print("[RESULT] ", len(object_list), " objects were recognized from ", max_label+1)


for object_i in object_list:
    object_i.paint_uniform_color([1,0,0])
    
for unobject_i in unobject_list:
    unobject_i.paint_uniform_color([0,1,0])
    
plane_cloud.paint_uniform_color([0.9,0.9,0.9])

o3d.visualization.draw([plane_cloud]+object_list+[object_cloud]+unobject_list)

# fitness_threshold = 0.3

# if regis_result.fitness > fitness_threshold:
#     isMatch = True
# else:
#     isMatch = False

# print("[INFO] ","object ", i+1 ,"Is matched? -> ", isMatch)



