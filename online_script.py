import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import OPTICS

camera_ply_file = "/data/raw/online/90_real_6_pre.ply"

cad_model_file = "/data/cad_models/Pipe_02.ply"

partial_db_dir = "/data/database/partial_views/dodecahedron/"
descriptor_db_dir = "/data/database/descriptors/dodecahedron/"

voxel_size = 0.005

# %% 0 Show info

partial_db_files = os.listdir(os.getcwd()+partial_db_dir)
descriptor_db_files = os.listdir(os.getcwd() +descriptor_db_dir)

print("[INFO] Global point cloud : ", camera_ply_file)
print("[INFO] Descriptors collection: ", descriptor_db_dir)
print("[INFO] Number of reference data: ", len(partial_db_files))


# %% 1 Read pcd

print("\n[PROCESS] Read global Point Cloud ", os.getcwd()+camera_ply_file)
global_pcd = o3d.io.read_point_cloud(os.getcwd()+camera_ply_file)


# %% 2 Voxel downasmpling

print("[PROCESS] Voxel Downsampling with size ", voxel_size)
global_pcd_vox = global_pcd.voxel_down_sample(voxel_size = voxel_size)


# %% 3 Remove outlier

print("[PROCESS] Remove Outlier Using Radius Method")
# cl, ind = global_pcd_vox.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
cl, ind = global_pcd_vox.remove_radius_outlier(nb_points=40, radius=0.05)
object_cloud = global_pcd_vox.select_by_index(ind)
# o3d.visualization.draw([cl])

# %% 4 Remove plane

# global_pcd_vox_plane = global_pcd_vox.select_by_index(ind)


# plane_model, inliers = global_pcd_vox_plane.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

# print("[PROCESS] Plane Removal")
# #o3d.visualization.draw([preprocessed_ply])

# [a, b, c, d] = plane_model
# print(f"[INFO] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# plane_cloud = global_pcd_vox_plane.select_by_index(inliers)
# object_cloud = global_pcd_vox_plane.select_by_index(inliers, invert=True)
# object_cloud.paint_uniform_color([1.0, 0, 0])

# o3d.visualization.draw([object_cloud])  

# %% Clustering

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         object_cloud.cluster_dbscan(eps=0.05, min_points=20, print_progress=True))

# max_label = labels.max()

# print("[PROCESS] Point Cloud Clustering by DBSCAN")

# print(f"[INFO] Point cloud has {max_label + 1} clusters")


# print("[PROCESS] OPTICS Clustering")

# object_np = np.asarray(object_cloud.points)

# cluster = OPTICS(min_samples=30, xi=.04, min_cluster_size=.05)

# cluster.fit(object_np)

# labels = cluster.labels_

# max_label = labels.max()

# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# object_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

# # o3d.visualization.draw([object_cloud])

# print("[INFO] ", max_label+1, " cluster found")

# CASE NO CLUSTERING

max_label = 0

# %%% define registration object

# print("[PROCESS] Read DB pcd")




# %%% Matching and registration
def execute_global_registration_refine(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    
    # Global registration
    distance_threshol_regis = voxel_size * 1.5
    regis_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshol_regis,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), 
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshol_regis),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(0.9)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.7))
    
    # print("[INFO]", regis_result)
    
    # Point cloud refinement
    distance_threshold_refine = voxel_size * 0.5
    refine_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold_refine, regis_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    print("[INFO] fitness -> ", refine_result.fitness)
    print("[INFO] inlier RMSE -> ", refine_result.inlier_rmse)
    
    
    
    return refine_result


# %% For each cluster

print("[PROCESS] Matching and Registration\n")

target_object = None
# unobject_list = []
target_object_pose = None
target_object_cad = None

pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.055, origin = np.array([0., 0., -0.06]))

cad_model = o3d.io.read_triangle_mesh(os.getcwd()+cad_model_file)
cad_model = cad_model.scale(scale=0.001, center=np.array([0,0,0]))
cad_model.compute_vertex_normals()

for each_cluster in range(max_label+1):
    
# limit at max labels

    best_fit = 0.0
    
    best_regis_result = None
    
    object_i = copy.deepcopy(object_cloud)
    
    radius_feature = voxel_size * 5
    
    radius_normal = voxel_size *2
    
    object_i.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30))
    
    object_fpfh = o3d.pipelines.registration.compute_fpfh_feature(object_i, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    

    for partial, descriptor in zip(partial_db_files, descriptor_db_files):
        
        # Read a partial view in db
        db_pcd = o3d.io.read_point_cloud(os.getcwd()+partial_db_dir + partial)
        db_des = o3d.io.read_feature(os.getcwd()+descriptor_db_dir + descriptor)

        #define temp partial view for final transformation
        model_temp = copy.deepcopy(db_pcd)
        axis_temp = copy.deepcopy(pose_axis)
        cad_temp = copy.deepcopy(cad_model)
    
        # object_i = object_cloud.select_by_index(np.transpose(np.where(labels==each_cluster)))
    
        
        print("[INFO] Matching to partial view: ", partial)
        # o3d.visualization.draw([db_pcd])
    
    # %% Registration
    
    # %%% compute fpfh
    
       
        regis_result = execute_global_registration_refine(db_pcd, object_i, db_des, object_fpfh ,voxel_size)
    
    
        # print("\nRegistration transformation: \n", regis_result.transformation)
        
        
        if regis_result.fitness >= best_fit:
            
            best_fit = regis_result.fitness
            # transformation = regis_result.transformation
            
            best_regis_result = regis_result
            
            
            print("[INFO]-- partial view ", partial, " is best fit, find next... ****************\n")
            
        else:
            # unobject_list.append(model_temp.transform(regis_result.transformation))
            print("[INFO]-- partial view ", partial, " not fit, find next...\n")
            
            

    target_object= model_temp.transform(best_regis_result.transformation)     
            
    target_object_pose = axis_temp.transform(best_regis_result.transformation)
            
    target_object_cad = cad_temp.transform(best_regis_result.transformation)
    
    object_center_on_axis = target_object_cad.get_center()

print("\n[RESULT] ", " objects detected at ",  object_center_on_axis)


# for object_i in object_list:
target_object.paint_uniform_color([1,0,0])
    
# for unobject_i in unobject_list:
#     unobject_i.paint_uniform_color([0,1,0])
    
# plane_cloud.paint_uniform_color([0.9,0.9,0.9])


# print("[INFO] found object was located at ", object_center_on_axis)

o3d.visualization.draw_geometries([object_cloud, 
                                   target_object, 
                                   target_object_pose, 
                                   target_object_cad],
                                  mesh_show_wireframe=False,
                                  point_show_normal=False,
                                  mesh_show_back_face=True)
# o3d.visualization.draw_geometries([object_list[0].create_arrow()])

# fitness_threshold = 0.3

# if regis_result.fitness > fitness_threshold:
#     isMatch = True
# else:
#     isMatch = False

# print("[INFO] ","object ", i+1 ,"Is matched? -> ", isMatch)



