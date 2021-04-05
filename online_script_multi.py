import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import OPTICS
from scipy.spatial.transform import Rotation as R


import time

camera_ply_file = "/data/raw/online/90_real_21_pre.ply"

cad_model_file = "/data/cad_models/Pipe_02.ply"

partial_db_dir = "/data/database/partial_views/dodecahedron/"
descriptor_db_dir = "/data/database/descriptors/dodecahedron/"

voxel_size = 0.005

fitness_threshold = 0.5

# %% 0 Show info

partial_db_files = os.listdir(os.getcwd()+partial_db_dir)
descriptor_db_files = os.listdir(os.getcwd() +descriptor_db_dir)

print("[INFO] Global point cloud : ", camera_ply_file)
print("[INFO] Descriptors collection: ", descriptor_db_dir)
print("[INFO] Number of reference data: ", len(partial_db_files))
t0 = time.time()

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
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.7))
    
    # print("[INFO]", regis_result)
    
    # Point cloud refinement
    distance_threshold_refine = voxel_size * 0.5
    refine_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold_refine, regis_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # print("[INFO] fitness -> ", refine_result.fitness)
    # print("[INFO] inlier RMSE -> ", refine_result.inlier_rmse)
    
    return refine_result

def detect_object(pc_scene, fitness_threshold=0.5, voxel_size=0.005):
    
    best_fitness = 0.0
    
    best_regis_result = None
    pc_object = None
    
    radius_feature = voxel_size * 5
    radius_normal = voxel_size *2
    
    pc_scene.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30))
    
    pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pc_scene, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    from tqdm import tqdm
    pbar = tqdm(total=len(partial_db_files), desc='find object')
    
    for partial, descriptor in zip(partial_db_files, descriptor_db_files):
        
        # Read a partial view in db
        db_pcd = o3d.io.read_point_cloud(os.getcwd()+partial_db_dir + partial)
        db_des = o3d.io.read_feature(os.getcwd()+descriptor_db_dir + descriptor)
    
        regis_result = execute_global_registration_refine(db_pcd, pc_scene, db_des, pc_fpfh ,voxel_size)
                 
        if (regis_result.fitness >= best_fitness) & (regis_result.fitness >= fitness_threshold):
            
            best_fitness = regis_result.fitness
            
            best_regis_result = regis_result         
            pc_object = db_pcd.transform(regis_result.transformation)

        pbar.update(1)
    pbar.close()
    
    
    return pc_object, best_regis_result



def remove_object_from_bin(pc_scene, pc_obj, dist_threshold=0.01):
    select_points = []

    current_scene = np.asarray(pc_scene.points)
    current_object = np.asarray(pc_obj.points)
    
    from tqdm import tqdm
    pbar = tqdm(total=current_scene.shape[0], desc='removing overlapping points')
    for i in range(current_scene.shape[0]):
        if min([np.linalg.norm(current_scene[i]-current_object[j]) for j in range(current_object.shape[0])]) >= dist_threshold:
            select_points.append(current_scene[i])
        pbar.update(1)
    pbar.close()       
    select_points = np.asarray(select_points)
            
    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(select_points)
    
    return new_pc

def generate_cad_model_to_scene(trans_list):
    
    cad_list = []
    axis_list = []
    xyz_list = []
    rot_list = []
    
    cad_model = o3d.io.read_triangle_mesh(os.getcwd()+cad_model_file)
    cad_model = cad_model.scale(scale=0.001, center=np.array([0,0,0]))
    cad_model.compute_vertex_normals()
    
    
    for trans_i in trans_list:
        # # create axis frame
        pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.055, origin = np.array([0., 0., -0.06]))

        # # create cad model mesh
        cad_temp = copy.deepcopy(cad_model)
        
        target_object_pose = pose_axis.transform(trans_i.transformation)
        target_object_cad = cad_temp.transform(trans_i.transformation)
        cad_list.append(target_object_cad)
        axis_list.append(target_object_pose)
    
        # get x y z of object
        object_center_on_axis = target_object_cad.get_center()
        xyz_list.append(object_center_on_axis)
        
        # get rotation pose of object
        rotation_matrix = trans_i.transformation[:3,:3]
        r = R.from_matrix(rotation_matrix.tolist())
        rotation = r.as_euler('xyz', degrees=True)
        rot_list.append(rotation)
        
    return cad_list, axis_list, xyz_list, rot_list

# %% MAIN 
done = False

object_list = []
transform_list = []

find_scene = object_cloud

while not done:
    
    found_object, found_trans = detect_object(find_scene, fitness_threshold=fitness_threshold, voxel_size=0.005)
    
    if found_object == None:
        done = True
        print("No more object found")
        continue
    
    found_object.paint_uniform_color([1,0,0])
    object_list.append(found_object)
    transform_list.append(found_trans)
    
    find_scene = remove_object_from_bin(find_scene, found_object, dist_threshold = voxel_size)
    
virtual_object, virtual_axis, virtual_xyz, virtual_rot = generate_cad_model_to_scene(transform_list)

o3d.visualization.draw_geometries(virtual_object+virtual_axis+[find_scene])
# %%
# print("[PROCESS] Matching and Registration\n")

# # Init detected object
# target_object = None
# target_object_pose = None
# target_object_cad = None
# regis_result_list = []

# # create axis frame
# pose_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.055, origin = np.array([0., 0., -0.06]))

# # create cad model mesh
# cad_model = o3d.io.read_triangle_mesh(os.getcwd()+cad_model_file)
# cad_model = cad_model.scale(scale=0.001, center=np.array([0,0,0]))
# cad_model.compute_vertex_normals()

# # Loop case of multiple cluster
# for each_cluster in range(1):

#     # init parameter
#     best_fit = 0.0
    
#     best_regis_result = None
    
#     best_transform = None
    
#     object_i = copy.deepcopy(object_cloud)
    
#     radius_feature = voxel_size * 5
    
#     radius_normal = voxel_size *2
    
#     # estimate normal and compute descriptor for global point cloud
#     object_i.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn=30))
    
#     object_fpfh = o3d.pipelines.registration.compute_fpfh_feature(object_i, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    

#     # Loop start - for each partial views from database
#     for partial, descriptor in zip(partial_db_files, descriptor_db_files):
        
#         # Read a partial view in db
#         db_pcd = o3d.io.read_point_cloud(os.getcwd()+partial_db_dir + partial)
#         db_des = o3d.io.read_feature(os.getcwd()+descriptor_db_dir + descriptor)

#         #define temp partial view for final transformation
#         model_temp = copy.deepcopy(db_pcd)
#         axis_temp = copy.deepcopy(pose_axis)
#         cad_temp = copy.deepcopy(cad_model)
    
#         # object_i = object_cloud.select_by_index(np.transpose(np.where(labels==each_cluster)))
    
        
#         print("[INFO] Matching to partial view: ", partial)
#         # o3d.visualization.draw([db_pcd])
    
       
#         regis_result = execute_global_registration_refine(db_pcd, object_i, db_des, object_fpfh ,voxel_size)
    
    
#         # print("\nRegistration transformation: \n", regis_result.transformation)
        
        
#         if (regis_result.fitness >= best_fit) & (regis_result.fitness >= fitness_threshold):
            
#             best_fit = regis_result.fitness
#             # transformation = regis_result.transformation
            
#             best_regis_result = regis_result
            
#             regis_result_list.append(regis_result)
            
            
#             print("[INFO]-- partial view ", partial, " is best fit, find next... ****************\n")
            
#         else:
        
#             print("[INFO]-- partial view ", partial, " not fit, find next...\n")
            
#         # if (regis_result.fitness >= fitness_threshold):
            
#         #     # best_fit = regis_result.fitness
#         #     # transformation = regis_result.transformation
            
#         #     # best_regis_result = regis_result
            
#         #     regis_result_list.append(regis_result)
            
            
#         #     print("[INFO]-- partial view ", partial, " is best fit, find next... ****************\n")
            
#         # else:
        
#         #     print("[INFO]-- partial view ", partial, " not fit, find next...\n")
    
    
#     # Loop end
            
#     # Finalize object found in global point cloud
#     if best_regis_result != None:
#         # get best transformation
#         best_transform = best_regis_result.transformation       
        
#         # transform a partial view to global point cloud 
#         target_object= model_temp.transform(best_regis_result.transformation) 
    
#         # transform a axis frame to detected object
#         target_object_pose = axis_temp.transform(best_regis_result.transformation)
            
#         # transform a cad model to detected position
#         target_object_cad = cad_temp.transform(best_regis_result.transformation)
    
#         # get x y z of object
#         object_center_on_axis = target_object_cad.get_center()
    
#         # get rotation pose of object
#         rotation_matrix = best_transform[:3,:3]
#         r = R.from_matrix(rotation_matrix.tolist())
#         rotation = r.as_euler('xyz', degrees=True)

#     else:
#         object_center_on_axis = 'No axis'
#         rotation = 'No rotation'
#     time.sleep(2)

#     print("\n[RESULT] ", " the object center located at [x y z]:",  object_center_on_axis)
#     print("[RESULT] ", " the object rotation is (euler-degrees)[x y z]:",  rotation)
#     print("[REMARK] ", " the object position referenced from camera origin")
    


# t1 = time.time()
# print ("[TIME]"," process complete in ",(t1-t0), " sec")
# # # for object_i in object_list:
# target_object.paint_uniform_color([1,0,0])
    
# # for unobject_i in unobject_list:
# #     unobject_i.paint_uniform_color([0,1,0])
    
# # plane_cloud.paint_uniform_color([0.9,0.9,0.9])

# # print("[INFO] found object was located at ", object_center_on_axis)

# o3d.visualization.draw_geometries([object_cloud, 
#                                     target_object], 
#                                    # target_object_pose, 
#                                 #    target_object_cad],
#                                   mesh_show_wireframe=False,
#                                   point_show_normal=False,
#                                   mesh_show_back_face=True)
# o3d.visualization.draw_geometries([object_list[0].create_arrow()])

# %% filter

# %%

# %% gen new 


