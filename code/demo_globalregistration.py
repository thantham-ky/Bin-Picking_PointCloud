import open3d as o3d
import copy
import numpy as np

source_pcd_file = 'D:/PointCloud/Project/data/database/90_single_1_pre_db.pcd'
source_des_file = 'D:/PointCloud/Project/data/database/90_single_1_pre_db_des.des'

target_pcd_file = 'D:/PointCloud/Project/data/database/120_single_1_pre_db.pcd'
target_des_file = 'D:/PointCloud/Project/data/database/120_single_1_pre_db_des.des'

# %% define function

def visualize_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])
    
def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    
    distance_threshold = voxel_size * 1.5
    
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(10000, 0.999))
    
    return result

# def execute_fast_global_registration(source_down, target_down, source_fpfh,
#                                      target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 0.5
#     print(":: Apply fast global registration with distance threshold %.3f" \
#             % distance_threshold)
#     result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh,
#         o3d.pipelines.registration.FastGlobalRegistrationOption(
#             maximum_correspondence_distance=distance_threshold))
#     return result

# %% Read
source_pcd = o3d.io.read_point_cloud(source_pcd_file)
source_des = o3d.io.read_feature(source_des_file)

target_pcd = o3d.io.read_point_cloud(target_pcd_file)
target_des = o3d.io.read_feature(target_des_file)

voxel_size = 0.0025

# result_ransac = execute_fast_global_registration(source_pcd, target_pcd, source_des, target_des, voxel_size)
result_ransac = execute_global_registration(source_pcd, target_pcd, source_des, target_des,voxel_size)

print(result_ransac)

transform_init = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])

# visualize_registration_result(source_pcd, target_pcd, result_ransac.transformation)
# visualize_registration_result(source_pcd, target_pcd, transform_init)

# %% Local refinement

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


result_icp = refine_registration(source_pcd, target_pcd, source_des, target_des,voxel_size)

print(result_icp)
visualize_registration_result(source_pcd, target_pcd, result_icp.transformation)
