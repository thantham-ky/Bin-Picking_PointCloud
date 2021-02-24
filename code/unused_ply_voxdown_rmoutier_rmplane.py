import open3d as o3d

ply_file = 'D:/PointCloud/Project/data/raw/offline/90_down_45.ply'

object_file = 'D:/PointCloud/Project/data/preprocessed/90_down_45_pre.ply'

#%% Read point cloud
ply = o3d.io.read_point_cloud(ply_file)

#o3d.visualization.draw([ply])

# %% Voxel DOwn asmple
voxel_down_ply = ply.voxel_down_sample(voxel_size=0.0025)

# o3d.visualization.draw([voxel_down_ply])

# %%Remove Outlier
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw([inlier_cloud])
    
print("Statistical oulier removal")
cl, ind = voxel_down_ply.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
# display_inlier_outlier(voxel_down_ply, ind)


# %% remove plane

preprocessed_ply = voxel_down_ply.select_by_index(ind)

plane_model, inliers = preprocessed_ply.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

#o3d.visualization.draw([preprocessed_ply])

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

object_cloud = preprocessed_ply.select_by_index(inliers, invert=True)
object_cloud.paint_uniform_color([1.0, 0, 0])
plane_cloud = preprocessed_ply.select_by_index(inliers)
         
o3d.visualization.draw([object_cloud])             

# %% Save object point cloud

o3d.io.write_point_cloud(object_file, object_cloud, write_ascii=True)
