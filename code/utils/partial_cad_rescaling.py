import open3d as o3d
import numpy as np

pcd_file = "D:/PointCloud/Project/data/raw/offline/180_cad_plane.ply"

pcd_save_to = "D:/PointCloud/Project/data/raw/offline/180_cad_plane_rsc.pcd"

pcd = o3d.io.read_point_cloud(pcd_file)

# pcd_vox = pcd.voxel_down_sample(voxel_size=0.0025)

# o3d.visualization.draw([pcd])

points = np.asarray(pcd.points)

points = points*0.001

new_pcd = o3d.geometry.PointCloud()

new_pcd.points = o3d.utility.Vector3dVector(points)

# new_pcd = new_pcd.voxel_down_sample(voxel_size = 0.0025)

# o3d.visualization.draw([new_pcd])

o3d.io.write_point_cloud(pcd_save_to, new_pcd, write_ascii=True)
