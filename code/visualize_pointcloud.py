import open3d as o3d
import sys

pcd_model = o3d.io.read_point_cloud('D:/PointCloud/Project/data/database/partial_views/dodecahedron/1_part.pcd')
# pcd_cam = o3d.io.read_point_cloud('D:/PointCloud/Project/data/virtual_cam_position/virtual_cam_position_dodecahedron.pcd')

o3d.visualization.draw_geometries([pcd_model])
