import open3d as o3d
import sys

pcd_model = o3d.io.read_point_cloud('D:/PointCloud/Project/data/cad_models/Pipe_02.ply')
pcd_cam = o3d.io.read_point_cloud('D:/PointCloud/Project/data/virtual_cam_position/virtual_cam_position_dodecahedron.pcd')

o3d.visualization.draw([pcd_model,pcd_cam])