import open3d as o3d
import sys

pcd = o3d.io.read_point_cloud('D:/PointCloud/Project/data/raw/180_fo_1.ply')

o3d.visualization.draw([pcd])