import copy
import numpy as np
import open3d as o3d

camera_distance = 180

xyz = np.array([[-2,1,0],
                [-2,-1,0],
                [2,1,0],
                [2,-1,0],
                
                [0,2,1],
                [0,2,-1],
                [0,-2,-1],
                [0,-2,1],
                
                [-1,0,2],
                [1,0,2],
                [1,0,-2],
                [-1,0,-2]])

xyz = xyz*camera_distance
# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


# o3d.visualization.draw([pcd])

o3d.io.write_point_cloud('D:/thantham/bin_picking_data/3D Pipe file/virtual_cam_position_icosahedron.pcd', pcd, write_ascii=True)
