import copy
import numpy as np
import open3d as o3d
import math


l = 1
camera_distance = 250


# Golden ratio formula
phi = (1+math.sqrt(5))/2

xyz = np.array([
                # A B C D
                 [phi,  0, l/phi],
                [-phi, 0, l/phi],
                [-phi, 0, -l/phi],
                [ phi,  0, -l/phi],
                # E F G H
                [l/phi,  phi,  0],
                [l/phi,  -phi, 0],
                [-l/phi, -phi, 0],
                [-l/phi, phi,  0],
                # I J K L
                [0, l/phi,  phi],
                [0, l/phi,  -phi],
                [0, -l/phi, -phi],
                [0, -l/phi, phi],
                # M N O P Q R S T
                [ l,   l,  l],
                [ l,  -l,  l],
                [-l, -l,  l],
                [-l,  l,  l],
                [-l,  l, -l],
                [ l,   l, -l],
                [ l,  -l, -l],
                [-l, -l, -l]])

xyz = xyz*camera_distance

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


# o3d.visualization.draw([pcd])
o3d.io.write_point_cloud('D:/thantham/bin_picking_data/3D Pipe file/virtual_cam_position_dodecahedron.pcd', pcd, write_ascii=True)
