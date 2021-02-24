import numpy as np
import open3d as o3d

pt_xyz = np.array([[0,0.02,-0.04]])

transformation = regis_result.transformation

pt_pcd = o3d.geometry.PointCloud()

pt_pcd.points = o3d.utility.Vector3dVector(pt_xyz)

pt_pcd.transform(transformation)

pt_new = np.array(pt_pcd.points)

print(pt_new)

