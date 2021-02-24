import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud('D:/PointCloud/Point-Feature-Histogram-master/data/90_single_3_pre_db.pcd')

np.save('D:/PointCloud/Point-Feature-Histogram-master/data/90_single_3_pre_db.npy', np.asarray(pcd.points))