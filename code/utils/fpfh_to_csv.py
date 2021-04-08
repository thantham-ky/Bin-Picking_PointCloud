import open3d as o3d
import numpy

pcd = o3d.io.read_point_cloud('D:/PointCloud/Project/data/database/partial_views/dodecahedron/9_part.pcd')
feature = o3d.io.read_feature('D:/PointCloud/Project/data/database/descriptors/dodecahedron/9_fpfh.des')

feature = object_fpfh
fpfh = feature.data.transpose()

o3d.visualization.draw([pcd])
numpy.savetxt("D:/PointCloud/Project/90_test_real_30_hist.csv", fpfh, delimiter=",")
