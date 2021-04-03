import open3d as o3d
import numpy

pcd = o3d.io.read_point_cloud('D:/PointCloud/Project/data/database/partial_views/dodecahedron/10_part.pcd')
feature = o3d.io.read_feature('D:/PointCloud/Project/data/database/descriptors/dodecahedron/10_fpfh.des')
fpfh = feature.data.transpose()

o3d.visualization.draw([pcd])
numpy.savetxt("D:/PointCloud/Project/90_db_part_10_hist.csv", fpfh, delimiter=",")
