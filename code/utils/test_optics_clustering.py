from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import open3d as o3d

x = np.asarray(object_cloud.points)

model = OPTICS(min_samples=20, xi=.05, min_cluster_size=.05)

model.fit(x)


pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(x)

label = model.labels_

pcd_select = pcd.select_by_index(np.transpose(np.where(label==1)))
o3d.visualization.draw([pcd_select])

