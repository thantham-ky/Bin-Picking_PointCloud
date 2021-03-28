import open3d as o3d
import numpy as np
import pyvista as pv

camera_pc = "D:/thantham/Project/Bin-Picking_PointCloud/data/virtual_cam_position/virtual_cam_position_dodecahedron.pcd"
cad_mesh = "D:/thantham/Project/Bin-Picking_PointCloud/data/cad_models/Pipe_02.ply"

camera = o3d.io.read_point_cloud(camera_pc)
points = np.asarray(camera.points)

cloud = pv.PolyData(points)
# cloud.plot()

volume = cloud.delaunay_3d()
shell = volume.extract_geometry()
# shell.plot()

cad = pv.read(cad_mesh)


p = pv.Plotter()

p.add_mesh(volume, ambient=0., opacity=0.2)
p.add_mesh(cad, color='lightblue')
p.add_points(cloud, point_size=20, color='darkgray')

p.set_background(color='white')
p.show(cpos='xy')
# o3d.visualization.draw_geometries([mesh, cad])

# o3d.io.write_triangle_mesh("D:/thantham/Project/Bin-Picking_PointCloud/data/virtual_cam_position/virtual_cam_position_dodecahedron.ply", camera)
