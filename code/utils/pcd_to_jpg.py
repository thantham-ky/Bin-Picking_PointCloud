import open3d as o3d

pcd = o3d.io.read_triangle_mesh("D:/PointCloud/Project/data/raw/online/90_real_21_pre.ply")
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
vis.get_render_option().point_size = 3.0
vis.add_geometry(pcd)
vis.capture_screen_image("D:/PointCloud/Project/data/90_real_21_pre.jpg", do_render=True)
vis.destroy_window()