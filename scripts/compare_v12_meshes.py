import open3d as o3d, cv2, numpy as np
from pathlib import Path

v12_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')

mesh_p_path = v12_dir / 'v12_poisson_mesh.ply'
mesh_d_path = v12_dir / 'v12_delaunay_mesh.ply'

mesh_p = o3d.io.read_triangle_mesh(str(mesh_p_path))
mesh_d = o3d.io.read_triangle_mesh(str(mesh_d_path))

print('=== Candidate A: COLMAP Poisson Mesh ===')
print('  Vertices:', len(mesh_p.vertices))
print('  Triangles:', len(mesh_p.triangles))
print('  Bounds min:', mesh_p.get_min_bound())
print('  Bounds max:', mesh_p.get_max_bound())
print('  Is edge manifold:', mesh_p.is_edge_manifold())
print('  Is vertex manifold:', mesh_p.is_vertex_manifold())

print('\n=== Candidate B: COLMAP Delaunay Mesh ===')
print('  Vertices:', len(mesh_d.vertices))
print('  Triangles:', len(mesh_d.triangles))
print('  Bounds min:', mesh_d.get_min_bound())
print('  Bounds max:', mesh_d.get_max_bound())
print('  Is edge manifold:', mesh_d.is_edge_manifold())
print('  Is vertex manifold:', mesh_d.is_vertex_manifold())

# Function to render mesh views
def render_mesh_views(mesh, prefix):
    mesh.compute_vertex_normals()
    
    # Crop to scene bounds for clear visualization
    # X in [-20, 35], Y in [-10, 10], Z in [-10, 35]
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([-20.0, -10.0, -10.0]),
        max_bound=np.array([35.0, 10.0, 35.0])
    )
    m_crop = mesh.crop(bbox)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1200, height=900)
    vis.add_geometry(m_crop)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    
    bb = m_crop.get_axis_aligned_bounding_box()
    c = bb.get_center()
    
    # 1. Oblique View
    ctrl = vis.get_view_control()
    ctrl.set_lookat(c)
    ctrl.set_front([0.8, 0.8, 0.8])
    ctrl.set_up([0, -1, 0])
    ctrl.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    buf_obl = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
    cv2.imwrite(str(v12_dir / f'{prefix}_oblique.png'), cv2.cvtColor(buf_obl, cv2.COLOR_RGB2BGR))
    
    # 2. Top-down View
    ctrl.set_front([0.0, 1.0, 0.0])
    ctrl.set_up([0, 0, -1])
    ctrl.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    buf_top = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
    cv2.imwrite(str(v12_dir / f'{prefix}_top.png'), cv2.cvtColor(buf_top, cv2.COLOR_RGB2BGR))
    
    # 3. Roof Closeup View
    ctrl.set_lookat(c + np.array([2.0, 0.0, 2.0]))
    ctrl.set_front([0.6, 0.7, 0.5])
    ctrl.set_up([0, -1, 0])
    ctrl.set_zoom(0.4)
    vis.poll_events()
    vis.update_renderer()
    buf_roof = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
    cv2.imwrite(str(v12_dir / f'{prefix}_roof_closeup.png'), cv2.cvtColor(buf_roof, cv2.COLOR_RGB2BGR))
    
    vis.destroy_window()
    print(f'Rendered {prefix} views successfully!')

render_mesh_views(mesh_p, 'v12_poisson')
render_mesh_views(mesh_d, 'v12_delaunay')
