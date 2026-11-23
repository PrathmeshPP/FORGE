import open3d as o3d, cv2, numpy as np
from pathlib import Path

v12_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')
pcd_path = v12_dir / 'final_best_canonical_v12_point_cloud.ply'
pcd = o3d.io.read_point_cloud(str(pcd_path))

# Crop extreme distant outliers for clean scene visualization
# Central scene is roughly X in [-20, 30], Y in [-10, 10], Z in [-10, 35]
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=np.array([-25.0, -15.0, -15.0]),
    max_bound=np.array([35.0, 15.0, 40.0])
)
pcd_scene = pcd.crop(bbox)
print(f'Total points: {len(pcd.points)}, Scene points inside bounding box: {len(pcd_scene.points)}')

vis = o3d.visualization.Visualizer()
vis.create_window(visible=False, width=1200, height=900)
vis.add_geometry(pcd_scene)

opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
opt.point_size = 2.5

bb = pcd_scene.get_axis_aligned_bounding_box()
c = bb.get_center()

# 1. Oblique View (matching V10/V11 diagnostics)
ctrl = vis.get_view_control()
ctrl.set_lookat(c)
ctrl.set_front([0.8, 0.8, 0.8])
ctrl.set_up([0, -1, 0])
ctrl.set_zoom(0.7)
vis.poll_events()
vis.update_renderer()
buf_obl = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
cv2.imwrite(str(v12_dir / 'v12_pcd_oblique.png'), cv2.cvtColor(buf_obl, cv2.COLOR_RGB2BGR))

# 2. Top-down View
ctrl.set_front([0.0, 1.0, 0.0])
ctrl.set_up([0, 0, -1])
ctrl.set_zoom(0.7)
vis.poll_events()
vis.update_renderer()
buf_top = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
cv2.imwrite(str(v12_dir / 'v12_pcd_top.png'), cv2.cvtColor(buf_top, cv2.COLOR_RGB2BGR))

# 3. Facade / Side View
ctrl.set_front([0.0, 0.0, 1.0])
ctrl.set_up([0, -1, 0])
ctrl.set_zoom(0.7)
vis.poll_events()
vis.update_renderer()
buf_side = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
cv2.imwrite(str(v12_dir / 'v12_pcd_facade.png'), cv2.cvtColor(buf_side, cv2.COLOR_RGB2BGR))

# 4. Roof & Building Closeup View
ctrl.set_lookat(c + np.array([2.0, 0.0, 3.0]))
ctrl.set_front([0.6, 0.7, 0.5])
ctrl.set_up([0, -1, 0])
ctrl.set_zoom(0.4)
vis.poll_events()
vis.update_renderer()
buf_roof = (np.asarray(vis.capture_screen_float_buffer(do_render=True)) * 255).astype(np.uint8)
cv2.imwrite(str(v12_dir / 'v12_pcd_roof_closeup.png'), cv2.cvtColor(buf_roof, cv2.COLOR_RGB2BGR))

vis.destroy_window()
print('Rendered all 4 point cloud views successfully!')
