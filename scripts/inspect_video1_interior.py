"""
Inspects Video 1 interior geometry:
Renders interior camera views, point cloud, TSDF mesh, and cutaway Poisson mesh.
"""

from pathlib import Path
import open3d as o3d
import numpy as np
import json

OUT_DIR = Path("outputs/indoor/video_1_170234")
MESH_DIR = OUT_DIR / "meshes"
RENDERS_DIR = OUT_DIR / "renders"

def main():
    pcd_path = OUT_DIR / "filtered_point_cloud.ply"
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    pts = np.asarray(pcd.points)
    print(f"Point cloud: {len(pts):,} points, bounds min: {pts.min(axis=0)}, max: {pts.max(axis=0)}")

    # Offscreen visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1280, height=720)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.94, 0.95, 0.96])
    opt.light_on = True

    # 1. Point Cloud Interior View (from door looking in towards beds/cooler)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_front([0.2, -0.2, -0.9])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.65)
    vis.poll_events()
    vis.update_renderer()
    pcd_render = RENDERS_DIR / "video_1_pcd_interior.png"
    vis.capture_screen_image(str(pcd_render), do_render=True)
    print(f"Saved: {pcd_render.name}")
    vis.clear_geometries()

    # 2. TSDF Mesh
    tsdf_path = MESH_DIR / "tsdf_mesh_canonical.obj"
    if tsdf_path.exists():
        tsdf_mesh = o3d.io.read_triangle_mesh(str(tsdf_path))
        tsdf_mesh.compute_vertex_normals()
        vis.add_geometry(tsdf_mesh)
        ctr.set_front([0.3, -0.3, -0.8])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.65)
        vis.poll_events()
        vis.update_renderer()
        tsdf_render = RENDERS_DIR / "video_1_tsdf_interior.png"
        vis.capture_screen_image(str(tsdf_render), do_render=True)
        print(f"Saved: {tsdf_render.name}")
        vis.clear_geometries()

    # 3. Poisson Mesh with Ceiling Cutaway (so interior beds, desks, cooler are visible from above)
    poisson_path = MESH_DIR / "postprocessed_mesh_canonical.obj"
    if poisson_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(poisson_path))
        mesh.compute_vertex_normals()
        v = np.asarray(mesh.vertices)
        # Cut top 20% (ceiling) to see into the room
        y_cut = np.percentile(v[:, 1], 80)
        # In canonical Three.js coordinates, let's see which axis is up
        # We can crop the ceiling vertices
        tri = np.asarray(mesh.triangles)
        v_mask = v[:, 1] > y_cut  # top vertices
        # remove triangles with all 3 vertices in ceiling
        tri_mask = (v_mask[tri[:, 0]] & v_mask[tri[:, 1]] & v_mask[tri[:, 2]])
        mesh_cut = o3d.geometry.TriangleMesh(mesh)
        mesh_cut.remove_triangles_by_mask(tri_mask)
        mesh_cut.remove_unreferenced_vertices()

        vis.add_geometry(mesh_cut)
        ctr.set_front([0.4, -0.5, -0.7])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.65)
        vis.poll_events()
        vis.update_renderer()
        cut_render = RENDERS_DIR / "video_1_cutaway_overview.png"
        vis.capture_screen_image(str(cut_render), do_render=True)
        print(f"Saved: {cut_render.name}")
        vis.clear_geometries()

    vis.destroy_window()

if __name__ == "__main__":
    main()

