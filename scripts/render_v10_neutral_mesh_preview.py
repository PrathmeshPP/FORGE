#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Render Neutral Geometry Preview on Pure White (#FFFFFF) Background
"""

import sys
import os
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import pycolmap

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
PILOT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm"
MESH_PATH = PILOT_DIR / "mesh_test/candidate_a_screened_poisson_mesh.ply"
OUT_IMG = PILOT_DIR / "mesh_test/neutral_geometry_preview.png"

def main():
    mesh = o3d.io.read_triangle_mesh(str(MESH_PATH))
    mesh.compute_vertex_normals()
    print(f"Loaded mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # Bounding box and center
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    max_dim = np.max(extent)
    print(f"Center: {center}, Max Dim: {max_dim:.2f}m")

    # Render 4 views from registered drone camera perspectives
    rec = pycolmap.Reconstruction(str(PILOT_DIR / "sparse/0"))
    images = sorted(list(rec.images.values()), key=lambda x: x.name)

    # 4 distinct viewpoints across the flyover trajectory: entry, solar building, courtyard, departure
    view_indices = [2, 8, 14, 20]
    view_titles = [
        f"Approach View (t={32.0 + 2*1.565:.1f}s)",
        f"Solar Building Facade & Roof (t={32.0 + 8*1.565:.1f}s)",
        f"Courtyard & Road Pavement (t={32.0 + 14*1.565:.1f}s)",
        f"Oblique Flyover Margin (t={32.0 + 20*1.565:.1f}s)"
    ]

    W, H = 1000, 750
    rendered_panels = []

    for v_idx, cam_idx in enumerate(view_indices):
        im_info = images[cam_idx]
        cfw = im_info.cam_from_world()
        R = cfw.rotation.matrix()
        t = cfw.translation
        cam_center = -R.T @ t
        forward = R.T @ np.array([0.0, 0.0, 1.0])
        up = R.T @ np.array([0.0, -1.0, 0.0])

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=W, height=H)
        
        # Neutral clay shading: assign uniform soft grey-blue color and compute normals
        m_clay = o3d.geometry.TriangleMesh(mesh)
        m_clay.paint_uniform_color([0.76, 0.80, 0.85])
        m_clay.compute_vertex_normals()
        vis.add_geometry(m_clay)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1.0, 1.0, 1.0]) # Pure white (#FFFFFF)
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        opt.light_on = True

        ctrl = vis.get_view_control()
        # Look from camera position towards center
        ctrl.set_lookat(cam_center + forward * 8.0)
        ctrl.set_front(-forward)
        ctrl.set_up(up)
        ctrl.set_zoom(0.68)

        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        img = (np.asarray(buf) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw panel title
        title = view_titles[v_idx]
        cv2.putText(img_bgr, title, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img_bgr, f"PyCOLMAP Registered Camera {cam_idx:02d} ({im_info.name})", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (90, 90, 90), 1, cv2.LINE_AA)
        rendered_panels.append(img_bgr)

    # Assemble 2x2 grid (2000 x 1500)
    top_row = np.hstack([rendered_panels[0], rendered_panels[1]])
    bot_row = np.hstack([rendered_panels[2], rendered_panels[3]])
    grid = np.vstack([top_row, bot_row])

    # Header banner (120 px)
    header = np.ones((120, 2000, 3), dtype=np.uint8) * 255
    cv2.putText(header, "F.O.R.G.E V10 Candidate A — Neutral Surface Geometry Reconstruction", (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(header, f"Screened Poisson Surface Reconstruction | Edge-Manifold: TRUE | Vertices: {len(mesh.vertices):,} | Triangles: {len(mesh.triangles):,}",
                (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (80, 80, 80), 1, cv2.LINE_AA)
    cv2.line(header, (0, 119), (2000, 119), (220, 220, 220), 1)

    final_image = np.vstack([header, grid])
    cv2.imwrite(str(OUT_IMG), final_image)
    print(f"Saved neutral geometry preview: {OUT_IMG.name}")

if __name__ == "__main__":
    main()
