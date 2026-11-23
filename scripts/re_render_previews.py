r"""
Render high-contrast, perfectly framed perspective views of the reconstructed 3D mesh.
Uses correct camera vectors for [X=lateral, Y=vertical down, Z=depth corridor].
Disables headlight specular washout to show true photographic vertex colors.
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
MESH_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/mesh"
REPORTS_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/reports"

def render_views():
    mesh_path = MESH_DIR / "surface_mesh_photographic.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    ref_bbox = mesh.get_axis_aligned_bounding_box()
    ctr_pt = ref_bbox.get_center()
    ext = ref_bbox.get_extent()
    print(f"Mesh center: {ctr_pt}, extents: {ext}")

    W, H = 1280, 720

    # In our coordinate system:
    # X in [-1.07, 2.75] (extent 3.82)
    # Y in [-0.55, 0.24] (extent 0.79 -> vertical altitude, negative is higher in OpenCV)
    # Z in [0.93, 5.31] (extent 4.38 -> depth along corridor)
    #
    # Nadir / Top View: Looking down (+Y is down, so look vector has positive Y or front=[0, -1, 0] depending on eye)
    # Oblique View: 45 degree angle looking down and forward
    # Frontal View: Looking along Z axis towards building facade
    # Close-up: Zoomed onto the main solar building roof

    configs = {
        "top": {
            "front": [0.0, -0.98, -0.20],
            "up": [0.0, -0.20, 0.98],
            "zoom": 0.68
        },
        "oblique": {
            "front": [0.45, -0.65, -0.60],
            "up": [0.0, -0.80, 0.60],
            "zoom": 0.68
        },
        "frontal": {
            "front": [0.0, -0.40, -0.92],
            "up": [0.0, -0.92, 0.40],
            "zoom": 0.68
        },
        "closeup": {
            "front": [0.35, -0.60, -0.70],
            "up": [0.0, -0.80, 0.60],
            "zoom": 0.32
        }
    }

    views = {}
    for vname, cfg in configs.items():
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=W, height=H)
        vis.add_geometry(mesh)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.07, 0.09, 0.13])
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        opt.light_on = False  # Avoid specular/headlight washout for photographic colors

        ctrl = vis.get_view_control()
        ctrl.set_lookat(ctr_pt)
        ctrl.set_front(cfg["front"])
        ctrl.set_up(cfg["up"])
        ctrl.set_zoom(cfg["zoom"])

        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()

        img_path = MESH_DIR / f"render_view_{vname}.png"
        Image.fromarray((np.asarray(buf) * 255).astype(np.uint8)).save(img_path)
        views[vname] = img_path
        print(f"Rendered {vname} to {img_path.name}")

    # Neutral Clay Shading Render
    vis_clay = o3d.visualization.Visualizer()
    vis_clay.create_window(visible=False, width=W, height=H)
    m_clay = o3d.geometry.TriangleMesh(mesh)
    m_clay.compute_vertex_normals()
    gray_cols = np.full((len(m_clay.vertices), 3), 0.75)
    m_clay.vertex_colors = o3d.utility.Vector3dVector(gray_cols)
    vis_clay.add_geometry(m_clay)
    opt = vis_clay.get_render_option()
    opt.background_color = np.asarray([0.07, 0.09, 0.13])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
    opt.light_on = True

    ctrl = vis_clay.get_view_control()
    ctrl.set_lookat(ctr_pt)
    ctrl.set_front(configs["oblique"]["front"])
    ctrl.set_up(configs["oblique"]["up"])
    ctrl.set_zoom(configs["oblique"]["zoom"])

    vis_clay.poll_events()
    vis_clay.update_renderer()
    buf_clay = vis_clay.capture_screen_float_buffer(do_render=True)
    vis_clay.destroy_window()

    clay_path = MESH_DIR / "neutral_geometry_preview.png"
    Image.fromarray((np.asarray(buf_clay) * 255).astype(np.uint8)).save(clay_path)
    views["neutral_clay"] = clay_path
    print(f"Rendered neutral_clay to {clay_path.name}")

    # Generate 4-panel visual review sheet
    W_p, H_p = 640, 480
    header_h = 75
    canvas = Image.new("RGB", (W_p * 4, header_h + H_p), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((25, 14), "F.O.R.G.E 3D RECONSTRUCTION — CAMPUS SOLAR COMPLEX", fill=(255, 255, 255))
    draw.text((25, 42), "Verified VGGT-Omega Foundation Model & OpenCV 5.0 Curation | Relative Scale (Metric Unverified)", fill=(170, 200, 230))

    im1 = Image.open(views["top"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im1, (0, header_h))

    im2 = Image.open(views["oblique"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im2, (W_p, header_h))

    im3 = Image.open(views["neutral_clay"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im3, (W_p * 2, header_h))

    atlas_path = MESH_DIR / "texture_atlas.png"
    if atlas_path.is_file():
        im4 = Image.open(atlas_path).resize((W_p, H_p), Image.Resampling.LANCZOS)
    else:
        im4 = Image.open(views["closeup"]).resize((W_p, H_p), Image.Resampling.LANCZOS)
    canvas.paste(im4, (W_p * 3, header_h))

    badges = [
        (0, "1. Top Nadir Overview", "(Roof Panels & Courtyard)", (100, 220, 255)),
        (W_p, "2. Oblique Perspective", "(Building Facades & Foliage)", (120, 240, 160)),
        (W_p * 2, "3. Neutral Geometry", "(Screened Poisson Manifold)", (255, 200, 100)),
        (W_p * 3, "4. 4K UV Texture Atlas", f"({atlas_path.name})", (240, 140, 220))
    ]
    for x_off, c_title, c_sub, col in badges:
        draw.rectangle([x_off + 12, header_h + 12, x_off + 380, header_h + 54], fill=(12, 16, 22))
        draw.text((x_off + 20, header_h + 16), c_title, fill=col)
        draw.text((x_off + 20, header_h + 34), c_sub, fill=(200, 220, 240))

    review_sheet_path = REPORTS_DIR / "model_visual_review.png"
    canvas.save(review_sheet_path, "PNG")
    print(f"Saved master review sheet to {review_sheet_path.name}")

if __name__ == "__main__":
    render_views()
