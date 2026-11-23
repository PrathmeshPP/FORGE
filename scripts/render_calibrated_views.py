r"""
Render calibrated, stunning perspective views of the reconstructed 3D campus model.
Uses verified drone camera calibrations and overhead nadir vectors.
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
MESH_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/mesh"
REPORTS_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/reports"
CAMERA_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/camera"

def render_calibrated_views():
    mesh_path = MESH_DIR / "surface_mesh_photographic.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()

    poses = np.load(CAMERA_DIR / "camera_poses.npz")
    exts = poses["extrinsics"]
    intrs = poses["intrinsics"]

    W, H = 1280, 720
    views = {}

    def render_pinhole(ext_idx, zoom_factor=1.0, lookat_offset=None, bg=[0.06, 0.08, 0.12], clay=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=W, height=H)
        
        if clay:
            m_render = o3d.geometry.TriangleMesh(mesh)
            m_render.compute_vertex_normals()
            gray_cols = np.full((len(m_render.vertices), 3), 0.72)
            m_render.vertex_colors = o3d.utility.Vector3dVector(gray_cols)
            vis.add_geometry(m_render)
            opt = vis.get_render_option()
            opt.background_color = np.asarray(bg)
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
            opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
            opt.light_on = True
        else:
            vis.add_geometry(mesh)
            opt = vis.get_render_option()
            opt.background_color = np.asarray(bg)
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
            opt.light_on = False

        ctrl = vis.get_view_control()
        pinhole = ctrl.convert_to_pinhole_camera_parameters()

        ext_4x4 = np.eye(4)
        ext_4x4[:3, :] = exts[ext_idx]
        pinhole.extrinsic = ext_4x4

        K = intrs[ext_idx].copy()
        K[0, :] *= (W / 688.0) * zoom_factor
        K[1, :] *= (H / 384.0) * zoom_factor
        pinhole.intrinsic.set_intrinsics(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        ctrl.convert_from_pinhole_camera_parameters(pinhole, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        buf = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        return Image.fromarray((np.asarray(buf) * 255).astype(np.uint8))

    # 1. Oblique Drone Perspective (Keyframe 3 - Direct View over Campus Solar Complex)
    print("Rendering Oblique Perspective...")
    img_oblique = render_pinhole(ext_idx=3, zoom_factor=0.95)
    p_oblique = MESH_DIR / "render_view_oblique.png"
    img_oblique.save(p_oblique)
    views["oblique"] = p_oblique

    # 2. Frontal Facade Perspective (Keyframe 0 - Approaching Solar Roof & Entrance)
    print("Rendering Frontal Facade Perspective...")
    img_frontal = render_pinhole(ext_idx=0, zoom_factor=0.95)
    p_frontal = MESH_DIR / "render_view_frontal.png"
    img_frontal.save(p_frontal)
    views["frontal"] = p_frontal

    # 3. Close-up Rooftop Solar Inspection (Keyframe 5 - Magnified Center)
    print("Rendering Close-up Solar Rooftop Inspection...")
    img_closeup = render_pinhole(ext_idx=5, zoom_factor=1.45)
    p_closeup = MESH_DIR / "render_view_closeup.png"
    img_closeup.save(p_closeup)
    views["closeup"] = p_closeup

    # 4. Neutral Clay Shading Geometry (Keyframe 3)
    print("Rendering Neutral Clay Geometry...")
    img_clay = render_pinhole(ext_idx=3, zoom_factor=0.95, clay=True)
    p_clay = MESH_DIR / "neutral_geometry_preview.png"
    img_clay.save(p_clay)
    views["neutral_clay"] = p_clay

    # 5. Top Nadir Overview (Looking straight down at building and courtyard)
    print("Rendering Top Nadir Overview...")
    vis_top = o3d.visualization.Visualizer()
    vis_top.create_window(visible=False, width=W, height=H)
    vis_top.add_geometry(mesh)
    opt = vis_top.get_render_option()
    opt.background_color = np.asarray([0.06, 0.08, 0.12])
    opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    opt.light_on = False

    ref_bbox = mesh.get_axis_aligned_bounding_box()
    ctr = ref_bbox.get_center()

    ctrl = vis_top.get_view_control()
    ctrl.set_lookat([ctr[0], ctr[1], ctr[2] * 0.9])
    ctrl.set_front([0.0, 0.98, 0.20])
    ctrl.set_up([0.0, -0.20, 0.98])
    ctrl.set_zoom(0.65)

    vis_top.poll_events()
    vis_top.update_renderer()
    buf_top = vis_top.capture_screen_float_buffer(do_render=True)
    vis_top.destroy_window()

    img_top = Image.fromarray((np.asarray(buf_top) * 255).astype(np.uint8))
    p_top = MESH_DIR / "render_view_top.png"
    img_top.save(p_top)
    views["top"] = p_top

    # Master 4-Panel Review Sheet
    print("Generating 4-panel visual review sheet...")
    W_p, H_p = 640, 480
    header_h = 75
    canvas = Image.new("RGB", (W_p * 4, header_h + H_p), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)

    draw.text((25, 14), "F.O.R.G.E 3D RECONSTRUCTION — CAMPUS SOLAR COMPLEX", fill=(255, 255, 255))
    draw.text((25, 42), "Verified VGGT-Omega Foundation Model & OpenCV 5.0 Curation | Relative Scale (Metric Unverified)", fill=(170, 200, 230))

    canvas.paste(img_top.resize((W_p, H_p), Image.Resampling.LANCZOS), (0, header_h))
    canvas.paste(img_oblique.resize((W_p, H_p), Image.Resampling.LANCZOS), (W_p, header_h))
    canvas.paste(img_clay.resize((W_p, H_p), Image.Resampling.LANCZOS), (W_p * 2, header_h))

    atlas_path = MESH_DIR / "texture_atlas.png"
    if atlas_path.is_file():
        im_atlas = Image.open(atlas_path).resize((W_p, H_p), Image.Resampling.LANCZOS)
        canvas.paste(im_atlas, (W_p * 3, header_h))

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
    print(f"Saved master review sheet to: {review_sheet_path.name}")


if __name__ == "__main__":
    render_calibrated_views()
