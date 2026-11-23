"""
Renders comparative multi-method reviews for Video 1 and Video 2:
Poisson vs Untrimmed vs TSDF vs BPA vs Point Cloud.
"""

from pathlib import Path
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw

def render_comparison_suite(base_dir: Path, vid_name: str):
    mesh_dir = base_dir / "meshes"
    renders_dir = base_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("Screened Poisson (Postprocessed)", mesh_dir / "postprocessed_mesh_canonical.obj", False),
        ("Screened Poisson (Untrimmed)", mesh_dir / "raw_untrimmed_mesh_canonical.obj", False),
        ("Open3D TSDF Volume", mesh_dir / "tsdf_mesh_canonical.obj", False),
        ("Ball-Pivoting (BPA)", mesh_dir / "bpa_mesh_canonical.obj", False),
        ("Filtered Point Cloud", base_dir / "filtered_point_cloud.ply", True)
    ]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=540)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.94, 0.95, 0.96])
    opt.light_on = True

    rendered_images = []
    for title, path, is_pcd in comparisons:
        if not path.exists():
            continue
        if is_pcd:
            geom = o3d.io.read_point_cloud(str(path))
            opt.point_size = 2.0
        else:
            geom = o3d.io.read_triangle_mesh(str(path))
            geom.compute_vertex_normals()

        vis.add_geometry(geom)
        ctr = vis.get_view_control()
        ctr.set_front([0.4, -0.35, -0.7])
        ctr.set_up([0.0, -1.0, 0.0])
        ctr.set_zoom(0.7)
        vis.poll_events()
        vis.update_renderer()

        out_img = renders_dir / f"compare_{path.stem}.png"
        vis.capture_screen_image(str(out_img), do_render=True)
        rendered_images.append((title, str(out_img)))
        vis.clear_geometries()

    vis.destroy_window()

    # Create 2-column montage
    if rendered_images:
        w, h = 960, 540
        cols = 2
        rows = 3
        montage = Image.new("RGB", (cols * w, rows * h), (240, 240, 240))
        draw = ImageDraw.Draw(montage)

        for i, (title, img_p) in enumerate(rendered_images):
            cx = (i % cols) * w
            cy = (i // cols) * h
            img = Image.open(img_p).resize((w, h))
            montage.paste(img, (cx, cy))
            draw.rectangle([cx, cy, cx + w, cy + 30], fill=(30, 30, 30))
            draw.text((cx + 15, cy + 8), f"{vid_name} — {title}", fill=(255, 255, 255))

        montage_path = renders_dir / "comparative_meshing_review.png"
        montage.save(str(montage_path))
        print(f"[+] Saved comparative review for {vid_name}: {montage_path}")

def main():
    render_comparison_suite(Path("outputs/indoor/video_1_170234"), "Video 1 (Hostel Room)")
    render_comparison_suite(Path("outputs/indoor/video_2_170500"), "Video 2 (Corridor Walkthrough)")

if __name__ == "__main__":
    main()

