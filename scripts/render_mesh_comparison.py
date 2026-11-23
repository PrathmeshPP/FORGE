"""
Renders comparative side-by-side diagnostic previews of all reconstruction methods:
1. Screened Poisson (Postprocessed)
2. Screened Poisson (Untrimmed)
3. TSDF Volume Integration
4. Ball-Pivoting Algorithm (BPA)
5. Filtered Dense Point Cloud
"""

import sys
from pathlib import Path
import open3d as o3d
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("outputs/indoor/room_001/attempt_4_new_video")
MESH_DIR = OUTPUT_DIR / "meshes"
RENDERS_DIR = OUTPUT_DIR / "renders"

def render_mesh_from_view(geom, save_path: Path, is_pcd: bool = False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=960, height=540)
    vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.94, 0.95, 0.96])
    opt.light_on = True
    if is_pcd:
        opt.point_size = 2.5

    ctr = vis.get_view_control()
    ctr.set_front([0.5, -0.35, -0.7])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(save_path), do_render=True)
    vis.destroy_window()

def main():
    RENDERS_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = [
        ("Screened Poisson (Postprocessed)", MESH_DIR / "postprocessed_mesh_canonical.obj", False),
        ("Screened Poisson (Untrimmed)", MESH_DIR / "raw_untrimmed_mesh_canonical.obj", False),
        ("Open3D TSDF Volume", MESH_DIR / "tsdf_mesh_canonical.obj", False),
        ("Ball-Pivoting (BPA)", MESH_DIR / "bpa_mesh_canonical.obj", False),
        ("Filtered Point Cloud", OUTPUT_DIR / "filtered_point_cloud.ply", True)
    ]

    rendered_images = []
    for title, path, is_pcd in comparisons:
        if not path.exists():
            print(f"Skipping {title}, file not found: {path}")
            continue
        print(f"Loading {title} from {path.name}...")
        if is_pcd:
            geom = o3d.io.read_point_cloud(str(path))
        else:
            geom = o3d.io.read_triangle_mesh(str(path))
            geom.compute_vertex_normals()

        save_name = f"comparison_{path.stem}.png"
        save_path = RENDERS_DIR / save_name
        render_mesh_from_view(geom, save_path, is_pcd=is_pcd)
        print(f"  Saved render: {save_path.name}")
        rendered_images.append((title, str(save_path)))

    # Assemble a comparative contact sheet
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

            # Banner
            draw.rectangle([cx, cy, cx + w, cy + 30], fill=(30, 30, 30))
            draw.text((cx + 15, cy + 8), title, fill=(255, 255, 255))

        montage_path = RENDERS_DIR / "comparative_meshing_review.png"
        montage.save(str(montage_path))
        print(f"[+] Comparative meshing review saved to: {montage_path}")

if __name__ == "__main__":
    main()

