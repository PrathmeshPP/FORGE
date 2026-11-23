"""
Builds the Master QA Comparison Panel and Heatmap Panel for Photographic Texture Quality Recovery.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

EXP_DIR = Path("outputs/indoor/photographic_texture_quality_recovery_video1")
RENDERS_DIR = EXP_DIR / "renders"
FRAMES_DIR = Path("outputs/indoor/texture_experiment_video1_reproduced_video_170234/frames")
ARTIFACTS_DIR = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953")

VIEWS = [
    {
        "id": "full_room",
        "title": "VIEW 1: FULL-ROOM OVERVIEW",
        "frame": FRAMES_DIR / "frame_06_f0197_06.61s.png",
        "frame_label": "Source Video Frame 06 (06.61s)"
    },
    {
        "id": "bed",
        "title": "VIEW 2: BED CLOSE-UP (MATTRESS & FABRIC)",
        "frame": FRAMES_DIR / "frame_20_f0446_14.97s.png",
        "frame_label": "Source Video Frame 20 (14.97s)"
    },
    {
        "id": "desk_computer",
        "title": "VIEW 3: DESK & COMPUTER CLOSE-UP (SCREEN & DESK)",
        "frame": FRAMES_DIR / "frame_32_f0655_21.98s.png",
        "frame_label": "Source Video Frame 32 (21.98s)"
    },
    {
        "id": "wall_poster",
        "title": "VIEW 4: WALL & POSTER CLOSE-UP (WALL & SWITCHES)",
        "frame": FRAMES_DIR / "frame_45_f0865_29.03s.png",
        "frame_label": "Source Video Frame 45 (29.03s)"
    },
    {
        "id": "floor",
        "title": "VIEW 5: FLOOR CLOSE-UP (TILES & BASEBOARD)",
        "frame": FRAMES_DIR / "frame_62_f1198_40.20s.png",
        "frame_label": "Source Video Frame 62 (40.20s)"
    }
]

def build_master_comparison_panel():
    cell_w = 460
    cell_h = 270
    pad_x = 24
    pad_y = 30
    header_h = 130
    row_gap = 45

    n_cols = 4
    n_rows = len(VIEWS)

    total_w = pad_x * 2 + n_cols * cell_w + (n_cols - 1) * 20
    total_h = header_h + n_rows * (cell_h + row_gap) + pad_y

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    # Main Header
    draw.text((pad_x, 24), "F.O.R.G.E. - PHOTOGRAPHIC TEXTURE QUALITY RECOVERY & COMPARATIVE QA", fill=(56, 189, 248))
    draw.text((pad_x, 52), "Dataset: WhatsApp Video 2026-10-30 at 17.02.34.mp4 (478x850 Native Resolution) | Three.js WebGL Render Verification", fill=(148, 163, 184))
    draw.text((pad_x, 76), "Direct comparison across 5 required interior viewpoints: Ground Truth Video vs Candidate A vs Candidate B vs Candidate C", fill=(203, 213, 225))

    col_headers = [
        ("SOURCE VIDEO FRAME", "(Ground Truth Optical Photograph)", (203, 213, 225)),
        ("CANDIDATE A: BASELINE", "(Downscaled 512px, Inverted UVs, Dark)", (248, 113, 113)),
        ("CANDIDATE B: WINNER-TAKE-ALL", "(1024px Native Tiles, Sharpest-View, Normals)", (56, 189, 248)),
        ("CANDIDATE C: SEAM-AWARE BLENDED", "(1024px Native Tiles, Harmonized Exposure)", (74, 222, 128))
    ]

    for c_idx, (title, sub, col_color) in enumerate(col_headers):
        x = pad_x + c_idx * (cell_w + 20)
        draw.rectangle([x, 102, x + cell_w, 126], fill=(30, 41, 59))
        draw.text((x + 8, 106), title, fill=col_color)

    for r_idx, v in enumerate(VIEWS):
        y_top = header_h + r_idx * (cell_h + row_gap)

        # Row Title Banner
        draw.rectangle([pad_x, y_top - 20, pad_x + n_cols * cell_w + (n_cols - 1) * 20, y_top - 2], fill=(24, 32, 47))
        draw.text((pad_x + 8, y_top - 18), v["title"], fill=(255, 255, 255))

        # Col 0: Original video frame
        # Native frame is 478x850 (portrait). Center-crop or fit into 460x270
        im_frame = Image.open(v["frame"]).convert("RGB")
        # To display nicely in landscape cell, fit with aspect ratio preserved
        im_frame_ratio = im_frame.width / im_frame.height
        target_h = cell_h
        target_w = int(target_h * im_frame_ratio)
        frame_resized = im_frame.resize((target_w, target_h), Image.Resampling.LANCZOS)
        frame_cell = Image.new("RGB", (cell_w, cell_h), color=(10, 15, 26))
        off_x = (cell_w - target_w) // 2
        frame_cell.paste(frame_resized, (off_x, 0))
        # Draw frame tag
        d_fc = ImageDraw.Draw(frame_cell)
        d_fc.rectangle([4, 4, 260, 24], fill=(0, 0, 0))
        d_fc.text((8, 6), v["frame_label"], fill=(255, 255, 255))

        x0 = pad_x
        canvas.paste(frame_cell, (x0, y_top))
        draw.rectangle([x0, y_top, x0 + cell_w, y_top + cell_h], outline=(71, 85, 105), width=1)

        # Col 1: Candidate A
        path_a = RENDERS_DIR / f"render_candidate_a_{v['id']}.png"
        if path_a.exists():
            im_a = Image.open(path_a).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x1 = pad_x + 1 * (cell_w + 20)
            canvas.paste(im_a, (x1, y_top))
            draw.rectangle([x1, y_top, x1 + cell_w, y_top + cell_h], outline=(239, 68, 68), width=2)

        # Col 2: Candidate B
        path_b = RENDERS_DIR / f"render_candidate_b_{v['id']}.png"
        if path_b.exists():
            im_b = Image.open(path_b).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x2 = pad_x + 2 * (cell_w + 20)
            canvas.paste(im_b, (x2, y_top))
            draw.rectangle([x2, y_top, x2 + cell_w, y_top + cell_h], outline=(59, 130, 246), width=2)

        # Col 3: Candidate C
        path_c = RENDERS_DIR / f"render_candidate_c_{v['id']}.png"
        if path_c.exists():
            im_c = Image.open(path_c).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x3 = pad_x + 3 * (cell_w + 20)
            canvas.paste(im_c, (x3, y_top))
            draw.rectangle([x3, y_top, x3 + cell_w, y_top + cell_h], outline=(34, 197, 94), width=2)

    out_panel_path = RENDERS_DIR / "master_texture_quality_comparison_panel.png"
    canvas.save(out_panel_path, format="PNG")
    print(f"[+] Saved Master Comparison Panel: {out_panel_path}")

    # Copy to artifacts directory
    dst_artifact = ARTIFACTS_DIR / "master_texture_quality_comparison_panel.png"
    canvas.save(dst_artifact, format="PNG")
    print(f"[+] Copied to artifacts directory: {dst_artifact}")


def build_heatmap_panel():
    cell_w = 460
    cell_h = 270
    pad_x = 24
    pad_y = 30
    header_h = 140

    views = VIEWS
    n_views = len(views)
    total_w = pad_x * 2 + 3 * cell_w + 2 * 20
    total_h = header_h + 2 * (cell_h + 40) + pad_y

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad_x, 24), "F.O.R.G.E. - 4-CATEGORY PHOTOGRAPHIC COVERAGE HEATMAP", fill=(245, 158, 11))
    draw.text((pad_x, 52), "Classification: Green = Directly Photographed (41.33%) | Blue = Blended Transition (22.57%) | Total Photographic = 63.90%", fill=(226, 232, 240))
    draw.text((pad_x, 76), "Yellow = Occluded by Neural Depth (2.10%) | Red = Never-Observed / Outside View Frustum (34.00%)", fill=(148, 163, 184))

    # Legend bar
    legend_items = [
        ("Directly Photographed (41.33%)", (46, 204, 113)),
        ("Blended Transition Region (22.57%)", (52, 152, 219)),
        ("Depth Occluded (2.10%)", (241, 196, 15)),
        ("Never Observed (34.00%)", (231, 76, 60))
    ]
    lx = pad_x
    for label, col in legend_items:
        draw.rectangle([lx, 105, lx + 20, 123], fill=col)
        draw.text((lx + 26, 106), label, fill=(255, 255, 255))
        lx += 340

    coords = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1)
    ]
    for idx, v in enumerate(views):
        c, r = coords[idx]
        x = pad_x + c * (cell_w + 20)
        y = header_h + r * (cell_h + 40)

        path_heat = RENDERS_DIR / f"render_heatmap_{v['id']}.png"
        if path_heat.exists():
            im = Image.open(path_heat).convert("RGB").resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(245, 158, 11), width=2)
            draw.rectangle([x, y - 22, x + cell_w, y], fill=(30, 41, 59))
            draw.text((x + 8, y - 18), v["title"], fill=(255, 255, 255))

    out_heat_path = RENDERS_DIR / "coverage_heatmap_panel.png"
    canvas.save(out_heat_path, format="PNG")
    print(f"[+] Saved Heatmap Panel: {out_heat_path}")

    dst_artifact = ARTIFACTS_DIR / "coverage_heatmap_panel.png"
    canvas.save(dst_artifact, format="PNG")
    print(f"[+] Copied to artifacts directory: {dst_artifact}")


if __name__ == "__main__":
    build_master_comparison_panel()
    build_heatmap_panel()

