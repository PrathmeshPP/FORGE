"""
Builds the Master 10-View Geometry Quality Comparison Panel:
  Candidate B (Prev. Unbounded Sim3) vs Candidate D (Rigid SE3 Poisson HQ) vs Candidate E (Rigid SE3 TSDF)
and copies to artifacts directory.
"""

from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

EXP_DIR = Path("outputs/indoor/geometry_quality_improvement_video1")
RENDERS_DIR = EXP_DIR / "renders"
FRAMES_DIR = Path("outputs/indoor/texture_experiment_video1_reproduced_video_170234/frames")
ARTIFACTS_DIR = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953")

VIEWS = [
    ("v01_complete_room", "VIEW 1: Complete Room Overview"),
    ("v02_top_down_layout", "VIEW 2: Top-Down Room Layout"),
    ("v03_front_elevation", "VIEW 3: Front Elevation"),
    ("v04_side_elevation", "VIEW 4: Side Elevation"),
    ("v05_isometric", "VIEW 5: Isometric (3/4 Angle)"),
    ("v06_bed_closeup", "VIEW 6: Bed Close-Up"),
    ("v07_desk_computer", "VIEW 7: Desk & Computer"),
    ("v08_wardrobe", "VIEW 8: Wardrobe"),
    ("v09_wall_window", "VIEW 9: Wall & Window"),
    ("v10_floor_ceiling", "VIEW 10: Floor & Ceiling"),
]

CANDIDATES = [
    ("candidate_b", "B: Unbounded Sim(3)", (248, 113, 113)),
    ("candidate_d", "D: Rigid SE(3) Poisson HQ", (56, 189, 248)),
    ("candidate_e", "E: Rigid SE(3) TSDF 1.5cm", (74, 222, 128)),
]

def build_master_panel():
    cell_w = 420
    cell_h = 236
    pad_x = 24
    pad_y = 30
    header_h = 100
    row_label_h = 22
    row_gap = 32

    n_cols = len(CANDIDATES)
    n_rows = len(VIEWS)

    total_w = pad_x * 2 + n_cols * cell_w + (n_cols - 1) * 12
    total_h = header_h + n_rows * (cell_h + row_label_h + row_gap) + pad_y

    canvas = Image.new("RGB", (total_w, total_h), color=(11, 15, 25))
    draw = ImageDraw.Draw(canvas)

    # Main Header
    draw.text((pad_x, 18), "F.O.R.G.E. GEOMETRY QUALITY IMPROVEMENT - 10-VIEW COMPARISON PANEL", fill=(56, 189, 248))
    draw.text((pad_x, 42), "Hostel Room B 314 | WhatsApp Video 2026-10-30 at 17.02.34.mp4 | Rigid SE(3) Scale Lock vs Unbounded Sim(3)", fill=(148, 163, 184))

    # Column Headers
    for c_idx, (cand_id, label, col_color) in enumerate(CANDIDATES):
        x = pad_x + c_idx * (cell_w + 12)
        draw.rectangle([x, 70, x + cell_w, 92], fill=(24, 32, 47))
        draw.text((x + 8, 74), label, fill=col_color)

    for r_idx, (view_id, view_title) in enumerate(VIEWS):
        y_row = header_h + r_idx * (cell_h + row_label_h + row_gap)

        # Row title
        draw.rectangle([pad_x, y_row, pad_x + n_cols * cell_w + (n_cols - 1) * 12, y_row + row_label_h], fill=(20, 27, 40))
        draw.text((pad_x + 8, y_row + 3), view_title, fill=(255, 255, 255))

        for c_idx, (cand_id, label, col_color) in enumerate(CANDIDATES):
            x = pad_x + c_idx * (cell_w + 12)
            y = y_row + row_label_h + 4

            render_path = RENDERS_DIR / f"render_{cand_id}_{view_id}.png"
            if render_path.exists():
                im = Image.open(render_path).convert("RGB")
                im_resized = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas.paste(im_resized, (x, y))
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(30, 41, 59), outline=(55, 65, 81))
                draw.text((x + 10, y + cell_h // 2 - 8), f"Render Missing: {cand_id}/{view_id}", fill=(148, 163, 184))

    # Save
    out_path = EXP_DIR / "master_geometry_quality_comparison_panel.png"
    canvas.save(out_path, quality=95)
    print(f"[+] Saved master panel: {out_path}")

    art_path = ARTIFACTS_DIR / "master_geometry_quality_comparison_panel.png"
    canvas.save(art_path, quality=95)
    print(f"[+] Exported artifact: {art_path}")


def build_proportions_summary_panel():
    """Creates a visual proportions comparison chart showing improvement."""
    import json
    audit_path = EXP_DIR / "diagnostics" / "geometric_proportion_audit.json"
    with open(audit_path) as f:
        audit = json.load(f)

    w, h = 900, 520
    canvas = Image.new("RGB", (w, h), color=(11, 15, 25))
    draw = ImageDraw.Draw(canvas)

    draw.text((30, 16), "F.O.R.G.E. GEOMETRIC PROPORTIONS RECOVERY AUDIT", fill=(56, 189, 248))
    draw.text((30, 38), "Hostel Room B 314 | Rigid SE(3) Scale Lock vs Unbounded Sim(3)", fill=(148, 163, 184))

    # Table headers
    col_x = [30, 220, 440, 650]
    row_y = 80
    headers = ["Measurement", "Candidate B (Prev.)", "Candidate D (New)", "Recovery"]
    for i, hdr in enumerate(headers):
        draw.text((col_x[i], row_y), hdr, fill=(203, 213, 225))
    draw.line([30, row_y + 18, w - 30, row_y + 18], fill=(51, 65, 85))

    b = audit["candidate_b"]
    d = audit["candidate_d"]

    rows = [
        ("Room Height", f'{b["room_height_m"]:.2f} m', f'{d["room_height_m"]:.2f} m', f'{d["room_height_m"] / b["room_height_m"]:.1f}x taller'),
        ("Room Width (X)", f'{b["room_width_m"]:.2f} m', f'{d["room_width_m"]:.2f} m', f'{d["room_width_m"] / b["room_width_m"]:.1f}x wider'),
        ("Room Depth (Z)", f'{b["room_depth_m"]:.2f} m', f'{d["room_depth_m"]:.2f} m', f'{d["room_depth_m"] / b["room_depth_m"]:.1f}x deeper'),
        ("Bed Height", f'{b["bed_height_m"]:.2f} m ({b["bed_ratio_pct"]:.1f}% room)', f'{d["bed_height_m"]:.2f} m ({d["bed_ratio_pct"]:.1f}% room)', 'FIXED' if d["bed_ratio_pct"] < 32 else 'PARTIAL'),
        ("Desk Height", f'{b["desk_height_m"]:.2f} m ({b["desk_ratio_pct"]:.1f}% room)', f'{d["desk_height_m"]:.2f} m ({d["desk_ratio_pct"]:.1f}% room)', 'FIXED' if d["desk_ratio_pct"] < 42 else 'PARTIAL'),
        ("Chair Detected", "Missing (filtered out)", f'{d["chair_points_count"]:,} mesh verts', 'RECOVERED'),
        ("Scale Drift", "s=0.51 to 2.17 (4.25x)", "s=1.0000 (locked)", 'ELIMINATED'),
    ]

    for ri, (label, val_b, val_d, status) in enumerate(rows):
        y = row_y + 28 + ri * 32
        draw.text((col_x[0], y), label, fill=(226, 232, 240))
        draw.text((col_x[1], y), val_b, fill=(248, 113, 113))
        draw.text((col_x[2], y), val_d, fill=(56, 189, 248))
        sc = (74, 222, 128) if 'FIXED' in status or 'ELIMINATED' in status or 'RECOVERED' in status else (250, 204, 21)
        draw.text((col_x[3], y), status, fill=sc)

    # Scale Drift Visualization
    bar_y = row_y + 28 + len(rows) * 32 + 40
    draw.text((30, bar_y), "CUMULATIVE SCALE DRIFT COMPARISON", fill=(203, 213, 225))

    # Candidate B scale bars
    b_scales = [1.0, 0.51, 2.11, 2.08, 2.17, 2.12, 1.95, 1.87, 1.00, 1.00, 2.07, 2.15]
    d_scales = [1.0] * len(b_scales)

    bar_w_max = 350
    bar_h = 12
    for i, (sb, sd) in enumerate(zip(b_scales[:8], d_scales[:8])):
        y = bar_y + 24 + i * 20
        draw.text((30, y), f"W{i:02d}", fill=(148, 163, 184))
        # B bar (red)
        bw = int(sb / 2.2 * bar_w_max * 0.5)
        draw.rectangle([80, y, 80 + bw, y + bar_h], fill=(248, 113, 113))
        # D bar (blue)
        dw = int(sd / 2.2 * bar_w_max * 0.5)
        draw.rectangle([80 + bar_w_max + 40, y, 80 + bar_w_max + 40 + dw, y + bar_h], fill=(56, 189, 248))

    draw.text((80, bar_y + 24 + 8 * 20 + 5), "Candidate B: Chaotic Scale Drift", fill=(248, 113, 113))
    draw.text((80 + bar_w_max + 40, bar_y + 24 + 8 * 20 + 5), "Candidate D: Perfect Scale Lock", fill=(56, 189, 248))

    out_path = EXP_DIR / "proportions_recovery_panel.png"
    canvas.save(out_path)
    art_path = ARTIFACTS_DIR / "proportions_recovery_panel.png"
    canvas.save(art_path)
    print(f"[+] Saved proportions panel: {out_path}")
    print(f"[+] Exported artifact: {art_path}")


if __name__ == "__main__":
    build_proportions_summary_panel()
    build_master_panel()

