"""
Builds the Master Comparison Panel for Candidates F & G
"""

from pathlib import Path
from PIL import Image, ImageDraw

EXP_DIR = Path("outputs/indoor/critical_reconstruction_recovery_video1")
RENDERS_DIR = EXP_DIR / "renders"
ARTIFACTS_DIR = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953")

VIEWS = [
    ("v01_room_overview", "VIEW 1: Complete Room Overview"),
    ("v02_top_down_layout", "VIEW 2: Top-Down Room Layout"),
    ("v03_room_entrance", "VIEW 3: Room Entrance (Front)"),
    ("v04_rear_window_view", "VIEW 4: Rear Window (Back)"),
    ("v05_isometric_enclosure", "VIEW 5: Isometric Enclosure (3/4)"),
    ("v06_bed_closeup", "VIEW 6: Bed Close-Up"),
    ("v07_desk_chair_closeup", "VIEW 7: Desk & Chair Close-Up"),
    ("v08_wardrobe_closeup", "VIEW 8: Wardrobe Close-Up"),
    ("v09_left_wall_window", "VIEW 9: Left Wall & Window"),
    ("v10_floor_ceiling", "VIEW 10: Floor & Ceiling Inspection")
]

MODELS = [
    ("candidate_f", "CANDIDATE F: Continuous Poisson HQ (Topology Filtered)", (56, 189, 248)),
    ("candidate_g", "CANDIDATE G: Scalable TSDF Volumetric (1.5cm Voxel)", (74, 222, 128))
]

def build_master_panel():
    cell_w = 540
    cell_h = 304
    pad_x = 24
    pad_y = 30
    header_h = 100
    row_label_h = 24
    row_gap = 36

    n_cols = len(MODELS)
    n_rows = len(VIEWS)

    total_w = pad_x * 2 + n_cols * cell_w + (n_cols - 1) * 16
    total_h = header_h + n_rows * (cell_h + row_label_h + row_gap) + pad_y

    canvas = Image.new("RGB", (total_w, total_h), color=(11, 15, 25))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad_x, 20), "F.O.R.G.E. CRITICAL INDOOR RECONSTRUCTION QUALITY RECOVERY", fill=(56, 189, 248))
    draw.text((pad_x, 48), "Hostel Room B 314 | True Room Enclosure (3.20m x 2.27m x 3.75m) | Analytical Camera Chaining & Topology Filtering", fill=(148, 163, 184))

    for c_idx, (m_id, label, color) in enumerate(MODELS):
        x = pad_x + c_idx * (cell_w + 16)
        draw.rectangle([x, 76, x + cell_w, 98], fill=(24, 32, 47))
        draw.text((x + 10, 80), label, fill=color)

    for r_idx, (view_id, view_title) in enumerate(VIEWS):
        y_row = header_h + r_idx * (cell_h + row_label_h + row_gap)
        draw.rectangle([pad_x, y_row, pad_x + n_cols * cell_w + (n_cols - 1) * 16, y_row + row_label_h], fill=(20, 27, 40))
        draw.text((pad_x + 8, y_row + 4), view_title, fill=(255, 255, 255))

        for c_idx, (m_id, label, color) in enumerate(MODELS):
            x = pad_x + c_idx * (cell_w + 16)
            y = y_row + row_label_h + 6
            img_path = RENDERS_DIR / f"render_{m_id}_{view_id}.png"
            if img_path.exists():
                im = Image.open(img_path).convert("RGB")
                im_res = im.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                canvas.paste(im_res, (x, y))

    out_file = EXP_DIR / "critical_recovery_master_panel.png"
    canvas.save(out_file, quality=95)
    canvas.save(ARTIFACTS_DIR / "critical_recovery_master_panel.png", quality=95)
    print(f"[+] Saved master panel to {out_file} and artifacts.")

if __name__ == "__main__":
    build_master_panel()
