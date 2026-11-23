"""
F.O.R.G.E. — Master Frame Density Review Panel Builder.
Stitches identical-viewpoint renders into multi-configuration comparison panels.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = PROJECT_ROOT / "outputs/indoor/frame_density_experiment"
COMP_DIR = EXP_ROOT / "comparisons"


def build_dataset_panel(dataset_prefix: str, dataset_title: str, out_name: str):
    configs = [
        ("baseline_1_5fps", "Baseline (~1.5 FPS)"),
        ("exp_a_2fps", "Exp A (2.0 FPS)"),
        ("exp_b_3fps", "Exp B (3.0 FPS)"),
        ("exp_c_4fps", "Exp C (4.0 FPS)"),
        ("exp_d_dual_pool", "Exp D (Dual Pool: 2.0g/4.0t)")
    ]
    views = [
        ("overview", "Full Scene Overview"),
        ("furniture_closeup" if dataset_prefix == "v1" else "door_closeup", "Primary Detail Close-Up"),
        ("wall_closeup", "Wall Surface Detail"),
        ("floor_closeup", "Floor Perspective & Surface")
    ]

    card_w, card_h = 480, 300
    header_h = 70
    canvas_w = card_w * len(configs)
    canvas_h = card_h * len(views) + header_h

    panel = Image.new("RGB", (canvas_w, canvas_h), color=(15, 18, 24))
    draw = ImageDraw.Draw(panel)

    # Header
    draw.rectangle([0, 0, canvas_w, header_h], fill=(22, 27, 34))
    draw.text((25, 20), f"F.O.R.G.E. FRAME DENSITY BENCHMARK — {dataset_title}", fill=(56, 189, 248))
    draw.text((canvas_w - 480, 26), "Identical Perspective Controlled Experiment", fill=(148, 163, 184))

    for col_idx, (cfg_id, cfg_label) in enumerate(configs):
        x = col_idx * card_w
        for row_idx, (view_id, view_label) in enumerate(views):
            y = header_h + row_idx * card_h
            img_path = COMP_DIR / f"{dataset_prefix}_{cfg_id}_{view_id}.png"
            if img_path.exists():
                im = Image.open(img_path).resize((card_w, card_h), Image.Resampling.LANCZOS)
                panel.paste(im, (x, y))
            else:
                # Black placeholder
                draw.rectangle([x, y, x + card_w, y + card_h], fill=(30, 35, 45))
                draw.text((x + 20, y + card_h // 2), f"Pending: {cfg_id}_{view_id}", fill=(100, 116, 139))

            # Overlay label
            draw.rectangle([x, y, x + card_w, y + 28], fill=(15, 23, 42))
            draw.text((x + 10, y + 6), f"{cfg_label} — {view_label}", fill=(255, 255, 255))
            draw.rectangle([x, y, x + card_w, y + card_h], outline=(51, 65, 85), width=2)

    out_path = COMP_DIR / out_name
    panel.save(out_path, quality=95)
    print(f"[+] Saved Master Density Panel: {out_path.name}")


def main():
    COMP_DIR.mkdir(parents=True, exist_ok=True)
    build_dataset_panel("v1", "Student Hostel Room B 314", "video_1_frame_density_master_panel.png")
    build_dataset_panel("v2", "Hostel Corridor Walkthrough", "video_2_frame_density_master_panel.png")


if __name__ == "__main__":
    main()

