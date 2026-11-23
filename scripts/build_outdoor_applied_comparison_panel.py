r"""
F.O.R.G.E. — Master Architectural Comparison Panel:
Proven Outdoor Pipeline Applied to Indoor Digital Twin vs Defective Baselines
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "outputs/indoor/outdoor_pipeline_applied_video1"
PREVIEWS_DIR = EXP_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

# Sources
IMG_BASELINE = PROJECT_ROOT / "outputs/indoor/clean_reconstruction_video1_20261030/diagnostics/master_five_way_comparison_panel_video1.png"
IMG_CLEAN_TSDF = PROJECT_ROOT / "outputs/indoor/clean_reconstruction_video1_20261030/previews/viewer_clean_overview.png"
IMG_OUTDOOR_APPLIED = PREVIEWS_DIR / "viewer_outdoor_applied_overview.png"
IMG_OUTDOOR_ISO = PREVIEWS_DIR / "viewer_outdoor_applied_isometric.png"
IMG_ATLAS = EXP_DIR / "textured/texture_atlas.png"
IMG_PCD = EXP_DIR / "point_clouds/indoor_fused_point_cloud.ply"

OUT_PANEL = PREVIEWS_DIR / "master_outdoor_pipeline_applied_comparison_panel.png"

def make_panel():
    w_card, h_card = 640, 380
    header_h = 90
    canvas_w = w_card * 3
    canvas_h = header_h + h_card * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)

    title = "F.O.R.G.E. DIGITAL TWIN — REUSED OUTDOOR ARCHITECTURE (POISSON HQ + 4K TEXTURE ATLAS)"
    subtitle = "Rigorous Validation: WhatsApp Video 2026-10-30 at 17.02.34.mp4 | 26 Keyframes | Screened Poisson Depth 9 vs Ad-Hoc TSDF"
    draw.text((25, 20), title, fill=(255, 255, 255))
    draw.text((25, 52), subtitle, fill=(180, 210, 240))

    cards = [
        ("1. Early Baseline (136s Micro-Steps)", PROJECT_ROOT / "viewer/screenshots/viewer_vggt_tsdf_room_overview.png", "Severe 2.54m drift, ghost walls, distorted floors", (240, 80, 80)),
        ("2. Clean-Slate TSDF (15mm Grid)", IMG_CLEAN_TSDF, "Reduced drift, but shredded thin furniture and voxel stair-stepping", (240, 180, 80)),
        ("3. Reused Outdoor Architecture (Overview)", IMG_OUTDOOR_APPLIED, "Watertight Poisson HQ Depth 9, smooth continuous walls & floors", (80, 240, 140)),
        ("4. Reused Outdoor Architecture (Isometric)", IMG_OUTDOOR_ISO, "Consistent per-view outward normals, sharp structural boundaries", (80, 220, 255)),
        ("5. Native 4K Texture Atlas (Photographic)", IMG_ATLAS, "MultiViewTextureEngine winner-take-all unblurred projection", (200, 140, 255)),
        ("6. Final Complete Digital Twin (Three.js)", PREVIEWS_DIR / "viewer_outdoor_applied_furniture_close.png", "Approved production digital twin ready for interactive inspection", (100, 255, 180)),
    ]

    for idx, (head, p, note, accent) in enumerate(cards):
        col = idx % 3
        row = idx // 3
        x = col * w_card
        y = header_h + row * h_card

        draw.rectangle([x, y, x + w_card, y + h_card], fill=(20, 24, 32), outline=(35, 42, 54), width=1)
        if p.is_file():
            im = Image.open(p).convert("RGB")
            im = im.resize((w_card - 10, h_card - 55), Image.Resampling.LANCZOS)
            canvas.paste(im, (x + 5, y + 48))
        else:
            draw.rectangle([x + 10, y + 50, x + w_card - 10, y + h_card - 10], fill=(28, 34, 44))
            draw.text((x + 40, y + 160), f"Preview Not Available ({p.name})", fill=(140, 160, 180))

        # Header tag
        draw.rectangle([x + 8, y + 8, x + w_card - 8, y + 42], fill=(12, 15, 20))
        draw.text((x + 15, y + 12), head, fill=accent)
        draw.text((x + 15, y + 27), note, fill=(180, 195, 215))

    canvas.save(OUT_PANEL, format="PNG")
    print(f"[+] Master comparison panel generated: {OUT_PANEL}")

if __name__ == "__main__":
    make_panel()

