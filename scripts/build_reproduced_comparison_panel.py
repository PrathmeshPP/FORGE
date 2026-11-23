r"""
F.O.R.G.E. — Master Comparison Panel:
texture_experiment_video1 (Reference) vs Reproduced Experiment (New Output)
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "outputs/indoor/texture_experiment_video1"
REP_DIR = PROJECT_ROOT / "outputs/indoor/texture_experiment_video1_reproduced_video_170234"
OUT_PANEL = REP_DIR / "renders/master_reproduction_comparison_panel.png"


def make_comparison_panel():
    w_card, h_card = 520, 380
    header_h = 90
    canvas_w = w_card * 3
    canvas_h = header_h + h_card * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (14, 18, 24))
    draw = ImageDraw.Draw(canvas)

    title = "F.O.R.G.E. EXACT REPRODUCTION VALIDATION: texture_experiment_video1"
    subtitle = "Comparing Reference Model vs Newly Reproduced Model on WhatsApp Video 2026-10-30 at 17.02.34.mp4"
    draw.text((25, 20), title, fill=(255, 255, 255))
    draw.text((25, 52), subtitle, fill=(180, 210, 240))

    cards = [
        # Row 1: Eye-Level
        ("1. Reference: Eye-Level Render", REF_DIR / "renders/exp_photographic_eye_level.png", "texture_experiment_video1 (Original)", (0, 220, 255)),
        ("2. Reproduced: Eye-Level Render", REP_DIR / "renders/exp_photographic_eye_level.png", "reproduced_video_170234 (New Run)", (80, 255, 140)),
        ("3. Video Source Reference Frame", REP_DIR / "frames/frame_10_f0267_08.96s.png", "Native optical frame @ 478x850", (255, 200, 80)),

        # Row 2: Furniture Closeup & Texture Atlas
        ("4. Reference: Furniture Closeup", REF_DIR / "renders/exp_photographic_furniture_closeup.png", "texture_experiment_video1 (Original)", (0, 220, 255)),
        ("5. Reproduced: Furniture Closeup", REP_DIR / "renders/exp_photographic_furniture_closeup.png", "reproduced_video_170234 (New Run)", (80, 255, 140)),
        ("6. Reproduced: 4K Texture Atlas", REP_DIR / "texture_atlas.png", "4096x4096 UV Packed Atlas", (200, 140, 255)),
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
    make_comparison_panel()

