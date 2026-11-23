#!/usr/bin/env python3
"""
Generates the consolidated 4-scene visual review contact sheet:
outputs/opencv5_experiments/final_release_qa/four_scene_visual_review.png

Combines textured multi-view perspective cards (Oblique, Top Nadir, Frontal)
and neutral clay geometry views for all four release models:
1. Campus 0049 (Final V3)
2. River/Bridge (Final V3)
3. Desert Road (Accepted Phase 2B)
4. Forest Road (Accepted Phase 2B)
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QA_DIR = PROJECT_ROOT / "outputs/opencv5_experiments/final_release_qa"
OUT_CONTACT_SHEET = QA_DIR / "four_scene_visual_review.png"

SCENES = [
    {
        "key": "campus_0049",
        "title": "1. Campus 0049 (Institutional Complex & Grounds)",
        "version_tag": "Release Candidate: Final V3 (Phase 2D Boundary Recovery)",
        "provenance": "Provenance: Single-window corridor baseline (16.12s, 8 frames) to avoid drone hover jitter",
        "specs": "127,658 verts | 252,930 tris | Extents: [5.53m, 3.92m, 0.35m] | Atlas: 4096x4096 (28.72MB)",
        "topology": "Topology: 0 True Non-Manifold Edges | 2,416 Natural Boundary Edges (Open Terrain)",
        "status": "RELEASE STATUS: PASS",
        "status_color": (34, 197, 94), # green
        "textured_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/campus_0049/model_preview.png",
        "neutral_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/campus_0049/neutral_geometry_preview.png"
    },
    {
        "key": "river_bridge",
        "title": "2. River / Bridge (Highway Bridge & Rip-Rap Riverbank)",
        "version_tag": "Release Candidate: Final V3 (Phase 2D Corridor Extension & Water Rejection)",
        "provenance": "Provenance: Extended 30.0s flight (17.99s-48.00s) + water noise rejection (conf_min=2.0)",
        "specs": "179,119 verts | 353,895 tris | Extents: [4.23m, 0.89m, 3.77m] | Atlas: 4096x4096 (20.50MB)",
        "topology": "Topology: 0 True Non-Manifold Edges | 4,815 Natural Boundary Edges (Open Rip-Rap Riverbank)",
        "status": "RELEASE STATUS: PASS",
        "status_color": (34, 197, 94), # green
        "textured_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/river_bridge/model_preview.png",
        "neutral_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models_v3/river_bridge/neutral_geometry_preview.png"
    },
    {
        "key": "desert_road",
        "title": "3. Desert Road (Continuous Straight Highway Corridor)",
        "version_tag": "Release Candidate: Final V1 (Phase 2B OpenCV 5 Full Coronal Reconstruction)",
        "provenance": "Provenance: Genuine OpenCV 5 Curated Frame Selection (19.85s span, 2.8x wider baseline)",
        "specs": "18,680 verts | 35,764 tris | Extents: [2.70m, 0.29m, 4.12m] | Per-Vertex Shading",
        "topology": "Topology: 90 Boundary T-Junctions (<0.16% edges) | 1,756 Natural Boundary Edges (Open Dunes)",
        "status": "RELEASE STATUS: PASS WITH LIMITATIONS (Terrain Open Boundary)",
        "status_color": (234, 179, 8), # yellow
        "textured_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/desert_road/model_preview.png",
        "neutral_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/desert_road/neutral_geometry_preview.png"
    },
    {
        "key": "forest_road",
        "title": "4. Forest Road (Overhead Canopy & Asphalt Curvature)",
        "version_tag": "Release Candidate: Final V1 (Phase 2B OpenCV 5 Nadir Reconstruction)",
        "provenance": "Provenance: Genuine OpenCV 5 Curated Frame Selection (24.02s span, 3.4x wider baseline)",
        "specs": "94,521 verts | 188,430 tris | Extents: [1.66m, 0.98m, 0.54m] | Per-Vertex Shading",
        "topology": "Topology: 28 Canopy T-Junctions (<0.01% edges) | 692 Natural Boundary Edges (Open Canopy)",
        "status": "RELEASE STATUS: PASS WITH LIMITATIONS (Canopy Open Boundary)",
        "status_color": (234, 179, 8), # yellow
        "textured_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/forest_road/model_preview.png",
        "neutral_path": PROJECT_ROOT / "outputs/opencv5_experiments/final_models/forest_road/neutral_geometry_preview.png"
    }
]

def create_contact_sheet():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    print("[*] Generating master 4-scene visual review contact sheet...")

    # Load images
    scene_data = []
    target_tex_w = 2400
    target_tex_h = 660
    target_neu_w = 800
    target_neu_h = 660

    for s in SCENES:
        img_tex = Image.open(s["textured_path"]).convert("RGB")
        img_neu = Image.open(s["neutral_path"]).convert("RGB")

        # Resize if needed to target height 660
        if img_tex.size != (target_tex_w, target_tex_h):
            img_tex = img_tex.resize((target_tex_w, target_tex_h), Image.Resampling.LANCZOS)
        if img_neu.size != (target_neu_w, target_neu_h):
            img_neu = img_neu.resize((target_neu_w, target_neu_h), Image.Resampling.LANCZOS)

        scene_data.append((s, img_tex, img_neu))

    # Canvas dimensions
    margin_x = 40
    margin_top = 130
    row_gap = 40
    header_h = 90
    row_content_h = target_tex_h
    row_total_h = header_h + row_content_h

    total_w = margin_x * 2 + target_tex_w + 30 + target_neu_w
    total_h = margin_top + len(SCENES) * row_total_h + (len(SCENES) - 1) * row_gap + 60

    canvas = Image.new("RGB", (total_w, total_h), (11, 15, 25)) # deep sleek dark slate
    draw = ImageDraw.Draw(canvas)

    # Master Header
    draw.rectangle([(0, 0), (total_w, margin_top - 20)], fill=(15, 23, 42))
    draw.text((margin_x, 24), "F.O.R.G.E — FOUR-MODEL FINAL RELEASE QA & VISUAL AUDIT", fill=(241, 245, 249))
    draw.text((margin_x, 56), "Independent Multi-View Visual Verification: Textured Perspective Cards vs. Shaded Neutral Clay Geometry", fill=(148, 163, 184))
    draw.text((margin_x, 82), "Active Environment: Python 3.12.7 | OpenCV 5.0.0.93 | Three.js WebGL Verified | Local Storage Only", fill=(56, 189, 248))

    status_summary = "RELEASE STATUS: 2 PASS (Campus 0049 V3, River/Bridge V3) | 2 PASS WITH LIMITATIONS (Desert Road, Forest Road)"
    draw.text((total_w - margin_x - 900, 56), status_summary, fill=(34, 197, 94))

    # Render each row
    curr_y = margin_top
    for idx, (meta, img_tex, img_neu) in enumerate(scene_data):
        # Row Header Box
        header_y1 = curr_y
        header_y2 = curr_y + header_h - 10
        draw.rectangle([(margin_x, header_y1), (total_w - margin_x, header_y2)], fill=(20, 27, 45), outline=(51, 65, 85))

        # Title & Version
        draw.text((margin_x + 20, header_y1 + 10), meta["title"], fill=(255, 255, 255))
        draw.text((margin_x + 20, header_y1 + 32), meta["version_tag"], fill=(56, 189, 248))
        draw.text((margin_x + 20, header_y1 + 54), meta["provenance"], fill=(148, 163, 184))

        # Specs & Topology (Right-aligned)
        draw.text((margin_x + 1300, header_y1 + 10), meta["specs"], fill=(226, 232, 240))
        draw.text((margin_x + 1300, header_y1 + 32), meta["topology"], fill=(148, 163, 184))
        draw.text((margin_x + 1300, header_y1 + 54), meta["status"], fill=meta["status_color"])

        # Content Images
        content_y = curr_y + header_h
        canvas.paste(img_tex, (margin_x, content_y))
        canvas.paste(img_neu, (margin_x + target_tex_w + 30, content_y))

        # Borders around images
        draw.rectangle([(margin_x, content_y), (margin_x + target_tex_w, content_y + target_tex_h)], outline=(51, 65, 85), width=2)
        draw.rectangle([(margin_x + target_tex_w + 30, content_y), (margin_x + target_tex_w + 30 + target_neu_w, content_y + target_neu_h)], outline=(51, 65, 85), width=2)

        # Labels on images
        draw.text((margin_x + target_tex_w + 45, content_y + 15), "Neutral Clay Shading (Pure Topology)", fill=(0, 255, 200))

        curr_y += row_total_h + row_gap

    canvas.save(OUT_CONTACT_SHEET, format="PNG", optimize=True)
    print(f"[+] Consolidated contact sheet saved to: {OUT_CONTACT_SHEET}")
    print(f"    Dimensions: {canvas.size[0]}x{canvas.size[1]} | Size: {round(OUT_CONTACT_SHEET.stat().st_size / (1024*1024), 2)} MB")

if __name__ == "__main__":
    create_contact_sheet()
