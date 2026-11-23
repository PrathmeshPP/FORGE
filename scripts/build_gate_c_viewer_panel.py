"""
Assemble Side-by-Side Three.js Viewer Comparison Panel:
- BASELINE — DEFECTIVE
- CANDIDATE A — CORRECTED SIM(3)
- CANDIDATE B — RIGID SE(3)
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GATE_C_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031/gate_c_extended"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

cols = [
    {
        "title": "BASELINE — DEFECTIVE",
        "tag": "Unrecomputed Translation (2.54m Drift)",
        "badge_color": (239, 68, 68),  # red
        "overview": GATE_C_DIR / "viewer_baseline_defective_overview.png",
        "top_plan": GATE_C_DIR / "viewer_baseline_defective_top_plan.png"
    },
    {
        "title": "CANDIDATE A — CORRECTED SIM(3)",
        "tag": "Bounded Sim(3) + Recomputed t + Cam Constraints (OPTIMAL)",
        "badge_color": (34, 197, 94),  # green
        "overview": GATE_C_DIR / "viewer_candidate_a_overview.png",
        "top_plan": GATE_C_DIR / "viewer_candidate_a_top_plan.png"
    },
    {
        "title": "CANDIDATE B — RIGID SE(3)",
        "tag": "Rigid s=1.000 + Recomputed t (12.05cm Error Spike)",
        "badge_color": (245, 158, 11),  # amber
        "overview": GATE_C_DIR / "viewer_candidate_b_overview.png",
        "top_plan": GATE_C_DIR / "viewer_candidate_b_top_plan.png"
    }
]

# We crop the 3D viewport region from the full 1680x980 window:
# Left sidebar is ~280px, right inspection panel is ~320px, top toolbar is ~60px
# Viewport is approx (x: 280..1360, y: 70..980) -> width 1080, height 910
# Or we can show the full viewer card with UI context: width 560, height 326 each!

cell_w, cell_h = 560, 326
header_h = 70
subhead_h = 36
panel_w = cell_w * 3
panel_h = header_h + subhead_h + cell_h * 2 + 10

panel = Image.new("RGB", (panel_w, panel_h), color=(15, 23, 42))
draw = ImageDraw.Draw(panel)

try:
    font_title = ImageFont.truetype("arialbd.ttf", 20)
    font_col = ImageFont.truetype("arialbd.ttf", 15)
    font_tag = ImageFont.truetype("arial.ttf", 11)
    font_row = ImageFont.truetype("arialbd.ttf", 12)
except Exception:
    font_title = ImageFont.load_default()
    font_col = ImageFont.load_default()
    font_tag = ImageFont.load_default()
    font_row = ImageFont.load_default()

# Header banner
draw.rectangle([0, 0, panel_w, header_h], fill=(30, 41, 59))
draw.text((25, 14), "F.O.R.G.E EXPLORER — GATE C (36 FRAMES, 11 WINDOWS) THREE.JS VIEWER COMPARISON", fill=(56, 189, 248), font=font_title)
draw.text((25, 42), "Evaluating Geometry Stability, Wall Continuity, and Ghost Surface Suppression in Production WebGL", fill=(148, 163, 184), font=font_tag)

for c_idx, col in enumerate(cols):
    x = c_idx * cell_w
    # Column Subheader
    draw.rectangle([x, header_h, x + cell_w, header_h + subhead_h], fill=(24, 32, 47))
    draw.rectangle([x + 12, header_h + 8, x + 20, header_h + 26], fill=col["badge_color"])
    draw.text((x + 28, header_h + 7), col["title"], fill=(241, 245, 249), font=font_col)
    draw.text((x + 28, header_h + 22), col["tag"], fill=(148, 163, 184), font=font_tag)
    draw.line([(x + cell_w, header_h), (x + cell_w, panel_h)], fill=(51, 65, 85), width=1)
    
    # Row 1: Perspective Overview
    y1 = header_h + subhead_h
    im_ov = Image.open(col["overview"]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    panel.paste(im_ov, (x, y1))
    draw.rectangle([x, y1, x + 160, y1 + 22], fill=(15, 23, 42, 220))
    draw.text((x + 8, y1 + 4), "PERSPECTIVE ROOM VIEW", fill=(226, 232, 240), font=font_row)
    
    # Row 2: Top-down Floorplan
    y2 = y1 + cell_h
    im_top = Image.open(col["top_plan"]).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
    panel.paste(im_top, (x, y2))
    draw.rectangle([x, y2, x + 160, y2 + 22], fill=(15, 23, 42, 220))
    draw.text((x + 8, y2 + 4), "TOP-DOWN PLAN (WALL CHECK)", fill=(226, 232, 240), font=font_row)
    
    # Cell outlines
    draw.rectangle([x, y1, x + cell_w, y1 + cell_h], outline=(51, 65, 85), width=1)
    draw.rectangle([x, y2, x + cell_w, y2 + cell_h], outline=(51, 65, 85), width=1)

out_panel = GATE_C_DIR / "viewer_gate_c_side_by_side_panel.png"
panel.save(out_panel, quality=95)
print(f"[+] Saved Three.js Viewer Side-by-Side Comparison Panel: {out_panel.name}")

# Also copy to artifacts directory
artifact_panel = ARTIFACT_DIR / "viewer_gate_c_side_by_side_panel.png"
panel.save(artifact_panel, quality=95)
print(f"[+] Copied to artifacts directory: {artifact_panel.name}")

# Also copy individual viewer images to artifact directory
for im_name in [
    "viewer_candidate_a_overview.png",
    "viewer_candidate_a_top_plan.png",
    "viewer_candidate_a_isometric.png",
    "viewer_candidate_a_point_cloud.png",
    "viewer_baseline_defective_overview.png",
    "viewer_baseline_defective_top_plan.png",
    "viewer_candidate_b_overview.png",
    "viewer_candidate_b_top_plan.png"
]:
    src = GATE_C_DIR / im_name
    if src.exists():
        dst = ARTIFACT_DIR / im_name
        import shutil
        shutil.copy(src, dst)
        print(f"  Copied {im_name} to artifacts directory")

