import os
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = r"outputs\indoor\photographic_texture_quality_recovery_video1\renders"
ARTIFACTS_DIR = r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953"

PANEL_ITEMS = [
    ("test_00_inspection_mode.png", "1. MODE A: 3D ORBIT INSPECTION (INITIAL OVERVIEW)", "#38bdf8"),
    ("test_01_walkthrough_entry.png", "2. MODE B: ENTER SPACE (HUMAN EYE-LEVEL SPAWN Y=0.093m)", "#22c55e"),
    ("test_02_walk_forward.png", "3. WASD WALKTHROUGH (FORWARD DOWN WALKABLE CORRIDOR)", "#06b6d4"),
    ("test_03_bed_wall_collision.png", "4. COLLISION PHYSICS (BED OBSTACLE SLIDE - NO PASS-THROUGH)", "#f59e0b"),
    ("test_04_boundary_limit.png", "5. SPATIAL CLAMPING (ROOM PERIMETER - NO ESCAPE)", "#ef4444"),
    ("test_05_camera_rotated.png", "6. MOUSE LOOK ROTATION (DESK, COMPUTER & POSTER TEXTURES)", "#a855f7"),
    ("test_06_minimap_active.png", "7. TOP-DOWN FLOOR PLAN MINI-MAP (LIVE FOV CONE & TRAJECTORY)", "#3b82f6"),
    ("test_07_pointcloud_mode.png", "8. POINT-CLOUD MODE TOGGLE ('P' KEY INTERIOR PERSPECTIVE)", "#10b981"),
]

def build_panel():
    cell_w, cell_h = 720, 450
    header_h = 42
    cols, rows = 2, 4
    total_w = cols * cell_w
    total_h = rows * (cell_h + header_h) + 80

    master = Image.new("RGB", (total_w, total_h), (15, 23, 42))
    draw = ImageDraw.Draw(master)

    # Top title
    draw.text((30, 20), "F.O.R.G.E. IMMERSIVE WALKTHROUGH ENGINE — 12-POINT QA VALIDATION PANEL", fill=(248, 250, 252))
    draw.text((30, 50), "Dual-Layer Collision Physics • Human Eye-Level Navigation • Live Floor Plan Mini-Map • Real-Time Capture Frame Sync", fill=(148, 163, 184))

    for idx, (fname, label, color) in enumerate(PANEL_ITEMS):
        c = idx % cols
        r = idx // cols
        x = c * cell_w
        y = 80 + r * (cell_h + header_h)

        # Header bar
        draw.rectangle([x, y, x + cell_w, y + header_h], fill=(30, 41, 59))
        draw.rectangle([x, y, x + 6, y + header_h], fill=color)
        draw.text((x + 16, y + 12), label, fill=(241, 245, 249))

        # Image
        img_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            master.paste(img_resized, (x, y + header_h))
        else:
            draw.rectangle([x, y + header_h, x + cell_w, y + header_h + cell_h], fill=(10, 15, 29))
            draw.text((x + cell_w // 2 - 80, y + header_h + cell_h // 2), f"Missing: {fname}", fill=(239, 68, 68))

    out_panel_path = os.path.join(OUT_DIR, "master_walkthrough_validation_panel.png")
    master.save(out_panel_path, quality=95)
    print(f"[+] Saved master panel to: {out_panel_path}")

    artifact_panel_path = os.path.join(ARTIFACTS_DIR, "master_walkthrough_validation_panel.png")
    master.save(artifact_panel_path, quality=95)
    print(f"[+] Saved master panel artifact to: {artifact_panel_path}")

if __name__ == "__main__":
    build_panel()

