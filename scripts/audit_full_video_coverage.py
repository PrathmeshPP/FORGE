#!/usr/bin/env python3
"""
F.O.R.G.E V10 — Full Source Video Coverage Audit & Contact Sheet Generator
Examines the entire 179.7s DJI master recording to identify all scene passes,
viewpoints, trajectories, and useful frame intervals.
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
MASTER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V10_DIR = CAMPAIGN_DIR / "final_best_canonical_v10"
V10_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR = V10_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 80)
    print("F.O.R.G.E V10 — FULL SOURCE VIDEO COVERAGE AUDIT")
    print("=" * 80)

    if not MASTER_VIDEO.exists():
        print(f"[!] Error: Master video not found at {MASTER_VIDEO}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(MASTER_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    print(f"[*] Video File: {MASTER_VIDEO.name}")
    print(f"    Dimensions: {native_w} x {native_h} (Native 4K UHD)")
    print(f"    Framerate:  {fps:.3f} fps")
    print(f"    Total Frames: {total_frames:,}")
    print(f"    Total Duration: {duration_s:.2f} seconds ({duration_s/60:.2f} minutes)")

    # Sample every 2.5 seconds across the full duration
    sample_interval_s = 2.5
    sample_times = np.arange(0.0, duration_s, sample_interval_s)
    print(f"\n[*] Sampling {len(sample_times)} frames at {sample_interval_s}s intervals across full video...")

    sampled_records = []
    contact_thumbs = []

    for idx, t_sec in enumerate(sample_times):
        frame_idx = min(int(round(t_sec * fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        # Compute sharpness
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Downscale for contact sheet thumbnail (e.g. 480 x 270)
        thumb = cv2.resize(frame_bgr, (480, 270), interpolation=cv2.INTER_AREA)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        
        # Save record
        rec = {
            "sample_index": idx,
            "timestamp_seconds": round(float(t_sec), 3),
            "frame_index": frame_idx,
            "laplacian_var": round(lap_var, 1)
        }
        sampled_records.append(rec)
        contact_thumbs.append((t_sec, lap_var, thumb_rgb))

    cap.release()
    print(f"    Extracted {len(contact_thumbs)} sample frames successfully.")

    # Assemble Full-Video Contact Sheet (Grid: 8 cols x 9 rows = 72 thumbs)
    cols = 8
    rows = int(np.ceil(len(contact_thumbs) / cols))
    tw, th = 320, 180
    sheet_w = cols * (tw + 12) + 20
    sheet_h = rows * (th + 32) + 100

    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font_def = ImageFont.load_default()

    # Header
    draw.rectangle([(0, 0), (sheet_w, 80)], fill=(248, 250, 252))
    draw.line([(0, 80), (sheet_w, 80)], fill=(226, 232, 240), width=2)
    draw.text((30, 18), "F.O.R.G.E V10 — FULL SOURCE VIDEO DURATION SURVEY CONTACT SHEET", fill=(15, 23, 42), font=font_def)
    draw.text((30, 48), f"Master Video: {MASTER_VIDEO.name} | Duration: {duration_s:.1f}s ({total_frames:,} frames @ {fps:.2f}fps) | Sample Step: {sample_interval_s}s", fill=(100, 116, 139), font=font_def)

    for i, (t_sec, lap_var, thumb_rgb) in enumerate(contact_thumbs):
        c = i % cols
        r = i // cols
        x = 20 + c * (tw + 12)
        y = 100 + r * (th + 32)

        im_pil = Image.fromarray(thumb_rgb).resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im_pil, (x, y + 24))

        # Highlight frames in the primary target building pass (e.g. 30s to 75s)
        in_target_pass = 32.0 <= t_sec <= 70.0
        border_col = (37, 99, 235) if in_target_pass else (203, 213, 225)
        bg_col = (239, 246, 255) if in_target_pass else (241, 245, 249)

        draw.rectangle([(x, y), (x + tw, y + 22)], fill=bg_col)
        draw.text((x + 6, y + 5), f"t={t_sec:5.1f}s | LapVar={lap_var:5.0f}{' [TARGET]' if in_target_pass else ''}", fill=(15, 23, 42), font=font_def)
        draw.rectangle([(x, y), (x + tw, y + th + 24)], outline=border_col, width=2 if in_target_pass else 1)

    out_sheet_path = DIAG_DIR / "full_video_coverage_contact_sheet.png"
    sheet.save(out_sheet_path, format="PNG")
    print(f"[*] Saved Full Video Contact Sheet: {out_sheet_path} ({out_sheet_path.stat().st_size / (1024**2):.2f} MB)")

    # Save metadata JSON
    manifest_path = DIAG_DIR / "full_video_samples_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "video_path": str(MASTER_VIDEO),
            "dimensions": [native_w, native_h],
            "fps": fps,
            "total_frames": total_frames,
            "duration_seconds": duration_s,
            "samples": sampled_records
        }, f, indent=2)
    print(f"[*] Saved Samples Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
