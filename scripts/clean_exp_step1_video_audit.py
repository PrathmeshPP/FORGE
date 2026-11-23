"""
F.O.R.G.E. Clean-Slate Indoor Reconstruction Experiment — Step 1: Video Audit & Adaptive Curation
Source: WhatsApp Video 2026-10-30 at 17.02.34.mp4
Output: outputs/indoor/clean_reconstruction_video1_20261030
"""

import sys
import os
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4")
EXP_DIR = Path("outputs/indoor/clean_reconstruction_video1_20261030")
AUDIT_DIR = EXP_DIR / "audit"
FRAMES_DIR = EXP_DIR / "frames"
DIAG_DIR = EXP_DIR / "diagnostics"

for d in [AUDIT_DIR, FRAMES_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — CLEAN-SLATE STEP 1: VIDEO AUDIT & ADAPTIVE CURATION")
print(f"  Target Video: {VIDEO_PATH.name}")
print(f"  Target Dir:   {EXP_DIR}")
print("=" * 76)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

# 1. SHA-256 Checksum & Metadata
h = hashlib.sha256()
with open(VIDEO_PATH, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha256 = h.hexdigest()
file_size_bytes = VIDEO_PATH.stat().st_size
file_size_mb = file_size_bytes / (1024 * 1024)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 29.800
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

print(f"[*] Video Metadata:")
print(f"    File Size:   {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
print(f"    SHA-256:     {sha256}")
print(f"    Resolution:  {width}x{height} ({'Portrait' if height > width else 'Landscape'})")
print(f"    Framerate:   {fps:.3f} FPS")
print(f"    Frames:      {total_frames}")
print(f"    Duration:    {duration_sec:.2f} s")
print(f"    Codec:       {fourcc_str}")

# 2. Comprehensive Frame-by-Frame Video Audit
print(f"\n[*] Scanning all {total_frames} frames for motion blur, sharpness, brightness & feature flow...")
orb = cv2.ORB_create(nfeatures=500)

frame_metrics = []
prev_gray = None
prev_kp = None
prev_des = None
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

contact_thumbs = []
contact_indices = np.linspace(0, total_frames - 1, 36, dtype=int)

f_idx = 0
t0_scan = time.perf_counter()

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    t_sec = f_idx / fps
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (sharpness)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_lum = float(np.mean(gray))
    std_lum = float(np.std(gray))

    # Inter-frame disparity / optical flow proxy
    if prev_gray is not None:
        g1 = cv2.resize(prev_gray, (160, 160))
        g2 = cv2.resize(gray, (160, 160))
        disp = float(np.mean(cv2.absdiff(g1, g2)))
    else:
        disp = 0.0

    # Feature match overlap
    kp, des = orb.detectAndCompute(gray, None)
    overlap_ratio = 1.0
    if prev_des is not None and des is not None and len(des) > 10 and len(prev_des) > 10:
        matches = bf.match(prev_des, des)
        overlap_ratio = len(matches) / max(1, min(len(prev_des), len(des)))

    # Save thumb for contact sheet
    if f_idx in contact_indices:
        thumb = cv2.resize(frame_bgr, (240, int(240 * height / width)))
        contact_thumbs.append((f_idx, t_sec, lap_var, thumb))

    frame_metrics.append({
        "frame_idx": f_idx,
        "timestamp_sec": round(t_sec, 3),
        "sharpness": round(lap_var, 2),
        "luminance": round(mean_lum, 2),
        "contrast": round(std_lum, 2),
        "disparity": round(disp, 2),
        "feature_count": len(kp) if kp is not None else 0,
        "overlap_ratio": round(overlap_ratio, 3)
    })

    prev_gray = gray
    prev_kp = kp
    prev_des = des
    f_idx += 1

cap.release()
scan_time = time.perf_counter() - t0_scan
print(f"[+] Full-duration scan completed in {scan_time:.2f}s ({total_frames/scan_time:.1f} FPS)")

# Audit Statistics
sharpnesses = np.array([m["sharpness"] for m in frame_metrics])
luminances = np.array([m["luminance"] for m in frame_metrics])
contrasts = np.array([m["contrast"] for m in frame_metrics])
disparities = np.array([m["disparity"] for m in frame_metrics])
overlaps = np.array([m["overlap_ratio"] for m in frame_metrics])

blurry_thresh = 50.0
blurry_count = int(np.sum(sharpnesses < blurry_thresh))
blurry_pct = blurry_count / total_frames * 100.0

audit_stats = {
    "total_frames": total_frames,
    "duration_sec": round(duration_sec, 3),
    "fps": round(fps, 3),
    "resolution": [width, height],
    "orientation": "portrait",
    "sharpness": {
        "min": round(float(np.min(sharpnesses)), 2),
        "p10": round(float(np.percentile(sharpnesses, 10)), 2),
        "p25": round(float(np.percentile(sharpnesses, 25)), 2),
        "median": round(float(np.median(sharpnesses)), 2),
        "mean": round(float(np.mean(sharpnesses)), 2),
        "p75": round(float(np.percentile(sharpnesses, 75)), 2),
        "max": round(float(np.max(sharpnesses)), 2),
        "blurry_threshold": blurry_thresh,
        "blurry_frames_count": blurry_count,
        "blurry_frames_pct": round(blurry_pct, 2)
    },
    "luminance": {
        "min": round(float(np.min(luminances)), 2),
        "median": round(float(np.median(luminances)), 2),
        "mean": round(float(np.mean(luminances)), 2),
        "max": round(float(np.max(luminances)), 2)
    },
    "motion_disparity": {
        "min": round(float(np.min(disparities)), 2),
        "median": round(float(np.median(disparities)), 2),
        "mean": round(float(np.mean(disparities)), 2),
        "p90": round(float(np.percentile(disparities, 90)), 2),
        "max": round(float(np.max(disparities)), 2)
    },
    "scene_phases": [
        {"phase": "1. Entrance & Door Approach", "time_range": [0.0, 5.0], "notes": "Camera approaches wooden door from hallway"},
        {"phase": "2. Doorway Transition", "time_range": [5.0, 9.5], "notes": "Walking through doorway; rapid translation & significant motion blur"},
        {"phase": "3. Bed & Room Interior", "time_range": [9.5, 23.0], "notes": "Panning across single bed, bedding, mattress, adjacent wall"},
        {"phase": "4. Desk, Window & Air Conditioner", "time_range": [23.0, 34.0], "notes": "Scanning study desk, window bars, exterior daylight, AC unit"},
        {"phase": "5. Wardrobe & Return Toward Door", "time_range": [34.0, 40.2], "notes": "Tall wooden wardrobe, shoe rack, turning back toward door exit"}
    ]
}

print(f"\n[*] Sharpness Summary:")
print(f"    Median: {audit_stats['sharpness']['median']} | Mean: {audit_stats['sharpness']['mean']} | Max: {audit_stats['sharpness']['max']}")
print(f"    Blurry frames (< {blurry_thresh}): {blurry_count}/{total_frames} ({blurry_pct:.1f}%)")

(AUDIT_DIR / "video_quality_audit.json").write_text(json.dumps(audit_stats, indent=2))
print(f"[+] Saved audit metrics to {AUDIT_DIR / 'video_quality_audit.json'}")

# 3. Render 6x6 Contact Sheet
print("[*] Generating full-duration 36-frame contact sheet...")
thumb_w, thumb_h = 240, int(240 * height / width)
cols, rows = 6, 6
sheet_w = cols * thumb_w
sheet_h = rows * thumb_h + 80
contact_sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 24, 30))
draw = ImageDraw.Draw(contact_sheet)

try:
    font_header = ImageFont.truetype("arial.ttf", 26)
    font_tag = ImageFont.truetype("arial.ttf", 14)
except Exception:
    font_header = ImageFont.load_default()
    font_tag = font_header

draw.rectangle([(0, 0), (sheet_w, 75)], fill=(30, 36, 46))
draw.text((25, 22), "F.O.R.G.E. — VIDEO AUDIT CONTACT SHEET (HOSTEL ROOM B 314)", fill=(240, 245, 255), font=font_header)

for i, (f_i, t_s, s_v, t_img) in enumerate(contact_thumbs[:36]):
    c = i % cols
    r = i // cols
    x = c * thumb_w
    y = 80 + r * thumb_h

    t_rgb = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
    pil_thumb = Image.fromarray(t_rgb)
    contact_sheet.paste(pil_thumb, (x, y))

    # Badge overlay
    draw.rectangle([(x + 5, y + 5), (x + 130, y + 38)], fill=(10, 12, 16, 220))
    tag_col = (100, 230, 120) if s_v >= 50.0 else (240, 100, 90)
    draw.text((x + 10, y + 8), f"t={t_s:04.1f}s", fill=(255, 255, 255), font=font_tag)
    draw.text((x + 10, y + 22), f"S={s_v:04.1f}", fill=tag_col, font=font_tag)

contact_sheet_p = AUDIT_DIR / "video_audit_contact_sheet.png"
contact_sheet.save(str(contact_sheet_p), quality=95)
print(f"[+] Saved contact sheet to {contact_sheet_p.name}")

# ==============================================================================
# 4. ADAPTIVE FRAME SELECTION
# ==============================================================================
print("\n" + "=" * 76)
print("  STEP 2: MULTI-CRITERIA ADAPTIVE FRAME SELECTION")
print("=" * 76)

# Baseline 1 FPS uniform selection
total_sec = int(math.ceil(duration_sec))
baseline_indices = []
for s in range(total_sec):
    st = int(s * fps)
    en = min(int((s + 1) * fps), total_frames)
    cands = frame_metrics[st:en]
    if cands:
        # best sharpness in that 1-second slot
        best_cand = max(cands, key=lambda c: c["sharpness"])
        baseline_indices.append(best_cand["frame_idx"])

print(f"[*] Baseline 1.0 FPS: {len(baseline_indices)} frames selected.")

# Adaptive multi-criteria selection
# We want:
# 1. Base ~1 FPS for steady global coverage
# 2. Rejection of severely blurred frames (sharpness < 25) unless absolutely no alternative
# 3. Targeted high-density sampling in Phase 2 (door entry transition: t=5s to 9.5s) to guarantee high correspondence in Window 2
# 4. Adaptive sampling when camera translation / angular disparity is high (> 20 disparity)
# 5. Overlap preservation so consecutive frames have >= 30% feature overlap

selected_set = set()

for s in range(total_sec):
    st = int(s * fps)
    en = min(int((s + 1) * fps), total_frames)
    cands = frame_metrics[st:en]
    if not cands:
        continue
    # Filter out severely blurred
    sharp_cands = [c for c in cands if c["sharpness"] >= 35.0]
    if not sharp_cands:
        sharp_cands = cands
    best_cand = max(sharp_cands, key=lambda c: c["sharpness"])
    selected_set.add(best_cand["frame_idx"])

# Specialized door entry reinforcement (t=4.5s to 9.5s)
door_cands = [m for m in frame_metrics if 4.5 <= m["timestamp_sec"] <= 9.5 and m["sharpness"] >= 40.0]
# Pick the top 8 sharpest non-redundant frames during the door transition
door_sorted = sorted(door_cands, key=lambda c: c["sharpness"], reverse=True)
door_added = 0
for dc in door_sorted:
    idx = dc["frame_idx"]
    # Check if too close to already selected (< 8 frames / ~0.27s)
    if not any(abs(idx - s_idx) < 8 for s_idx in selected_set):
        selected_set.add(idx)
        door_added += 1
        if door_added >= 6:
            break
print(f"[*] Added {door_added} door-transition keyframes to stabilize Window 2.")

# High disparity / turn regions
sorted_curr = sorted(list(selected_set))
turn_added = 0
for i in range(len(sorted_curr) - 1):
    idx_a = sorted_curr[i]
    idx_b = sorted_curr[i + 1]
    gap = idx_b - idx_a
    # If gap is > 20 frames (~0.7s) and mean disparity is high
    inter_cands = frame_metrics[idx_a + 5 : idx_b - 5]
    if inter_cands:
        mean_disp = np.mean([m["disparity"] for m in inter_cands])
        if mean_disp >= 22.0 or gap >= 35:
            sharp_inter = [m for m in inter_cands if m["sharpness"] >= 45.0]
            if sharp_inter:
                best_inter = max(sharp_inter, key=lambda c: c["sharpness"])
                selected_set.add(best_inter["frame_idx"])
                turn_added += 1

print(f"[*] Added {turn_added} motion/disparity keyframes in turns.")

final_selected_indices = sorted(list(selected_set))
final_count = len(final_selected_indices)
print(f"[+] Total Adaptive Selected Frames: {final_count} ({final_count / duration_sec:.2f} FPS)")

# Save curated frames as native-resolution PNGs
cap = cv2.VideoCapture(str(VIDEO_PATH))
curated_records = []

for rank, f_i in enumerate(final_selected_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_i)
    ret, frame_bgr = cap.read()
    if not ret:
        continue

    m = frame_metrics[f_i]
    fname = f"frame_{rank:03d}_f{f_i:04d}_{m['timestamp_sec']:05.2f}s.png"
    fpath = FRAMES_DIR / fname
    cv2.imwrite(str(fpath), frame_bgr)

    curated_records.append({
        "rank": rank,
        "filename": fname,
        "file_path": str(fpath.resolve()),
        "frame_index": f_i,
        "timestamp_sec": m["timestamp_sec"],
        "sharpness": m["sharpness"],
        "luminance": m["luminance"],
        "contrast": m["contrast"],
        "disparity": m["disparity"],
        "is_baseline_1fps": f_i in baseline_indices
    })

cap.release()

# Comparison summary
comparison_report = {
    "video_path": str(VIDEO_PATH),
    "video_sha256": sha256,
    "total_video_frames": total_frames,
    "duration_sec": round(duration_sec, 3),
    "baseline_1fps_count": len(baseline_indices),
    "baseline_1fps_density": round(len(baseline_indices) / duration_sec, 3),
    "adaptive_selected_count": final_count,
    "adaptive_density_fps": round(final_count / duration_sec, 3),
    "mean_sharpness_all_frames": round(float(np.mean(sharpnesses)), 2),
    "mean_sharpness_selected": round(float(np.mean([r["sharpness"] for r in curated_records])), 2),
    "median_sharpness_selected": round(float(np.median([r["sharpness"] for r in curated_records])), 2),
    "frames": curated_records
}

(DIAG_DIR / "frame_selection_comparison.json").write_text(json.dumps(comparison_report, indent=2))
print(f"[+] Saved frame comparison report to {DIAG_DIR / 'frame_selection_comparison.json'}")
print(f"    Selected frames mean sharpness: {comparison_report['mean_sharpness_selected']:.1f} (vs video mean {comparison_report['mean_sharpness_all_frames']:.1f})")

