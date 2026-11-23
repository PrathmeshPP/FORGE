r"""
F.O.R.G.E. — Phase 1: Video Audit, Sharpness Calibration & Adaptive Frame Selection
Target Video: C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4
Output: outputs/indoor/outdoor_pipeline_applied_video1/
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4")
OUT_DIR = PROJECT_ROOT / "outputs/indoor/outdoor_pipeline_applied_video1"
FRAMES_DIR = OUT_DIR / "frames"
DIAG_DIR = OUT_DIR / "diagnostics"

for d in [OUT_DIR, FRAMES_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — STEP 1: VIDEO AUDIT, SHARPNESS CALIBRATION & ADAPTIVE CURATION")
print(f"  Target Video: {VIDEO_PATH}")
print(f"  Target Dir:   {OUT_DIR}")
print("=" * 76)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Failed to open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 29.8
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = total_frames / fps

print(f"[*] Metadata: {width}x{height}, {fps:.2f} FPS, {total_frames} frames ({duration:.2f}s)")

# 1. Full Video Audit: Sharpness, Disparity & Feature Overlap
print(f"[*] Auditing all {total_frames} frames...")
orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

metrics = []
prev_gray = None
prev_kp = None
prev_des = None

f_idx = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break
    t_sec = f_idx / fps
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Laplacian sharpness
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Inter-frame disparity proxy
    if prev_gray is not None:
        g1 = cv2.resize(prev_gray, (160, 160))
        g2 = cv2.resize(gray, (160, 160))
        disp = float(np.mean(cv2.absdiff(g1, g2)))
    else:
        disp = 0.0
        
    kp, des = orb.detectAndCompute(gray, None)
    overlap = 1.0
    if prev_des is not None and des is not None and len(des) > 10 and len(prev_des) > 10:
        matches = bf.match(prev_des, des)
        overlap = len(matches) / max(1, min(len(prev_des), len(des)))
        
    metrics.append({
        "frame_idx": f_idx,
        "timestamp_sec": round(t_sec, 3),
        "sharpness": round(lap_var, 2),
        "disparity": round(disp, 2),
        "overlap": round(overlap, 3),
        "feature_count": len(kp) if kp is not None else 0
    })
    
    prev_gray = gray
    prev_kp = kp
    prev_des = des
    f_idx += 1

cap.release()

# 2. Calibrate Sharpness Distribution
sharpnesses = np.array([m["sharpness"] for m in metrics])
p10 = float(np.percentile(sharpnesses, 10))
p25 = float(np.percentile(sharpnesses, 25))
p50 = float(np.median(sharpnesses))
p75 = float(np.percentile(sharpnesses, 75))
p90 = float(np.percentile(sharpnesses, 90))

print("\n[*] Laplacian Sharpness Calibration:")
print(f"    Min:    {np.min(sharpnesses):.2f}")
print(f"    p10:    {p10:.2f}")
print(f"    p25:    {p25:.2f}")
print(f"    Median: {p50:.2f}")
print(f"    p75:    {p75:.2f}")
print(f"    p90:    {p90:.2f}")
print(f"    Max:    {np.max(sharpnesses):.2f}")

# Setting blur threshold at p25 (clearing out blurry turns while retaining coverage)
calibrated_blur_thresh = max(p25, 40.0)
print(f"    Calibrated Blur Threshold: {calibrated_blur_thresh:.2f}")

# 3. Adaptive Keyframe Selection: Initial Candidate Set (16 frames) + Adaptive Expansion
# Partition video into 16 uniform temporal bins, select sharpest frame with sufficient feature richness in each bin
target_base_bins = 16
bin_size = total_frames / target_base_bins
candidate_indices = []

for b in range(target_base_bins):
    b_start = int(b * bin_size)
    b_end = int((b + 1) * bin_size)
    bin_metrics = metrics[b_start:b_end]
    
    # Filter candidates by calibrated blur threshold
    valid_in_bin = [m for m in bin_metrics if m["sharpness"] >= calibrated_blur_thresh]
    if not valid_in_bin:
        # Fallback to the single best in this bin
        valid_in_bin = bin_metrics
        
    # Sort by a joint score of sharpness and feature count
    best_frame = max(valid_in_bin, key=lambda m: m["sharpness"] * np.log1p(m["feature_count"]))
    candidate_indices.append(best_frame["frame_idx"])

print(f"\n[*] Initial 16 Base Geometry Keyframes: {candidate_indices}")

# Adaptive Expansion: Check for coverage gaps or high disparity jumps between consecutive keyframes
# If cumulative disparity between consecutive keyframes is very high, insert intermediate sharp frames
cap = cv2.VideoCapture(str(VIDEO_PATH))
expanded_indices = list(candidate_indices)

# Check consecutive keyframe pairs for large disparity gaps
inserted_count = 0
for i in range(len(candidate_indices) - 1):
    idx_a = candidate_indices[i]
    idx_b = candidate_indices[i + 1]
    
    # Cumulative disparity between a and b
    segment_disps = [metrics[k]["disparity"] for k in range(idx_a, idx_b)]
    cum_disp = sum(segment_disps)
    
    # If disparity gap is above 75th percentile of all inter-frame gaps, insert mid-point sharp frame
    if cum_disp > 180.0 and (idx_b - idx_a) > 20:
        mid_range = range(idx_a + (idx_b - idx_a) // 3, idx_b - (idx_b - idx_a) // 3)
        mid_metrics = [metrics[k] for k in mid_range if metrics[k]["sharpness"] >= calibrated_blur_thresh]
        if mid_metrics:
            best_mid = max(mid_metrics, key=lambda m: m["sharpness"])
            expanded_indices.append(best_mid["frame_idx"])
            inserted_count += 1

expanded_indices = sorted(list(set(expanded_indices)))
print(f"[*] Adaptive Expansion added {inserted_count} intermediate frames -> Total Curated Frames: {len(expanded_indices)}")

# 4. Extract and Save Curated Keyframes
saved_records = []
for order_idx, f_no in enumerate(expanded_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
    ret, frame_bgr = cap.read()
    if not ret:
        continue
    t_sec = f_no / fps
    fn = FRAMES_DIR / f"frame_{order_idx:04d}_idx{f_no:05d}_{t_sec:.3f}s.png"
    cv2.imwrite(str(fn), frame_bgr)
    
    m_info = metrics[f_no]
    saved_records.append({
        "order_idx": order_idx,
        "frame_idx": f_no,
        "timestamp_sec": round(t_sec, 3),
        "sharpness": m_info["sharpness"],
        "disparity": m_info["disparity"],
        "overlap": m_info["overlap"],
        "feature_count": m_info["feature_count"],
        "file_path": str(fn.resolve()),
        "file_name": fn.name
    })

cap.release()

# 5. Save Frame Selection Manifest & Report
report = {
    "video_path": str(VIDEO_PATH),
    "total_video_frames": total_frames,
    "fps": fps,
    "duration_sec": duration,
    "resolution": [width, height],
    "sharpness_stats": {
        "p10": p10, "p25": p25, "median": p50, "p75": p75, "p90": p90,
        "calibrated_threshold": calibrated_blur_thresh
    },
    "base_keyframes_count": len(candidate_indices),
    "expanded_keyframes_count": len(expanded_indices),
    "selected_frames": saved_records
}

manifest_path = DIAG_DIR / "curated_frames_manifest.json"
manifest_path.write_text(json.dumps(report, indent=2))
print(f"[+] Saved curated frames manifest to: {manifest_path}")
print(f"[+] Curated {len(saved_records)} keyframes spanning {duration:.2f}s of indoor video.")
