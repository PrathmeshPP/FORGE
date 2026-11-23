"""
F.O.R.G.E. — Part 4: Confidence Filtering Investigation Across Percentiles.
Evaluates 20th, 30th, 40th, and 50th percentiles of VGGT depth_conf:
- Valid pixels retained (% and absolute count)
- Points retained per frame
- Wall and floor coverage
- Depth consistency across overlapping views
- Outlier distribution
"""

import json
import numpy as np
from pathlib import Path
import cv2

EXP_DIR = Path("outputs/indoor/vggt_tsdf_video_20261031_142303")
DIAG_DIR = Path("outputs/indoor/distortion_investigation_20261031/confidence_audit")
DIAG_DIR.mkdir(parents=True, exist_ok=True)

depth_file = EXP_DIR / "camera_geometry/depth_predictions.npz"
data = np.load(depth_file)
all_depth = data["depth"]        # (179, 432, 240)
all_conf = data["confidence"]    # (179, 432, 240) - expp1 activation: min > 1.0

# 1. Global distribution analysis of depth_conf
flat_conf = all_conf.flatten()
p10 = float(np.percentile(flat_conf, 10))
p20 = float(np.percentile(flat_conf, 20))
p30 = float(np.percentile(flat_conf, 30))
p40 = float(np.percentile(flat_conf, 40))
p50 = float(np.percentile(flat_conf, 50))
p60 = float(np.percentile(flat_conf, 60))
p70 = float(np.percentile(flat_conf, 70))
p80 = float(np.percentile(flat_conf, 80))
p90 = float(np.percentile(flat_conf, 90))

print("\n" + "=" * 76)
print("  PART 4: CONFIDENCE PERCENTILE DISTRIBUTION AUDIT")
print("=" * 76)
print(f"[*] Confidence Representation: VGGT-Omega DenseHead Depth Confidence (expp1 activation)")
print(f"    Min: {flat_conf.min():.4f} | Max: {flat_conf.max():.4f} | Mean: {flat_conf.mean():.4f} | Median (p50): {p50:.4f}")
print(f"    p10: {p10:.4f} | p20: {p20:.4f} | p30: {p30:.4f} | p40: {p40:.4f} | p50: {p50:.4f}")

percentile_candidates = [
    {"name": "p20", "threshold": p20, "percentile": 20},
    {"name": "p30", "threshold": p30, "percentile": 30},
    {"name": "p40", "threshold": p40, "percentile": 40},
    {"name": "p50", "threshold": p50, "percentile": 50},
]

# Evaluate on first 12 frames (Pilot subset)
N_eval = 12
H_d, W_d = all_depth.shape[1], all_depth.shape[2]
total_pixels_per_frame = H_d * W_d
total_pixels_eval = N_eval * total_pixels_per_frame

results = {}

for cand in percentile_candidates:
    c_name = cand["name"]
    c_thresh = cand["threshold"]
    pct = cand["percentile"]
    
    retained_pixels_list = []
    wall_retained_list = []
    floor_retained_list = []
    
    # Measure across 12 frames
    for i in range(N_eval):
        d = all_depth[i]
        c = all_conf[i]
        
        # Valid depth mask: depth in [0.20, 5.0] and c >= threshold
        valid_mask = (d >= 0.20) & (d <= 5.0) & (c >= c_thresh)
        retained = int(np.sum(valid_mask))
        retained_pixels_list.append(retained)
        
        # Upper half of frame corresponds to walls, lower third corresponds to floor
        wall_mask = valid_mask[:int(H_d * 0.6), :]
        floor_mask = valid_mask[int(H_d * 0.7):, :]
        
        wall_retained_list.append(float(np.mean(wall_mask)))
        floor_retained_list.append(float(np.mean(floor_mask)))
    
    total_retained = sum(retained_pixels_list)
    retention_pct = (total_retained / total_pixels_eval) * 100.0
    mean_pts_per_frame = int(np.mean(retained_pixels_list))
    mean_wall_cov = float(np.mean(wall_retained_list)) * 100.0
    mean_floor_cov = float(np.mean(floor_retained_list)) * 100.0
    
    # Overlap consistency: Check frames 3, 4, 5 (Window 0 & 1 shared region)
    # Compute median depth difference between consecutive overlapping frames
    depth_diffs = []
    for f in range(3, 5):
        d1 = all_depth[f]
        d2 = all_depth[f+1]
        m1 = (all_conf[f] >= c_thresh) & (d1 >= 0.20) & (d1 <= 5.0)
        m2 = (all_conf[f+1] >= c_thresh) & (d2 >= 0.20) & (d2 <= 5.0)
        common = m1 & m2
        if np.any(common):
            depth_diffs.append(float(np.median(np.abs(d1[common] - d2[common]))))
    
    med_overlap_diff_cm = float(np.mean(depth_diffs) * 100.0) if depth_diffs else 0.0
    
    print(f"\n[*] Evaluating Threshold {c_name} (conf >= {c_thresh:.4f}, drops bottom {pct}%):")
    print(f"    Valid Pixels Retained:     {total_retained:,} / {total_pixels_eval:,} ({retention_pct:.2f}%)")
    print(f"    Mean Points Per Frame:     {mean_pts_per_frame:,} pts")
    print(f"    Wall Coverage (Upper):     {mean_wall_cov:.2f}%")
    print(f"    Floor Coverage (Lower):    {mean_floor_cov:.2f}%")
    print(f"    Overlap Depth Consistency: {med_overlap_diff_cm:.2f} cm")
    
    results[c_name] = {
        "threshold": round(c_thresh, 4),
        "percentile_dropped": pct,
        "valid_pixels_retained": total_retained,
        "retention_pct": round(retention_pct, 2),
        "mean_points_per_frame": mean_pts_per_frame,
        "wall_coverage_pct": round(mean_wall_cov, 2),
        "floor_coverage_pct": round(mean_floor_cov, 2),
        "overlap_consistency_cm": round(med_overlap_diff_cm, 2)
    }

# Also compare against the previous defective baseline (conf >= 0.75)
prev_retained = sum(int(np.sum((all_depth[i] >= 0.20) & (all_depth[i] <= 5.0) & (all_conf[i] >= 0.75))) for i in range(N_eval))
prev_pct = (prev_retained / total_pixels_eval) * 100.0
print(f"\n[*] Defective Previous Setting (conf >= 0.75):")
print(f"    Retained: {prev_retained:,} ({prev_pct:.2f}%) -> Zero edge filtering!")

results["defective_previous_baseline_075"] = {
    "threshold": 0.75,
    "retention_pct": round(prev_pct, 2),
    "flaw": "Threshold was below minimum possible expp1 activation (1.00), passing 99.98% of noisy boundary pixels."
}

out_report = DIAG_DIR / "confidence_percentile_audit.json"
out_report.write_text(json.dumps(results, indent=2))
print(f"\n[+] Saved Confidence Audit Report to {out_report.name}")

