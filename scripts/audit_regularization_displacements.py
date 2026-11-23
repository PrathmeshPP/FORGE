"""
F.O.R.G.E. — Forensic Script: Quantifying the Translation Displacements
Caused by Scale Clamping without Recomputing t_w in run_vggt_tsdf_experiment.py.
"""

import json
import numpy as np
from pathlib import Path

EXP_DIR = Path("outputs/indoor/vggt_tsdf_video_20261031_142303")
perf_file = EXP_DIR / "camera_geometry/vggt_inference_performance.json"
poses_file = EXP_DIR / "camera_geometry/camera_poses.npz"

perf = json.loads(perf_file.read_text())
poses = np.load(poses_file)
all_ext = poses["extrinsics"]

# Compute camera centers: C = -R.T @ t
centers = np.array([-all_ext[i, :3, :3].T @ all_ext[i, :3, 3] for i in range(len(all_ext))])
steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)

print("\n" + "=" * 76)
print("  FORENSIC PROOF: DISPLACEMENT INDUCED BY SCALE CLAMPING")
print("=" * 76)

align_logs = perf.get("alignment_logs", [])
displacements = []

for log in align_logs:
    w = log["window"]
    mode = log.get("mode", "")
    s_used = log["scale_sw"]
    if "Regularized" in mode:
        # Extract original scale from mode string: "Regularized (orig s=0.376, cond=0.134)"
        try:
            part = mode.split("orig s=")[1].split(",")[0]
            s_orig = float(part)
        except Exception:
            s_orig = s_used
        
        delta_s = s_used - s_orig
        # In this window, what frames were added?
        # Window size 6, overlap 3 -> window w covers frames: [w*3, w*3+5]
        frame_start = w * 3
        frame_end = min(frame_start + 6, len(all_ext))
        
        # Mean camera center in this window
        win_centers = centers[frame_start:frame_end]
        mean_c = np.mean(win_centers, axis=0)
        norm_c = float(np.linalg.norm(mean_c))
        
        # Theoretical displacement: delta_s * ||mu||
        disp_approx = abs(delta_s) * norm_c
        
        # Actual step size at window boundary
        step_at_boundary = float(steps[frame_start - 1]) if frame_start > 0 and frame_start - 1 < len(steps) else 0.0
        
        displacements.append({
            "window": w,
            "s_orig": s_orig,
            "s_clamped": s_used,
            "delta_s": round(delta_s, 4),
            "estimated_shift_m": round(disp_approx, 3),
            "boundary_step_cm": round(step_at_boundary * 100.0, 2),
            "mode": mode
        })
        print(f"  Window {w:02d} (frames {frame_start:03d}:{frame_end:03d}): orig s={s_orig:.3f} -> clamped s={s_used:.3f} | delta_s={delta_s:+.3f} | Induced Shift ~ {disp_approx*100:.1f} cm | Boundary Step = {step_at_boundary*100:.1f} cm")

print("\n" + "=" * 76)
print(f"[*] Total Regularization Displacements: {len(displacements)}")
cumulative_induced_drift = sum(d["estimated_shift_m"] for d in displacements)
print(f"[*] Cumulative Uncorrected Translation Drift: {cumulative_induced_drift:.2f} meters!")
print("=" * 76)

out_path = Path("outputs/indoor/distortion_investigation_20261031/regularization_displacements.json")
out_path.write_text(json.dumps(displacements, indent=2))
print(f"[+] Saved regularization displacement analysis to {out_path.name}")

