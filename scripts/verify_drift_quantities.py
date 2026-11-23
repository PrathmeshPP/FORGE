"""
F.O.R.G.E. — Forensic Analysis:
Independently verifying whether the reported 2.54 m drift is the scalar sum of
individual translation-error magnitudes, the actual cumulative vector endpoint displacement,
or the trajectory divergence.
"""

import json
import numpy as np
from pathlib import Path

EXP_DIR = Path("outputs/indoor/vggt_tsdf_video_20261031_142303")
DIAG_DIR = Path("outputs/indoor/distortion_investigation_20261031")

perf = json.loads((EXP_DIR / "camera_geometry/vggt_inference_performance.json").read_text())
poses = np.load(EXP_DIR / "camera_geometry/camera_poses.npz")
all_ext = poses["extrinsics"]

# Compute camera centers: C = -R.T @ t
centers = np.array([-all_ext[i, :3, :3].T @ all_ext[i, :3, 3] for i in range(len(all_ext))])

align_logs = perf.get("alignment_logs", [])

shift_vectors = []
shift_magnitudes = []

print("\n" + "=" * 76)
print("  INDEPENDENT VERIFICATION OF TRANSLATION DRIFT QUANTITIES")
print("=" * 76)

for log in align_logs:
    w = log["window"]
    mode = log.get("mode", "")
    s_used = log["scale_sw"]
    if "Regularized" in mode:
        part = mode.split("orig s=")[1].split(",")[0]
        s_orig = float(part)
        delta_s = s_used - s_orig
        
        frame_start = w * 3
        frame_end = min(frame_start + 6, len(all_ext))
        win_centers = centers[frame_start:frame_end]
        mean_c = np.mean(win_centers, axis=0)
        
        # In Umeyama: Y' = s_clamped * R * X + t_orig = mu_Y + (s_clamped - s_orig) * R * mu_X
        # The vector shift introduced at the centroid is: delta_t_vec = (s_clamped - s_orig) * R * mu_X
        # Approximate direction along the line of sight/mean center
        vec_shift = delta_s * mean_c
        mag = float(np.linalg.norm(vec_shift))
        
        shift_vectors.append(vec_shift)
        shift_magnitudes.append(mag)
        print(f"  Window {w:02d}: delta_s={delta_s:+.3f} | shift_vector={vec_shift.round(4).tolist()} | magnitude={mag:.4f} m")

sum_magnitudes = float(np.sum(shift_magnitudes))
net_vector_displacement = np.sum(shift_vectors, axis=0)
net_vector_magnitude = float(np.linalg.norm(net_vector_displacement))

print("\n" + "-" * 76)
print(f"[*] Scalar Sum of Individual Translation-Error Magnitudes: {sum_magnitudes:.4f} meters")
print(f"[*] Net Cumulative Vector Endpoint Displacement:            {net_vector_magnitude:.4f} meters")
print(f"    Net Displacement Vector [X, Y, Z]:                      {net_vector_displacement.round(4).tolist()} meters")
print("-" * 76)

report = {
    "scalar_sum_of_error_magnitudes_m": round(sum_magnitudes, 4),
    "net_cumulative_vector_endpoint_displacement_m": round(net_vector_magnitude, 4),
    "net_displacement_vector_xyz_m": net_vector_displacement.round(4).tolist(),
    "individual_shifts": [
        {"window": align_logs[i]["window"] if i < len(align_logs) else i, "magnitude_m": round(mag, 4), "vector_m": vec.round(4).tolist()}
        for i, (mag, vec) in enumerate(zip(shift_magnitudes, shift_vectors))
    ]
}

(DIAG_DIR / "drift_quantities_verified.json").write_text(json.dumps(report, indent=2))
print(f"[+] Saved verified drift quantities to {DIAG_DIR / 'drift_quantities_verified.json'}")

