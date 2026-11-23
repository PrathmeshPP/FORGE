#!/usr/bin/env python3
"""
Inspect the nadir frame (frame_0007_49.973s.png) and overlay the roof mesh and Comp #42
"""
import cv2
import numpy as np
import trimesh
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
CAMPAIGN_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign"
V6_DIR = CAMPAIGN_DIR / "final_best_canonical_v6"
NATIVE_DIR = V6_DIR / "native_4k_frames"
PRED_DIR = CAMPAIGN_DIR / "upstream_16_frames/predictions"
V8_DIR = CAMPAIGN_DIR / "final_best_canonical_v8"

mesh = trimesh.load(str(V8_DIR / "final_best_canonical_v8_mesh.ply"), process=False)
components = mesh.split(only_watertight=False)
c0 = components[0]
c42 = components[42]

poses = np.load(PRED_DIR / "camera_poses.npz")
ext_cv = poses["extrinsics"]
intr = poses["intrinsics"]
sx = 3840.0 / 688.0
sy = 2160.0 / 384.0
intr_4k = intr[7].copy()
intr_4k[0, :] *= sx
intr_4k[1, :] *= sy

T_CV_TO_GLTF = np.diag([1.0, -1.0, -1.0])

# Load Nadir frame (Frame 7)
im7 = cv2.imread(str(NATIVE_DIR / "frame_0007_49.973s.png"))
h, w, _ = im7.shape
print(f"Frame 7 shape: {w}x{h}")

# Project c42 vertices onto Frame 7
R = ext_cv[7, :3, :3]
t = ext_cv[7, :3, 3]

def project_points(pts_canonical):
    p_cv0 = (T_CV_TO_GLTF @ pts_canonical.T).T
    p_cam = (R @ p_cv0.T).T + t
    valid = p_cam[:, 2] > 0.1
    u = (intr_4k[0, 0] * p_cam[:, 0] / p_cam[:, 2]) + intr_4k[0, 2]
    v = (intr_4k[1, 1] * p_cam[:, 1] / p_cam[:, 2]) + intr_4k[1, 2]
    return u, v, p_cam[:, 2], valid

u42, v42, z42, val42 = project_points(c42.vertices)
print(f"Comp #42 in Frame 7: u in [{u42.min():.1f}, {u42.max():.1f}], v in [{v42.min():.1f}, {v42.max():.1f}]")

# Let's project vertices of c0 with Y > 0.05 (the real roof and building)
roof_verts = c0.vertices[c0.vertices[:, 1] > 0.05]
ur, vr, zr, valr = project_points(roof_verts)
print(f"Real Roof in Frame 7: u in [{ur.min():.1f}, {ur.max():.1f}], v in [{vr.min():.1f}, {vr.max():.1f}]")

# Crop the region of Frame 7 around the roof and Comp #42
x1 = max(0, int(min(u42.min(), ur.min()) - 100))
x2 = min(w, int(max(u42.max(), ur.max()) + 100))
y1 = max(0, int(min(v42.min(), vr.min()) - 100))
y2 = min(h, int(max(v42.max(), vr.max()) + 100))

crop = im7[y1:y2, x1:x2].copy()
out_dir = PROJECT_ROOT / "scratch"
out_dir.mkdir(exist_ok=True)
cv2.imwrite(str(out_dir / "nadir_roof_clean.png"), crop)

# Draw Comp #42 outline on crop
crop_annotated = crop.copy()
for u, v in zip(u42[::10], v42[::10]):
    cu = int(u - x1)
    cv_ = int(v - y1)
    if 0 <= cu < crop.shape[1] and 0 <= cv_ < crop.shape[0]:
        cv2.circle(crop_annotated, (cu, cv_), 2, (0, 0, 255), -1)

cv2.imwrite(str(out_dir / "nadir_roof_annotated_c42.png"), crop_annotated)
print(f"Saved crops to {out_dir}: nadir_roof_clean.png and nadir_roof_annotated_c42.png")
