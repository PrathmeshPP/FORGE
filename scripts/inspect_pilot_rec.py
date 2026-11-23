#!/usr/bin/env python3
"""
Inspect the pilot sparse reconstruction and camera poses from PyCOLMAP
"""
import pycolmap
import numpy as np
import open3d as o3d
from pathlib import Path

PILOT_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\new_video_reconstructions\whatsapp_video_2026_10_24\best_reconstruction_campaign\final_best_canonical_v10\pilot_sfm")
REC_DIR = PILOT_DIR / "sparse/0"

rec = pycolmap.Reconstruction(str(REC_DIR))
print(f"Loaded Reconstruction from {REC_DIR}:")
print(f"  Num Cameras: {len(rec.cameras)}")
for cam_id, cam in rec.cameras.items():
    print(f"    Camera #{cam_id}: model={cam.model_name}, params={cam.params}, size={cam.width}x{cam.height}")

print(f"  Num Images: {len(rec.images)}")
cam_positions = []
for img_id, img in rec.images.items():
    # camera center in world coordinates: C = -R^T @ t
    # In PyCOLMAP, img.cam_from_world transforms world to cam: R = cam_from_world.rotation.matrix(), t = cam_from_world.translation
    cfw = img.cam_from_world()
    R = cfw.rotation.matrix()
    t = cfw.translation
    C = -R.T @ t
    cam_positions.append((img.name, C))
    print(f"    Image '{img.name}': C=({C[0]:.3f}, {C[1]:.3f}, {C[2]:.3f})")

cam_positions = sorted(cam_positions, key=lambda x: x[0])
pts_3d = np.array([p.xyz for p in rec.points3D.values()])
print(f"  3D Points Bounds:")
print(f"    X: [{pts_3d[:,0].min():.3f}, {pts_3d[:,0].max():.3f}]")
print(f"    Y: [{pts_3d[:,1].min():.3f}, {pts_3d[:,1].max():.3f}]")
print(f"    Z: [{pts_3d[:,2].min():.3f}, {pts_3d[:,2].max():.3f}]")
print(f"    Extent: {np.ptp(pts_3d, axis=0).round(3)}")
