#!/usr/bin/env python3
"""
Inspect source frame crops around Comp #42 projection
"""
import cv2
import numpy as np
from pathlib import Path

V6_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\new_video_reconstructions\whatsapp_video_2026_10_24\best_reconstruction_campaign\final_best_canonical_v6")
NATIVE_DIR = V6_DIR / "native_4k_frames"

frame_files = sorted(list(NATIVE_DIR.glob("*.png")))

# Let's inspect Cam 00, Cam 05, Cam 10, Cam 15
cams_to_check = [0, 5, 10, 15]
coords = {
    0: (1734, 4),
    5: (1545, 41),
    10: (1920, 146),
    15: (1836, 251)
}

for c_idx in cams_to_check:
    fn = frame_files[c_idx]
    im = cv2.imread(str(fn))
    x, y = coords[c_idx]
    # Crop a 100x100 region around (x, y)
    y1 = max(0, y - 50)
    y2 = min(im.shape[0], y + 50)
    x1 = max(0, x - 50)
    x2 = min(im.shape[1], x + 50)
    crop = im[y1:y2, x1:x2]
    mean_bgr = np.mean(crop, axis=(0,1))
    print(f"Cam {c_idx:02d} ({fn.name}): Crop [{y1}:{y2}, {x1}:{x2}] mean BGR={mean_bgr}, RGB=({mean_bgr[2]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[0]:.1f})")
    out_crop_path = Path(r"C:\Users\25beevdt025\Desktop\FORGE\scratch") / f"sky_crop_cam{c_idx:02d}.png"
    out_crop_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_crop_path), crop)
