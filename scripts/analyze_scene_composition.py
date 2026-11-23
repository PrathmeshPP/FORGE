#!/usr/bin/env python3
"""
Inspect the scene content at different times in the video
"""
import cv2
import numpy as np
from pathlib import Path

SCRATCH = Path(r"C:\Users\25beevdt025\Desktop\FORGE\scratch")

for t in [10, 25, 35, 50, 65, 80, 100, 120, 135, 150, 165]:
    im = cv2.imread(str(SCRATCH / f"thumb_{t:03d}s.jpg"))
    # Check green vs brown vs grey content
    # Green (vegetation): G > R and G > B
    # Grey/blue (sky/building): R ~ G ~ B
    b, g, r = im[:,:,0].astype(float), im[:,:,1].astype(float), im[:,:,2].astype(float)
    veg_pct = np.mean((g > r * 1.1) & (g > b * 1.1)) * 100.0
    sky_pct = np.mean((r > 180) & (g > 180) & (b > 180)) * 100.0
    dark_pct = np.mean((r < 50) & (g < 50) & (b < 50)) * 100.0
    print(f"t={t:3d}s: Mean BGR=({im[:,:,0].mean():.1f}, {im[:,:,1].mean():.1f}, {im[:,:,2].mean():.1f}) | Veg={veg_pct:4.1f}% | Sky/Bright={sky_pct:4.1f}% | Dark={dark_pct:4.1f}%")
