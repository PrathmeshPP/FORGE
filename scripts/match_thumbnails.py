#!/usr/bin/env python3
"""
Inspect what is inside thumbnails
"""
import cv2
import numpy as np
from pathlib import Path

SCRATCH = Path(r"C:\Users\25beevdt025\Desktop\FORGE\scratch")

# Compare feature matching between thumb_050s (the building) and the other thumbs
im50 = cv2.imread(str(SCRATCH / "thumb_050s.jpg"))
orb = cv2.ORB_create(1000)
kp50, des50 = orb.detectAndCompute(im50, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("ORB Feature Matches against Target Building (t=50s):")
print("-" * 55)
for t in [10, 25, 35, 50, 65, 80, 100, 120, 135, 150, 165]:
    im = cv2.imread(str(SCRATCH / f"thumb_{t:03d}s.jpg"))
    kp, des = orb.detectAndCompute(im, None)
    if des is not None and des50 is not None:
        matches = bf.match(des50, des)
        # Filter good matches
        good = [m for m in matches if m.distance < 50]
        print(f"t={t:3d}s: {len(good):4d} good matches (total matches: {len(matches):4d})")
    else:
        print(f"t={t:3d}s: No descriptors")
