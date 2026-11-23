#!/usr/bin/env python3
"""
Inspect the full video sample manifest and analyze scene segments.
"""
import json
import cv2
import numpy as np
from pathlib import Path

MANIFEST_PATH = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\new_video_reconstructions\whatsapp_video_2026_10_24\best_reconstruction_campaign\final_best_canonical_v10\diagnostics\full_video_samples_manifest.json")
MASTER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")

with open(MANIFEST_PATH) as f:
    data = json.load(f)

samples = data["samples"]
print(f"Total samples: {len(samples)}")

# Let's inspect intervals: 0-30s, 30-70s, 70-120s, 120-180s
# For a few key timestamps, let's read the image, check average color and dominant features
cap = cv2.VideoCapture(str(MASTER_VIDEO))
fps = data["fps"]

# Check timestamps every 10s: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170
test_times = list(range(0, 180, 10))

print("\nTimestamp Survey across Full 3-Minute Video:")
print(f"{'Time':<8} {'LapVar':<8} {'Mean RGB':<22} {'Description'}")
print("-" * 75)

for t in test_times:
    fidx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_bgr = frame.mean(axis=(0,1))
    mean_rgb = f"({mean_bgr[2]:.1f}, {mean_bgr[1]:.1f}, {mean_bgr[0]:.1f})"
    print(f"{t:4.0f}s     {lap:6.1f}   {mean_rgb:<22}")

cap.release()
