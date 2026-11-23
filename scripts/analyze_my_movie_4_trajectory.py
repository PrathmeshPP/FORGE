"""
F.O.R.G.E. — Analyze My movie 4.mp4 Coverage & Trajectory
"""

import cv2
import numpy as np
import json
from pathlib import Path

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\My movie 4.mp4")
cap = cv2.VideoCapture(str(VIDEO_PATH))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
duration = total_frames / fps

print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")

# Sample 1 frame every 2 seconds (~210 frames)
step = int(fps * 2.0)
sampled_records = []

prev_gray = None
for f_idx in range(0, total_frames, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        break
    t_sec = f_idx / fps
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (270, 480))
    sharpness = float(cv2.Laplacian(small_gray, cv2.CV_64F).var())
    lum = float(np.mean(small_gray))
    
    flow_mag = 0.0
    if prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_mag = float(np.mean(np.linalg.norm(flow, axis=-1)))
    prev_gray = small_gray

    sampled_records.append({
        "frame_idx": f_idx,
        "t_sec": round(t_sec, 2),
        "sharpness": round(sharpness, 2),
        "luminance": round(lum, 2),
        "optical_flow": round(flow_mag, 2)
    })

cap.release()

# Find sharpest segments and motion peaks
sharp_arr = [r["sharpness"] for r in sampled_records]
flow_arr = [r["optical_flow"] for r in sampled_records]
print(f"Analyzed {len(sampled_records)} points across {duration:.1f}s:")
print(f"  Sharpness mean: {np.mean(sharp_arr):.1f}, max: {np.max(sharp_arr):.1f}, min: {np.min(sharp_arr):.1f}")
print(f"  Optical flow mean: {np.mean(flow_arr):.2f}, max: {np.max(flow_arr):.2f}")

out_path = Path("outputs/indoor/dataset_my_movie_4/trajectory_sampling.json")
with open(out_path, "w") as f:
    json.dump(sampled_records, f, indent=2)
print("Saved trajectory sampling to:", out_path)

