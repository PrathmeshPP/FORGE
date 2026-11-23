#!/usr/bin/env python3
"""
Inspect the second half of the video (75s - 179s)
"""
import cv2
from pathlib import Path

MASTER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")
SCRATCH_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE\scratch")
SCRATCH_DIR.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(MASTER_VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)

# Inspect frames at 80s, 100s, 120s, 130s, 140s, 150s, 160s, 170s
for t in [10, 25, 35, 50, 65, 80, 100, 120, 135, 150, 165]:
    fidx = int(round(t * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if ret:
        thumb = cv2.resize(frame, (640, 360))
        out_p = SCRATCH_DIR / f"thumb_{t:03d}s.jpg"
        cv2.imwrite(str(out_p), thumb)
        print(f"Saved {out_p.name} for t={t}s")

cap.release()
