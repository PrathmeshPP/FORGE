"""
F.O.R.G.E. — Quality Frame Curation for My movie 4.mp4
Extracts high-sharpness, diverse viewpoints across the 421s video.
"""

import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\My movie 4.mp4")
BASE_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
OUT_DIR = BASE_DIR / "outputs" / "indoor" / "dataset_my_movie_4_adaptive"
FRAMES_DIR = OUT_DIR / "frames"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def curate_frames(target_count: int = 60, local_search_radius: int = 15):
    log("=" * 76)
    log(f"  F.O.R.G.E. KEYFRAME CURATION: {VIDEO_PATH.name}")
    log(f"  Target keyframe count: {target_count}")
    log("=" * 76)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps

    log(f"  Total video frames: {total_frames:,} ({duration:.1f}s at {fps:.2f} fps)")

    # Evenly space candidate anchor points across video duration
    # Leave 3% margins at start and end for camera stability
    start_f = int(total_frames * 0.03)
    end_f = int(total_frames * 0.97)
    anchor_indices = np.linspace(start_f, end_f, target_count, dtype=int)

    log(f"  Searching locally (+/-{local_search_radius} frames) around {target_count} anchors for maximum optical sharpness...")
    curated = []

    for rank, anchor in enumerate(anchor_indices):
        best_frame = None
        best_sharp = -1.0
        best_idx = anchor
        best_lum = 0.0
        best_contrast = 0.0

        for offset in range(-local_search_radius, local_search_radius + 1, 2):
            idx = int(np.clip(anchor + offset, 0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (270, 480))
            sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
            lum = float(np.mean(small))
            contrast = float(np.std(small))

            # Quality score: prioritize sharp frames with balanced lighting (lum between 50 and 200)
            score = sharp
            if lum < 40 or lum > 220:
                score *= 0.5

            if score > best_sharp or best_frame is None:
                best_sharp = score
                best_idx = idx
                best_frame = frame.copy()
                best_lum = lum
                best_contrast = contrast

        t_sec = best_idx / fps
        fname = f"frame_{rank:02d}_f{best_idx:05d}_{t_sec:05.1f}s.png"
        fpath = FRAMES_DIR / fname
        cv2.imwrite(str(fpath), best_frame)

        curated.append({
            "rank": rank,
            "anchor_index": int(anchor),
            "frame_index": int(best_idx),
            "timestamp_sec": round(t_sec, 2),
            "sharpness": round(best_sharp, 2),
            "luminance": round(best_lum, 2),
            "contrast": round(best_contrast, 2),
            "file": fname,
            "file_path": str(fpath.resolve())
        })

        if (rank + 1) % 10 == 0 or rank == target_count - 1:
            log(f"    Extracted {rank + 1}/{target_count} keyframes (latest: t={t_sec:.1f}s, sharp={best_sharp:.0f})")

    cap.release()

    report = {
        "video_path": str(VIDEO_PATH),
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_sec": round(duration, 2),
        "target_count": target_count,
        "mean_sharpness": round(float(np.mean([r["sharpness"] for r in curated])), 2),
        "frames": curated
    }

    report_path = OUT_DIR / "frame_sampling_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log(f"[+] Frame curation complete! Saved {len(curated)} high-sharpness frames to {FRAMES_DIR.name}")
    log(f"    Mean Sharpness: {report['mean_sharpness']:.1f}")
    return curated

if __name__ == "__main__":
    curate_frames(target_count=60, local_search_radius=15)

