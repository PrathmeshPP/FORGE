"""
F.O.R.G.E — Stage A: Indoor Room Video Inspection & Quality Diagnostics
"""
import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

VIDEO_PATH = Path("data/source_videos/indoor_room.mp4")
OUTPUT_DIR = Path("outputs/indoor/room_001/inspection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"=== Inspecting Indoor Room Video: {VIDEO_PATH} ===")
    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"Native Resolution: {width}x{height} (Aspect ratio: {width/height:.3f})")
    print(f"Frame Rate: {fps:.2f} fps")
    print(f"Total Frames: {total_frames}")
    print(f"Total Duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes)")

    # Sample frames across the video and compute sharpness, brightness, motion
    sample_step = max(1, int(fps / 2))  # sample ~2 frames per second
    frame_indices = list(range(0, total_frames, sample_step))
    print(f"Sampling {len(frame_indices)} probe frames across the video (step={sample_step})...")

    records = []
    prev_gray = None
    saved_samples_dir = OUTPUT_DIR / "sample_thumbs"
    saved_samples_dir.mkdir(exist_ok=True)

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        std_lum = float(np.std(gray))

        # Optical flow / frame difference if prev exists
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = float(np.mean(diff))
        else:
            motion_score = 0.0
        prev_gray = gray

        records.append({
            "probe_idx": i,
            "frame_idx": idx,
            "timestamp_sec": round(idx / fps, 3),
            "laplacian_sharpness": round(lap_var, 2),
            "mean_luminance": round(mean_lum, 2),
            "std_luminance": round(std_lum, 2),
            "motion_score": round(motion_score, 2)
        })

        # Save thumbnail every 10 samples
        if i % 10 == 0:
            thumb = cv2.resize(frame, (240, int(240 * height / width)))
            cv2.imwrite(str(saved_samples_dir / f"sample_{i:03d}_frame_{idx:05d}.jpg"), thumb)

    cap.release()

    sharpness_vals = [r["laplacian_sharpness"] for r in records]
    motion_vals = [r["motion_score"] for r in records]

    summary = {
        "video_path": str(VIDEO_PATH),
        "resolution": {"width": width, "height": height},
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "probe_samples_count": len(records),
        "sharpness": {
            "mean": round(float(np.mean(sharpness_vals)), 2),
            "std": round(float(np.std(sharpness_vals)), 2),
            "min": round(float(np.min(sharpness_vals)), 2),
            "max": round(float(np.max(sharpness_vals)), 2),
            "p25": round(float(np.percentile(sharpness_vals, 25)), 2),
            "median": round(float(np.median(sharpness_vals)), 2),
            "p75": round(float(np.percentile(sharpness_vals, 75)), 2),
        },
        "motion": {
            "mean": round(float(np.mean(motion_vals)), 2),
            "median": round(float(np.median(motion_vals)), 2),
            "max": round(float(np.max(motion_vals)), 2)
        }
    }

    with open(OUTPUT_DIR / "video_metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "probe_records.json", "w") as f:
        json.dump(records, f, indent=2)

    print("\n--- Summary ---")
    print(json.dumps(summary, indent=2))
    print(f"\nInspection results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

