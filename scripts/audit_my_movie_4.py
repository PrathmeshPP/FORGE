"""
F.O.R.G.E. — Audit & Quality Inspection of New Video: My movie 4.mp4
Dataset Video: C:\\Users\\25beevdt025\\Downloads\\My movie 4.mp4
Output: outputs/indoor/dataset_my_movie_4/
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\My movie 4.mp4")
BASE_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
OUT_DIR = BASE_DIR / "outputs" / "indoor" / "dataset_my_movie_4"
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUT_DIR / "inspection_samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def compute_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest()

def main():
    log("=" * 76)
    log("  F.O.R.G.E. — NEW VIDEO AUDIT: My movie 4.mp4")
    log("=" * 76)

    t0 = time.perf_counter()
    sha = compute_sha256(VIDEO_PATH)
    log(f"  Source Video: {VIDEO_PATH}")
    log(f"  SHA-256:      {sha}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {VIDEO_PATH}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    log(f"  Resolution:   {w} x {h}")
    log(f"  Frame Rate:   {fps:.2f} FPS")
    log(f"  Total Frames: {total_frames:,}")
    log(f"  Duration:     {duration_sec:.2f} s ({duration_sec / 60:.2f} min)")

    # Sample 40 frames uniformly across the entire 421s duration (approx every 10 seconds)
    sample_indices = np.linspace(0, total_frames - 1, 40, dtype=int)
    records = []
    contact_thumbs = []

    log("\n[*] Sampling 40 frames uniformly across video duration for optical inspection...")
    for rank, idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        t_sec = idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lum = float(np.mean(gray))
        contrast = float(np.std(gray))

        # Save inspection sample
        fname = f"sample_{rank:02d}_f{idx:05d}_{t_sec:05.1f}s.jpg"
        fpath = SAMPLE_DIR / fname
        cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        thumb = cv2.resize(frame, (180, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{t_sec:.1f}s | S:{sharpness:.0f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        contact_thumbs.append(thumb)

        records.append({
            "rank": rank,
            "frame_index": int(idx),
            "timestamp_sec": round(t_sec, 2),
            "sharpness": round(sharpness, 2),
            "luminance": round(lum, 2),
            "contrast": round(contrast, 2),
            "file": fname
        })

    cap.release()

    # Build contact sheet (5 rows x 8 cols = 40 samples)
    rows, cols = 5, 8
    thumb_w, thumb_h = 180, 320
    sheet_w = cols * thumb_w
    sheet_h = rows * thumb_h + 50
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    title = f"F.O.R.G.E. — NEW DATASET AUDIT CONTACT SHEET: My movie 4.mp4 (421s, 1080x1920, 30fps)"
    cv2.putText(sheet, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    for idx, thumb in enumerate(contact_thumbs):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = c_idx * thumb_w
        y0 = 50 + r_idx * thumb_h
        sheet[y0 : y0 + thumb_h, x0 : x0 + thumb_w] = thumb

    contact_path = OUT_DIR / "my_movie_4_audit_contact_sheet.png"
    cv2.imwrite(str(contact_path), sheet)
    cv2.imwrite(str(ARTIFACT_DIR / "my_movie_4_audit_contact_sheet.png"), sheet)
    log(f"[+] Saved audit contact sheet to: {contact_path.name}")

    sharpness_vals = [r["sharpness"] for r in records]
    lum_vals = [r["luminance"] for r in records]

    metadata = {
        "dataset_name": "My movie 4 (Full Indoor Walkthrough Survey)",
        "video_path": str(VIDEO_PATH),
        "sha256": sha,
        "resolution": [w, h],
        "fps": round(fps, 2),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "mean_sharpness": round(float(np.mean(sharpness_vals)), 2),
        "median_sharpness": round(float(np.median(sharpness_vals)), 2),
        "min_sharpness": round(float(np.min(sharpness_vals)), 2),
        "max_sharpness": round(float(np.max(sharpness_vals)), 2),
        "mean_luminance": round(float(np.mean(lum_vals)), 2),
        "samples": records
    }

    meta_path = OUT_DIR / "my_movie_4_audit_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"[+] Saved audit metadata to: {meta_path.name}")
    log(f"[*] Video audit completed in {time.perf_counter() - t0:.2f}s!")

if __name__ == "__main__":
    main()

