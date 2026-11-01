"""
Video Inspection and Audit for New Dataset:
WhatsApp Video 2026-11-01 at 18.05.30.mp4
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-11-01 at 18.05.30.mp4")
EXP_DIR = Path("outputs/indoor/texture_experiment_video_20261101_180530")
EXP_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\25beevdt025\.gemini\antigravity\brain\536a497a-fba6-496d-b4a9-b4f0cf756953")

def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    file_size_bytes = VIDEO_PATH.stat().st_size
    h = hashlib.sha256()
    with open(VIDEO_PATH, 'rb') as f:
        while chunk := f.read(1024*1024*4):
            h.update(chunk)
    sha256_hash = h.hexdigest()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[*] File Size: {file_size_bytes} bytes ({file_size_bytes/1024/1024:.2f} MB)")
    print(f"[*] SHA-256: {sha256_hash}")
    print(f"[*] Resolution: {width} x {height} ({'Portrait' if height > width else 'Landscape'})")
    print(f"[*] FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f} s")
    print(f"[*] FourCC: {fourcc}")

    frame_metrics = []
    prev_gray = None
    motion_values = []
    sampled_frames_for_sheet = []
    sheet_sample_indices = np.linspace(0, total_frames - 1, 30, dtype=int)

    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        std_contrast = float(np.std(gray))

        motion = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion = float(np.mean(diff))
            motion_values.append(motion)
        prev_gray = gray

        metric = {
            "index": f_idx,
            "timestamp": round(f_idx / fps, 3),
            "sharpness": round(lap_var, 2),
            "luminance": round(mean_lum, 2),
            "contrast": round(std_contrast, 2),
            "motion": round(motion, 2)
        }
        frame_metrics.append(metric)

        if f_idx in sheet_sample_indices:
            sampled_frames_for_sheet.append((f_idx, f_idx / fps, lap_var, frame.copy()))

        f_idx += 1

    cap.release()

    sharpness_vals = [m["sharpness"] for m in frame_metrics]
    lum_vals = [m["luminance"] for m in frame_metrics]
    contrast_vals = [m["contrast"] for m in frame_metrics]

    avg_sharpness = float(np.mean(sharpness_vals))
    max_sharpness = float(np.max(sharpness_vals))
    min_sharpness = float(np.min(sharpness_vals))
    avg_lum = float(np.mean(lum_vals))
    avg_contrast = float(np.mean(contrast_vals))
    avg_motion = float(np.mean(motion_values)) if motion_values else 0.0

    print(f"[+] Sharpness (Laplacian Var): Mean={avg_sharpness:.2f}, Min={min_sharpness:.2f}, Max={max_sharpness:.2f}")
    print(f"[+] Luminance: Mean={avg_lum:.2f}, Contrast Std: Mean={avg_contrast:.2f}")
    print(f"[+] Inter-frame Motion: Mean={avg_motion:.2f}")

    # Build Contact Sheet (5 rows x 6 cols = 30 tiles)
    cols = 6
    rows = 5
    tile_w = 280
    tile_h = int(tile_w * (height / width))
    header_h = 28

    canvas_w = cols * tile_w
    canvas_h = rows * (tile_h + header_h) + 60
    sheet = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Master title banner
    title_text = f"NEW DATASET AUDIT: WhatsApp Video 2026-11-01 at 18.05.30.mp4 ({duration_sec:.1f}s, {width}x{height}, {fps:.1f}fps, {total_frames} frames)"
    cv2.putText(sheet, title_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    for i, (idx, t_sec, sharp, f_img) in enumerate(sampled_frames_for_sheet[:30]):
        r = i // cols
        c = i % cols
        x0 = c * tile_w
        y0 = 60 + r * (tile_h + header_h)

        # Header bar
        header_bar = np.zeros((header_h, tile_w, 3), dtype=np.uint8)
        color = (0, 255, 0) if sharp > avg_sharpness else (0, 165, 255)
        cv2.putText(header_bar, f"#{idx} | {t_sec:.1f}s | Sh:{sharp:.0f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        resized = cv2.resize(f_img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        sheet[y0 : y0 + header_h, x0 : x0 + tile_w] = header_bar
        sheet[y0 + header_h : y0 + header_h + tile_h, x0 : x0 + tile_w] = resized

    out_sheet_path = EXP_DIR / "video_audit_contact_sheet.png"
    cv2.imwrite(str(out_sheet_path), sheet)
    print(f"[+] Saved contact sheet: {out_sheet_path}")

    # Copy to artifact dir
    artifact_sheet_path = ARTIFACT_DIR / "nov1_180530_contact_sheet.png"
    cv2.imwrite(str(artifact_sheet_path), sheet)
    print(f"[+] Saved artifact contact sheet: {artifact_sheet_path}")

    # Save summary metadata
    meta = {
        "video_path": str(VIDEO_PATH),
        "file_size_bytes": file_size_bytes,
        "sha256": sha256_hash,
        "resolution": [width, height],
        "aspect_ratio": round(width / height, 3),
        "orientation": "portrait" if height > width else "landscape",
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 2),
        "fourcc": fourcc,
        "sharpness": {
            "mean": round(avg_sharpness, 2),
            "min": round(min_sharpness, 2),
            "max": round(max_sharpness, 2)
        },
        "luminance_mean": round(avg_lum, 2),
        "contrast_mean": round(avg_contrast, 2),
        "motion_mean": round(avg_motion, 2)
    }

    out_meta_path = EXP_DIR / "video_audit_metadata.json"
    with open(out_meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[+] Saved metadata: {out_meta_path}")

if __name__ == "__main__":
    main()
