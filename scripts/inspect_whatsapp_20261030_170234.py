import os
import sys
import hashlib
from pathlib import Path
import cv2
import numpy as np
import json

video_path = Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4")

if not video_path.exists():
    print(f"ERROR: Video does not exist at {video_path}")
    sys.exit(1)

# SHA-256
h = hashlib.sha256()
with open(video_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha256 = h.hexdigest()
file_size_bytes = video_path.stat().st_size
file_size_mb = file_size_bytes / (1024 * 1024)

# OpenCV Video Capture
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"ERROR: Failed to open video {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
duration_sec = total_frames / fps if fps > 0 else 0

print(f"File: {video_path.name}")
print(f"Size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
print(f"SHA-256: {sha256}")
print(f"Resolution: {width}x{height} (Orientation: {'Portrait' if height > width else 'Landscape'})")
print(f"FPS: {fps:.3f}, Total Frames: {total_frames}, Duration: {duration_sec:.2f} s")
print(f"Codec FourCC: {fourcc_str}")

# Check existing videos in workspace to see if SHA matches anything
existing_vids = list(Path("data").rglob("*.mp4")) + list(Path("outputs").rglob("*.mp4"))
print("\nChecking matching SHA256 in workspace:")
for ev in existing_vids:
    if ev.is_file():
        eh = hashlib.sha256()
        with open(ev, "rb") as f:
            while c := f.read(1024 * 1024):
                eh.update(c)
        if eh.hexdigest() == sha256:
            print(f"  MATCH: {ev} has exact same SHA256!")

# Sample frames across video to evaluate sharpness, blur, brightness, visual coverage
sample_step = max(1, total_frames // 100)
sharpness_vals = []
brightness_vals = []
contrast_vals = []

frame_idx = 0
sampled_indices = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % sample_step == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Laplacian variance for sharpness
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        sharpness_vals.append(lap_var)
        brightness_vals.append(mean_val)
        contrast_vals.append(std_val)
        sampled_indices.append(frame_idx)
    frame_idx += 1
cap.release()

sharpness_vals = np.array(sharpness_vals)
brightness_vals = np.array(brightness_vals)
contrast_vals = np.array(contrast_vals)

print(f"\nSampled {len(sharpness_vals)} frames across video:")
print(f"Sharpness (Laplacian Var): Min={np.min(sharpness_vals):.1f}, Mean={np.mean(sharpness_vals):.1f}, Median={np.median(sharpness_vals):.1f}, P10={np.percentile(sharpness_vals, 10):.1f}, Max={np.max(sharpness_vals):.1f}")
print(f"Brightness (Mean Gray): Min={np.min(brightness_vals):.1f}, Mean={np.mean(brightness_vals):.1f}, Median={np.median(brightness_vals):.1f}, P10={np.percentile(brightness_vals, 10):.1f}, Max={np.max(brightness_vals):.1f}")
print(f"Contrast (Std Gray): Min={np.min(contrast_vals):.1f}, Mean={np.mean(contrast_vals):.1f}, Median={np.median(contrast_vals):.1f}")

# Check blurry frames percentage (lap_var < 50)
blur_pct = np.mean(sharpness_vals < 50.0) * 100
print(f"Estimated blurry frames (<50 LapVar): {blur_pct:.1f}%")

# Save inspection report json
out_meta = {
    "file_name": video_path.name,
    "file_size_mb": file_size_mb,
    "sha256": sha256,
    "width": width,
    "height": height,
    "orientation": "portrait" if height > width else "landscape",
    "fps": fps,
    "total_frames": total_frames,
    "duration_sec": duration_sec,
    "codec": fourcc_str,
    "sharpness": {
        "min": float(np.min(sharpness_vals)),
        "mean": float(np.mean(sharpness_vals)),
        "median": float(np.median(sharpness_vals)),
        "p10": float(np.percentile(sharpness_vals, 10)),
        "max": float(np.max(sharpness_vals)),
        "blurry_pct": float(blur_pct)
    },
    "brightness": {
        "min": float(np.min(brightness_vals)),
        "mean": float(np.mean(brightness_vals)),
        "median": float(np.median(brightness_vals)),
        "p10": float(np.percentile(brightness_vals, 10)),
        "max": float(np.max(brightness_vals))
    }
}

inspect_dir = Path("outputs/indoor/vggt_tsdf_video_20261030_170234")
inspect_dir.mkdir(parents=True, exist_ok=True)
with open(inspect_dir / "input_inspection.json", "w") as f:
    json.dump(out_meta, f, indent=2)
print(f"\nSaved metadata to {inspect_dir / 'input_inspection.json'}")

