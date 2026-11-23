r"""
Inspect DJI video details: MP4 metadata, visual timeline, and frame characteristics.
"""

import sys
import os
import json
import struct
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")
OUTPUT_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\new_video_reconstructions\whatsapp_video_2026_10_24")
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_mp4_atoms(file_path: Path):
    """Scan top-level and moov atoms for DJI metadata markers."""
    findings = {}
    with open(file_path, "rb") as f:
        file_size = file_path.stat().st_size
        offset = 0
        while offset < file_size:
            f.seek(offset)
            header = f.read(8)
            if len(header) < 8:
                break
            size, name = struct.unpack(">I4s", header)
            name = name.decode("latin1", errors="replace")
            if size == 1: # 64-bit size
                size = struct.unpack(">Q", f.read(8))[0]
            elif size == 0:
                size = file_size - offset
            findings[f"{name}@{offset}"] = size
            if name in ("moov", "udta"):
                # Can inspect child boxes
                pass
            offset += size
            if size <= 0:
                break
    return findings


def analyze_timeline(video_path: Path, num_samples: int = 16):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"Video: {video_path.name}")
    print(f"Resolution: {w}x{h}, FPS: {fps:.3f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    orb = cv2.ORB_create(nfeatures=1000)

    samples = []
    tile_w, tile_h = 320, 180
    cols = 4
    rows = (num_samples + cols - 1) // cols
    header_h = 60
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 30) + header_h), (20, 24, 30))
    draw = ImageDraw.Draw(sheet)

    draw.text((20, 15), f"DJI Drone Video Timeline Sampling — {video_path.name}", fill=(255, 255, 255))
    draw.text((20, 35), f"Resolution: {w}x{h} (4K) | FPS: {fps:.2f} | Duration: {duration:.2f}s ({num_samples} key points)", fill=(160, 180, 210))

    for idx, f_idx in enumerate(sample_indices):
        t_sec = f_idx / fps
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            print(f"Failed to read frame at index {f_idx}")
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_bright = float(np.mean(gray))
        kps = orb.detect(gray, None)
        num_kps = len(kps)

        samples.append({
            "sample_id": idx,
            "frame_index": int(f_idx),
            "timestamp_sec": round(t_sec, 3),
            "laplacian_var": round(lap_var, 2),
            "mean_brightness": round(mean_bright, 2),
            "orb_keypoints": num_kps,
        })
        print(f"  Sample {idx:02d} | f={f_idx:05d} | t={t_sec:6.2f}s | LapVar={lap_var:7.2f} | Brightness={mean_bright:5.1f} | ORB={num_kps:4d}")

        # Tile to sheet
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = header_h + r * (tile_h + 30)
        thumb = cv2.resize(frame_bgr, (tile_w - 4, tile_h), interpolation=cv2.INTER_AREA)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(thumb_rgb), (x + 2, y))

        draw.rectangle([x + 2, y + tile_h, x + tile_w - 2, y + tile_h + 28], fill=(28, 34, 44))
        draw.text((x + 8, y + tile_h + 4), f"S{idx:02d} | t={t_sec:.1f}s (f={f_idx})", fill=(240, 240, 255))
        draw.text((x + 8, y + tile_h + 16), f"Sharp: {lap_var:.1f} | ORB: {num_kps}", fill=(180, 205, 230))

    cap.release()

    sheet_path = REPORTS_DIR / "timeline_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"Saved timeline sheet: {sheet_path}")

    metrics_path = REPORTS_DIR / "timeline_samples.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    return samples


if __name__ == "__main__":
    atoms = parse_mp4_atoms(VIDEO_PATH)
    print("MP4 Atoms:", atoms)
    samples = analyze_timeline(VIDEO_PATH, num_samples=16)
