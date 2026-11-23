"""
Analysis script for the new indoor room video:
WhatsApp Video 2026-10-30 at 16.43.49.mp4
"""

import os
import sys
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 16.43.49.mp4"
OUTPUT_DIR = Path("outputs/indoor/room_001/attempt_4_new_video/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_video():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    aspect_ratio = width / height

    print(f"Video Path: {VIDEO_PATH}")
    print(f"Resolution: {width} x {height}")
    print(f"Aspect Ratio: {aspect_ratio:.3f} ({width}:{height})")
    print(f"Orientation: {'Portrait' if height > width else 'Landscape'}")
    print(f"FPS: {fps:.3f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration: {duration_sec:.2f} s")
    print(f"Codec FourCC: {fourcc}")

    # Analyze motion blur, luminance, contrast, motion across frames
    frame_metrics = []
    prev_gray = None
    motion_disparities = []

    # Sample every frame to have complete metrics
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = f_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        if prev_gray is not None:
            # Pairwise difference on small thumbnail
            g1 = cv2.resize(prev_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            diff = float(np.mean(cv2.absdiff(g1, g2)))
            motion_disparities.append(diff)
        else:
            diff = 0.0

        prev_gray = gray
        frame_metrics.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(t_sec, 3),
            "laplacian_sharpness": round(lap_var, 2),
            "mean_luminance": round(mean_lum, 2),
            "contrast_rms": round(contrast_std, 2),
            "interframe_diff": round(diff, 2)
        })

    cap.release()

    sharpness_vals = [m["laplacian_sharpness"] for m in frame_metrics]
    mean_sharpness = float(np.mean(sharpness_vals))
    median_sharpness = float(np.median(sharpness_vals))
    min_sharpness = float(np.min(sharpness_vals))
    max_sharpness = float(np.max(sharpness_vals))
    q25_sharpness = float(np.percentile(sharpness_vals, 25))
    q75_sharpness = float(np.percentile(sharpness_vals, 75))

    # Severe blur threshold (e.g. < 40 or < 0.3 * median)
    severe_blur_frames = [m for m in frame_metrics if m["laplacian_sharpness"] < 40.0]

    print("\n--- Motion Blur & Quality Analysis ---")
    print(f"Sharpness (Laplacian Var): Mean={mean_sharpness:.1f}, Median={median_sharpness:.1f}, Min={min_sharpness:.1f}, Max={max_sharpness:.1f}")
    print(f"Interquartile Range: Q25={q25_sharpness:.1f}, Q75={q75_sharpness:.1f}")
    print(f"Severely blurred frames (<40.0): {len(severe_blur_frames)} / {len(frame_metrics)} ({len(severe_blur_frames)/len(frame_metrics)*100:.1f}%)")

    # Generate a Contact Sheet of 24 representative frames (6 cols x 4 rows)
    n_contact = 24
    contact_indices = np.linspace(0, total_frames - 1, n_contact, dtype=int)
    cols = 6
    rows = 4
    thumb_w = 240
    thumb_h = int(thumb_w * height / width)  # maintain aspect ratio
    grid_img = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    cap = cv2.VideoCapture(VIDEO_PATH)
    for i, f_idx in enumerate(contact_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret:
            continue
        c_x = (i % cols) * thumb_w
        c_y = (i // cols) * thumb_h

        t_sec = f_idx / fps
        sharpness = frame_metrics[f_idx]["laplacian_sharpness"]

        # Resize and paste
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        grid_img.paste(pil_frame, (c_x, c_y))

        # Overlay text
        label = f"#{f_idx:03d} | {t_sec:.1f}s | Sh:{sharpness:.0f}"
        draw.rectangle([c_x, c_y, c_x + thumb_w, c_y + 22], fill=(0, 0, 0, 180))
        draw.text((c_x + 5, c_y + 3), label, fill=(255, 255, 0))

    cap.release()

    contact_path = OUTPUT_DIR / "new_video_contact_sheet.png"
    grid_img.save(str(contact_path))
    print(f"\n[+] Contact sheet generated and saved to: {contact_path}")

    # Also save individual representative frames at key intervals (e.g. t=0s, 4s, 8s, 12s, 16s, 20s)
    key_t = [0.5, 4.0, 8.0, 12.0, 16.0, 20.0]
    cap = cv2.VideoCapture(VIDEO_PATH)
    rep_frames_info = []
    for t in key_t:
        f_idx = min(int(t * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            fname = f"rep_frame_{t:.1f}s_f{f_idx:03d}.png"
            fpath = OUTPUT_DIR / fname
            cv2.imwrite(str(fpath), frame)
            rep_frames_info.append({
                "time_sec": t,
                "frame_idx": f_idx,
                "file": str(fpath),
                "sharpness": frame_metrics[f_idx]["laplacian_sharpness"]
            })
    cap.release()

    report = {
        "video_path": str(VIDEO_PATH),
        "resolution": {"width": width, "height": height},
        "aspect_ratio": f"{width}:{height} ({aspect_ratio:.3f})",
        "orientation": "portrait",
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration_sec, 3),
        "codec": fourcc,
        "sharpness_statistics": {
            "mean": round(mean_sharpness, 2),
            "median": round(median_sharpness, 2),
            "min": round(min_sharpness, 2),
            "max": round(max_sharpness, 2),
            "q25": round(q25_sharpness, 2),
            "q75": round(q75_sharpness, 2),
            "severe_blur_frames_count": len(severe_blur_frames),
            "severe_blur_ratio": round(len(severe_blur_frames) / len(frame_metrics), 4)
        },
        "contact_sheet_path": str(contact_path),
        "representative_frames": rep_frames_info
    }

    report_json_path = OUTPUT_DIR / "video_analysis_report.json"
    with open(report_json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[+] Video analysis report saved to: {report_json_path}")
    return report

if __name__ == "__main__":
    analyze_video()

