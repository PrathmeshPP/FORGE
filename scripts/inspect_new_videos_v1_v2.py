"""
Inspects and analyzes Video 1 and Video 2 for indoor 3D reconstruction suitability.
Generates contact sheets, evaluates motion blur, luminance, contrast, and camera motion.
"""

import os
import sys
import json
import math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

VIDEOS = [
    {
        "id": "video_1",
        "name": "WhatsApp Video 2026-10-30 at 17.02.34.mp4",
        "path": r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4",
        "out_dir": Path("outputs/indoor/video_1_170234/analysis")
    },
    {
        "id": "video_2",
        "name": "WhatsApp Video 2026-10-30 at 17.05.00.mp4",
        "path": r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.05.00.mp4",
        "out_dir": Path("outputs/indoor/video_2_170500/analysis")
    }
]

def analyze_video(vid_info):
    path = vid_info["path"]
    out_dir = vid_info["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    file_size = os.path.getsize(path)

    print(f"\n=======================================================")
    print(f"Analyzing {vid_info['id']}: {vid_info['name']}")
    print(f"Resolution: {w}x{h} ({'Portrait' if h > w else 'Landscape'})")
    print(f"Duration: {duration:.2f}s | FPS: {fps:.3f} | Total Frames: {total_frames}")
    print(f"File Size: {file_size / (1024**2):.2f} MB")

    # Sample every 5 frames for detailed metric profiling
    stride = 5
    sharpness_list = []
    lum_list = []
    contrast_list = []
    prev_gray = None
    motion_diffs = []

    for f_idx in range(0, total_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        rms_cont = float(np.std(gray))

        sharpness_list.append(lap_var)
        lum_list.append(mean_lum)
        contrast_list.append(rms_cont)

        if prev_gray is not None:
            g1 = cv2.resize(prev_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            motion_diffs.append(float(np.mean(cv2.absdiff(g1, g2))))
        prev_gray = gray

    # Contact sheet: 24 representative frames (4 rows x 6 cols)
    n_contact = 24
    contact_indices = np.linspace(0, total_frames - 1, n_contact, dtype=int)
    cols = 6
    rows = 4
    thumb_w = 240
    thumb_h = int(thumb_w * h / w)
    grid_img = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 20, 20))
    draw = ImageDraw.Draw(grid_img)

    for i, f_no in enumerate(contact_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_no))
        ret, frame = cap.read()
        if not ret:
            continue
        cx = (i % cols) * thumb_w
        cy = (i // cols) * thumb_h

        t_sec = f_no / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sh = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        grid_img.paste(pil_frame, (cx, cy))

        label = f"#{f_no:04d} | {t_sec:4.1f}s | Sh:{sh:.0f}"
        draw.rectangle([cx, cy, cx + thumb_w, cy + 22], fill=(0, 0, 0, 180))
        draw.text((cx + 5, cy + 3), label, fill=(255, 255, 0))

    cap.release()

    contact_path = out_dir / "contact_sheet.png"
    grid_img.save(str(contact_path))
    print(f"Saved Contact Sheet: {contact_path}")

    # Metrics
    mean_sh = float(np.mean(sharpness_list))
    med_sh = float(np.median(sharpness_list))
    min_sh = float(np.min(sharpness_list))
    max_sh = float(np.max(sharpness_list))
    blur_count = sum(1 for s in sharpness_list if s < 50.0)

    report = {
        "video_id": vid_info["id"],
        "filename": vid_info["name"],
        "path": path,
        "width": w,
        "height": h,
        "aspect_ratio": f"{w}:{h} ({w/h:.3f})",
        "orientation": "portrait" if h > w else "landscape",
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "duration_sec": round(duration, 3),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024**2), 2),
        "sharpness": {
            "mean": round(mean_sh, 2),
            "median": round(med_sh, 2),
            "min": round(min_sh, 2),
            "max": round(max_sh, 2),
            "severely_blurred_subsample_ratio": round(blur_count / len(sharpness_list), 4)
        },
        "mean_luminance": round(float(np.mean(lum_list)), 2),
        "mean_contrast": round(float(np.mean(contrast_list)), 2),
        "mean_motion_diff": round(float(np.mean(motion_diffs)), 2) if motion_diffs else 0.0,
        "contact_sheet": str(contact_path)
    }

    report_file = out_dir / "inspection_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Inspection Report: {report_file}")
    return report

def main():
    for vid in VIDEOS:
        analyze_video(vid)

if __name__ == "__main__":
    main()

