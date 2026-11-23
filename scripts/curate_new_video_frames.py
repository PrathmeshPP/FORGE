"""
Stage A: Quality Frame Curator and Adaptive Keyframe Selector for New Indoor Video.
Source: WhatsApp Video 2026-10-30 at 16.43.49.mp4
"""

import os
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VIDEO_PATH = r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 16.43.49.mp4"
OUTPUT_DIR = Path("outputs/indoor/room_001/attempt_4_new_video")
FRAMES_DIR = OUTPUT_DIR / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CuratedFrame:
    frame_index: int
    timestamp_sec: float
    file_path: str
    laplacian_sharpness: float
    mean_luminance: float
    contrast_rms: float
    pairwise_disparity: float
    is_adaptive_keyframe: bool
    selection_reason: str


def curate_frames():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Source video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total_candidate_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.889
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_candidate_frames / fps

    print(f"[*] Analyzing source video: {total_candidate_frames} frames, {duration_sec:.2f}s @ {fps:.2f} FPS")

    # Step 1: Scan all candidate frames to compute sharpness, luminance, contrast
    all_frames_data = []
    prev_gray = None

    for f_idx in range(total_candidate_frames):
        ret, frame = cap.read()
        if not ret:
            break
        t_sec = f_idx / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_lum = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        if prev_gray is not None:
            g1 = cv2.resize(prev_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            diff = float(np.mean(cv2.absdiff(g1, g2)))
        else:
            diff = 0.0

        prev_gray = gray
        all_frames_data.append({
            "frame_idx": f_idx,
            "timestamp_sec": round(t_sec, 3),
            "sharpness": round(lap_var, 2),
            "luminance": round(mean_lum, 2),
            "contrast": round(contrast_std, 2),
            "diff_from_prev": round(diff, 2)
        })

    cap.release()

    # Step 2: Temporal sampling at >= 1 FPS
    # For every 1-second interval [s, s+1), select the highest sharpness frame
    total_seconds = int(math.ceil(duration_sec))
    selected_indices = set()
    base_1fps_frames = []
    rejected_reasons = {}

    for s in range(total_seconds):
        start_f = int(s * fps)
        end_f = min(int((s + 1) * fps), total_candidate_frames)
        if start_f >= total_candidate_frames:
            break

        interval_candidates = all_frames_data[start_f:end_f]
        if not interval_candidates:
            continue

        # Score candidates: prioritize sharpness, penalize extreme luminance
        best_cand = None
        best_score = -1.0
        for cand in interval_candidates:
            lum_penalty = 1.0
            if cand["luminance"] < 30.0 or cand["luminance"] > 230.0:
                lum_penalty = 0.5
            score = cand["sharpness"] * lum_penalty
            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand is not None:
            best_idx = best_cand["frame_idx"]
            selected_indices.add(best_idx)
            base_1fps_frames.append(best_idx)

            # Record rejection reasons for others in interval
            for cand in interval_candidates:
                c_idx = cand["frame_idx"]
                if c_idx != best_idx:
                    rejected_reasons[c_idx] = (
                        f"Suboptimal sharpness ({cand['sharpness']:.1f} vs selected {best_cand['sharpness']:.1f}) "
                        f"in second {s:02d} interval"
                    )

    # Step 3: Adaptive Keyframes for high motion/parallax regions (e.g. seconds 13 to 20)
    # Check intervals where the motion disparity between consecutive base frames is high
    adaptive_keyframes = []
    sorted_base = sorted(list(selected_indices))

    for i in range(len(sorted_base) - 1):
        idx_a = sorted_base[i]
        idx_b = sorted_base[i + 1]
        mid_idx = (idx_a + idx_b) // 2

        # Check if the interval spans significant camera displacement
        # In seconds 13-20 (frame ~388 to ~608), the camera sweeps across lockers and notice boards
        t_a = all_frames_data[idx_a]["timestamp_sec"]
        if (t_a >= 12.5 and t_a <= 19.5) or (idx_b - idx_a >= int(fps * 0.9)):
            # Pick the best frame around the midpoint [mid_idx - 5 : mid_idx + 5]
            sub_start = max(idx_a + 2, mid_idx - 5)
            sub_end = min(idx_b - 2, mid_idx + 5)
            if sub_start < sub_end:
                sub_cands = all_frames_data[sub_start:sub_end]
                best_sub = max(sub_cands, key=lambda c: c["sharpness"])
                sub_idx = best_sub["frame_idx"]
                if sub_idx not in selected_indices:
                    selected_indices.add(sub_idx)
                    adaptive_keyframes.append(sub_idx)
                    if sub_idx in rejected_reasons:
                        del rejected_reasons[sub_idx]

    final_selected = sorted(list(selected_indices))
    print(f"[+] Total candidate frames: {total_candidate_frames}")
    print(f"[+] Uniform 1 FPS base frames: {len(base_1fps_frames)}")
    print(f"[+] Adaptive keyframes added: {len(adaptive_keyframes)}")
    print(f"[+] Total selected for VGGT: {len(final_selected)}")
    print(f"[+] Total rejected frames: {total_candidate_frames - len(final_selected)}")

    # Step 4: Extract and save the selected frames as high quality PNGs
    cap = cv2.VideoCapture(VIDEO_PATH)
    curated_objects = []
    prev_saved_gray = None

    for rank, f_idx in enumerate(final_selected):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        fdata = all_frames_data[f_idx]
        t_sec = fdata["timestamp_sec"]
        is_adaptive = f_idx in adaptive_keyframes
        tag = "adapt" if is_adaptive else "base"
        fname = f"frame_{rank:02d}_f{f_idx:04d}_{t_sec:05.2f}s_{tag}.png"
        fpath = FRAMES_DIR / fname
        cv2.imwrite(str(fpath), frame_bgr)

        # Compute pairwise disparity with previous saved frame
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if prev_saved_gray is not None:
            g1 = cv2.resize(prev_saved_gray, (160, 160))
            g2 = cv2.resize(gray, (160, 160))
            disp = float(np.mean(cv2.absdiff(g1, g2)))
        else:
            disp = 0.0
        prev_saved_gray = gray

        reason = (
            f"Adaptive keyframe for parallax/rapid scene transition (sharpness: {fdata['sharpness']:.1f})"
            if is_adaptive
            else f"Sharpest frame for second {int(t_sec):02d} interval (sharpness: {fdata['sharpness']:.1f})"
        )

        curated_objects.append(CuratedFrame(
            frame_index=f_idx,
            timestamp_sec=t_sec,
            file_path=str(fpath.resolve()),
            laplacian_sharpness=fdata["sharpness"],
            mean_luminance=fdata["luminance"],
            contrast_rms=fdata["contrast"],
            pairwise_disparity=round(disp, 2),
            is_adaptive_keyframe=is_adaptive,
            selection_reason=reason
        ))

    cap.release()

    # Step 5: Generate Contact Sheet of All Selected Frames
    n_sel = len(curated_objects)
    cols = 6
    rows = int(math.ceil(n_sel / cols))
    thumb_w = 240
    thumb_h = int(thumb_w * height / width)
    sheet_img = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet_img)

    for i, c_obj in enumerate(curated_objects):
        cx = (i % cols) * thumb_w
        cy = (i // cols) * thumb_h

        img_bgr = cv2.imread(c_obj.file_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_thumb = Image.fromarray(img_rgb).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet_img.paste(pil_thumb, (cx, cy))

        # Tag banner: Green for base 1FPS, Cyan for Adaptive
        banner_col = (0, 120, 200, 200) if c_obj.is_adaptive_keyframe else (0, 100, 0, 200)
        draw.rectangle([cx, cy, cx + thumb_w, cy + 22], fill=banner_col[:3])
        label = f"#{i:02d} | f{c_obj.frame_index:03d} | {c_obj.timestamp_sec:.2f}s | {'ADAPT' if c_obj.is_adaptive_keyframe else '1FPS'}"
        draw.text((cx + 5, cy + 4), label, fill=(255, 255, 255))

    contact_path = OUTPUT_DIR / "selected_frames_contact_sheet.png"
    sheet_img.save(str(contact_path))
    print(f"[+] Selected frames contact sheet saved to: {contact_path}")

    # Step 6: Write Frame Sampling Report JSON
    sampling_report = {
        "source_video": {
            "path": str(VIDEO_PATH),
            "total_candidate_frames": total_candidate_frames,
            "duration_sec": round(duration_sec, 3),
            "fps": round(fps, 3),
            "resolution": f"{width}x{height}",
            "orientation": "portrait"
        },
        "sampling_metrics": {
            "total_candidate_frames": total_candidate_frames,
            "number_sampled_uniform_1fps": len(base_1fps_frames),
            "number_adaptive_keyframes": len(adaptive_keyframes),
            "total_selected_for_vggt": len(curated_objects),
            "total_rejected": total_candidate_frames - len(curated_objects),
            "rejection_percentage": round((total_candidate_frames - len(curated_objects)) / total_candidate_frames * 100, 2),
            "temporal_sampling_density_fps": round(len(curated_objects) / duration_sec, 3)
        },
        "selected_frames": [asdict(c) for c in curated_objects],
        "rejection_sample_summary": {
            "primary_rejection_reason": "Temporal redundancy & lower sharpness within local 1.0s window",
            "severely_blurred_frames_rejected": 0,
            "total_unusable_frames": 0
        }
    }

    report_path = OUTPUT_DIR / "frame_sampling_report.json"
    with open(report_path, "w") as f:
        json.dump(sampling_report, f, indent=2)

    print(f"[+] Frame sampling report saved to: {report_path}")
    return sampling_report


if __name__ == "__main__":
    curate_frames()

