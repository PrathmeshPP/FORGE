r"""
Evaluate the campus building segment (t=34s to 68s) using OpenCV 5.
Extracts optimal keyframes with strong geometric overlap and parallax diversity.
"""

import sys
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import (
    verify_opencv5_runtime,
    compute_single_frame_metrics,
    compute_pairwise_consistency
)

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")
OUTPUT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24"
FRAMES_DIR = OUTPUT_DIR / "frames"
REPORTS_DIR = OUTPUT_DIR / "reports"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_campus_corridor():
    cv_ver = verify_opencv5_runtime()
    print(f"[*] OpenCV 5 verified: {cv_ver}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.940
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Evaluate corridor from t=34.0s to t=68.0s (34 seconds corridor over the solar campus complex)
    t_start, t_end = 34.0, 68.0

    # Evaluate 1 candidate per ~1.0s (35 candidates)
    sample_timestamps = np.linspace(t_start, t_end, 35)
    candidate_frames = []

    print(f"[*] Scanning {len(sample_timestamps)} candidates between t={t_start}s and t={t_end}s...")
    orb = cv2.ORB_create(nfeatures=1200)

    for idx, t in enumerate(sample_timestamps):
        f_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        metrics, kps, des = compute_single_frame_metrics(
            gray=gray,
            frame_idx=f_idx,
            timestamp_sec=t,
            orb_detector=orb
        )

        candidate_frames.append({
            "candidate_idx": idx,
            "frame_index": f_idx,
            "timestamp_sec": round(t, 3),
            "metrics": metrics,
            "gray": gray,
            "kps": kps,
            "des": des,
            "bgr": frame_bgr
        })

    print(f"[*] Successfully evaluated {len(candidate_frames)} candidates.")

    # Target 8 frames for VGGT-Omega (optimal for memory and accuracy on RTX 4050 / CPU)
    target_k = 8
    bin_size = len(candidate_frames) / target_k
    selected = []

    for b in range(target_k):
        start_i = int(round(b * bin_size))
        end_i = int(round((b + 1) * bin_size))
        end_i = min(len(candidate_frames), max(start_i + 1, end_i))
        bin_candidates = candidate_frames[start_i:end_i]

        # Score candidates by composite_quality_score, sharpness, contrast, and keypoints
        best_cand = max(
            bin_candidates,
            key=lambda c: (
                c["metrics"].composite_quality_score * 0.4 +
                c["metrics"].laplacian_sharpness * 0.3 +
                c["metrics"].orb_keypoint_count * 0.1 +
                c["metrics"].rms_contrast * 0.2
            )
        )
        selected.append(best_cand)

    print(f"\n[*] Selected {len(selected)} high-quality keyframes for 3D reconstruction:")
    selected_dicts = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i, s in enumerate(selected):
        m = s["metrics"]
        print(f"  Frame {i:02d}: idx={s['frame_index']:05d} | t={s['timestamp_sec']:6.2f}s | LapSharp={m.laplacian_sharpness:7.2f} | Tenengrad={m.tenengrad_sharpness:8.1f} | KPs={m.orb_keypoint_count:4d} | Score={m.composite_quality_score:.1f}")

        # Save extracted 4K frame to frames directory
        out_fn = FRAMES_DIR / f"frame_{i:04d}_{s['timestamp_sec']:.3f}s.png"
        cv2.imwrite(str(out_fn), s["bgr"])

        selected_dicts.append({
            "frame_idx": i,
            "original_frame_index": s["frame_index"],
            "timestamp_sec": s["timestamp_sec"],
            "file_path": str(out_fn.resolve()),
            "width": w,
            "height": h,
            "laplacian_sharpness": m.laplacian_sharpness,
            "tenengrad_sharpness": m.tenengrad_sharpness,
            "mean_intensity": m.mean_intensity,
            "rms_contrast": m.rms_contrast,
            "orb_keypoints": m.orb_keypoint_count,
            "grid_occupied_cells": m.grid_occupied_cells,
            "composite_score": m.composite_quality_score
        })

    # Pairwise overlap verification between successive selected keyframes
    pairwise_stats = []
    print("\n[*] Pairwise geometric overlap between selected keyframes:")
    for i in range(len(selected) - 1):
        s1 = selected[i]
        s2 = selected[i + 1]
        pair_report = compute_pairwise_consistency(
            prev_gray=s1["gray"],
            prev_kps=s1["kps"],
            prev_des=s1["des"],
            curr_gray=s2["gray"],
            curr_kps=s2["kps"],
            curr_des=s2["des"],
            prev_idx=s1["frame_index"],
            curr_idx=s2["frame_index"],
            bf_matcher=bf
        )
        print(f"  Pair ({i} -> {i+1}): Matches={pair_report.raw_orb_matches}, Inliers={pair_report.inlier_matches} ({pair_report.inlier_ratio*100:.1f}%), Model={pair_report.geometric_model} | Notes={pair_report.verification_notes}")
        pairwise_stats.append({
            "pair": [i, i + 1],
            "timestamps": [s1["timestamp_sec"], s2["timestamp_sec"]],
            "raw_matches": pair_report.raw_orb_matches,
            "inliers": pair_report.inlier_matches,
            "inlier_ratio": pair_report.inlier_ratio,
            "model": pair_report.geometric_model,
            "status": pair_report.verification_notes
        })

    # Create high-res contact sheet of selected keyframes
    cols = 4
    rows = (len(selected) + cols - 1) // cols
    tile_w, tile_h = 480, 270
    header_h = 70
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + 35) + header_h), (18, 22, 28))
    draw = ImageDraw.Draw(sheet)

    draw.text((25, 18), "F.O.R.G.E — OpenCV 5 Curated Video Keyframes: Campus Solar Complex", fill=(255, 255, 255))
    draw.text((25, 42), f"8 Curated Views | Flight Corridor t=34.0s - 68.0s | Verified OpenCV {cv_ver} Pipeline", fill=(160, 185, 220))

    for idx, s in enumerate(selected):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = header_h + r * (tile_h + 35)

        thumb = cv2.resize(s["bgr"], (tile_w - 6, tile_h), interpolation=cv2.INTER_AREA)
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        sheet.paste(Image.fromarray(thumb_rgb), (x + 3, y))

        m = s["metrics"]
        draw.rectangle([x + 3, y + tile_h, x + tile_w - 3, y + tile_h + 32], fill=(26, 32, 42))
        draw.text((x + 10, y + tile_h + 4), f"Keyframe {idx:02d} | t={s['timestamp_sec']:.2f}s (f={s['frame_index']})", fill=(235, 245, 255))
        draw.text((x + 10, y + tile_h + 18), f"Sharp: {m.laplacian_sharpness:.1f} | KPs: {m.orb_keypoint_count} | Contrast: {m.rms_contrast:.1f}", fill=(170, 195, 220))

    sheet_path = REPORTS_DIR / "selected_keyframes_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\n[*] Saved keyframe contact sheet to: {sheet_path}")

    # Save selection metadata json
    meta_path = REPORTS_DIR / "frame_selection_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "opencv_version": cv_ver,
            "video_path": str(VIDEO_PATH.resolve()),
            "video_resolution": [w, h],
            "video_fps": round(fps, 3),
            "video_duration_sec": round(total_frames / fps, 3),
            "selected_corridor": {
                "start_time_sec": t_start,
                "end_time_sec": t_end,
                "duration_sec": t_end - t_start,
                "num_selected_frames": len(selected)
            },
            "selected_frames": selected_dicts,
            "pairwise_relationships": pairwise_stats
        }, f, indent=2)
    print(f"[*] Saved selection metadata to: {meta_path}")

    cap.release()
    return selected_dicts


if __name__ == "__main__":
    analyze_campus_corridor()
