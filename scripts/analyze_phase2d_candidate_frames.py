"""F.O.R.G.E — Phase 2D: OpenCV 5 Candidate Frame Analysis and Contact Sheet Generation.

Analyzes source videos using OpenCV 5 to find candidate frames that add useful information
about missing regions for Campus 0049 and River/Bridge:
- Campus: Courtyard, ground sections, building roofs/facades, vegetation margins, outer boundary.
- River/Bridge: Terrain, riverbanks, bridge deck/piers, road continuity, corridor perimeter.

Outputs:
- outputs/opencv5_experiments/phase2d/campus_0049/candidate_frames_contact_sheet.png
- outputs/opencv5_experiments/phase2d/river_bridge/candidate_frames_contact_sheet.png
- Detailed frame selection logs and JSON metadata.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer

OUT_BASE = PROJECT_ROOT / "outputs/opencv5_experiments/phase2d"

CAMPUS_VIDEO = PROJECT_ROOT / "data/source_videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4"
if not CAMPUS_VIDEO.exists():
    CAMPUS_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20260727_172754_0049_1785175051818_video.mp4")

RIVER_VIDEO = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20260727_172812_0050_1785175896547_video.mp4")
if not RIVER_VIDEO.exists():
    RIVER_VIDEO = PROJECT_ROOT / "data/source_videos/segment_52s_64s.mp4"


def build_candidate_contact_sheet(
    frames_info: list,
    out_path: Path,
    scene_title: str
):
    """Builds a labeled contact sheet with metrics, timestamps, and selection rationale."""
    N = len(frames_info)
    cols = min(4, N)
    rows = int(np.ceil(N / cols))

    cell_w, cell_h = 480, 270
    card_h = cell_h + 80
    header_h = 70

    canvas_w = cell_w * cols
    canvas_h = header_h + rows * card_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 20, 26))
    draw = ImageDraw.Draw(canvas)

    draw.text((25, 14), f"F.O.R.G.E Phase 2D — Candidate Frame Analysis: {scene_title.upper()}", fill=(255, 255, 255))
    draw.text((25, 38), "OpenCV 5 Quality, Viewpoint Diversity & Coverage Recovery Frame Curation", fill=(160, 200, 240))

    for idx, f_meta in enumerate(frames_info):
        r = idx // cols
        c = idx % cols
        x = c * cell_w
        y = header_h + r * card_h

        img_p = Path(f_meta["file_path"])
        if img_p.is_file():
            im = Image.open(img_p).resize((cell_w, cell_h), Image.Resampling.LANCZOS)
        else:
            im = Image.new("RGB", (cell_w, cell_h), (30, 30, 40))

        canvas.paste(im, (x, y))

        # Bottom info card
        card_rect = [x, y + cell_h, x + cell_w, y + card_h]
        draw.rectangle(card_rect, fill=(22, 28, 38))
        draw.line([x, y + cell_h, x + cell_w, y + cell_h], fill=(45, 55, 75), width=2)

        t_sec = f_meta["timestamp_sec"]
        fn = f_meta["frame_index"]
        reason = f_meta["selection_reason"]
        q_score = f_meta["metrics"]["composite_quality_score"]

        draw.text((x + 12, y + cell_h + 8), f"#{idx+1} | t={t_sec:.3f}s (Frame {fn}) | Q-Score: {q_score:.3f}", fill=(0, 255, 200))
        draw.text((x + 12, y + cell_h + 30), f"Sharpness: {f_meta['metrics']['laplacian_sharpness']:.1f} | RMS Cont: {f_meta['metrics']['rms_contrast']:.1f}", fill=(200, 215, 230))
        draw.text((x + 12, y + cell_h + 52), f"Reason: {reason[:55]}", fill=(255, 200, 100))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[+] Saved contact sheet: {out_path}")


def analyze_campus():
    print("\n" + "=" * 70)
    print("ANALYZING CAMPUS 0049 CANDIDATE FRAMES")
    print("=" * 70)
    out_dir = OUT_BASE / "campus_0049"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = OpenCV5FrameAnalyzer()
    
    # 8 candidate frames spanning the 16.12s flight
    extracted_records, report = analyzer.extract_curated_frames(
        video_path=str(CAMPUS_VIDEO),
        output_dir=str(out_dir / "candidate_frames"),
        target_num_frames=8
    )

    frames_data = []
    for sf in report.selected_frames:
        d = sf.to_dict()
        # Find matching extracted record file path
        for r in extracted_records:
            if r["frame_index"] == sf.frame_index:
                d["file_path"] = r["file_path"]
                break
        t = sf.timestamp_sec
        if t < 2.0:
            d["selection_reason"] = "Initial courtyard overview, east facade and walkway entry"
        elif t < 5.0:
            d["selection_reason"] = "Courtyard lettering, pavers and reflecting pond margin"
        elif t < 8.0:
            d["selection_reason"] = "Central building rooflines, HVAC structures and solar panels"
        elif t < 11.0:
            d["selection_reason"] = "Courtyard lawn perimeter, landscaping and tree canopy"
        elif t < 14.0:
            d["selection_reason"] = "North building facade and perimeter walkway continuity"
        else:
            d["selection_reason"] = "West courtyard boundary, peripheral turf and perimeter road"
        frames_data.append(d)

    with open(out_dir / "candidate_frames_metadata.json", "w") as f:
        json.dump(frames_data, f, indent=2)

    build_candidate_contact_sheet(
        frames_data,
        out_dir / "candidate_frames_contact_sheet.png",
        scene_title="Campus 0049"
    )


def analyze_river_bridge():
    print("\n" + "=" * 70)
    print("ANALYZING RIVER/BRIDGE CANDIDATE FRAMES (CORRIDOR 18.0s - 48.0s)")
    print("=" * 70)
    out_dir = OUT_BASE / "river_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "candidate_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    analyzer = OpenCV5FrameAnalyzer()

    # If desktop video exists, use the 18.0s - 48.0s corridor; otherwise use segment_52s_64s.mp4
    video_to_use = RIVER_VIDEO
    cap = cv2.VideoCapture(str(video_to_use))
    fps = cap.get(cv2.CAP_PROP_FPS) or 59.94
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total_frames / fps

    if dur > 50.0:
        t_start, t_end = 18.0, 48.0
    else:
        t_start, t_end = 0.0, dur

    target_num_frames = 8
    f_start = int(t_start * fps)
    f_end = int(t_end * fps)
    window_frames = (f_end - f_start) // target_num_frames

    frames_data = []
    reasons = [
        "Approach road continuity, river south embankment & tree line",
        "Near bridge abutment, south pier foundation & rip-rap bank",
        "Bridge deck entry, railings and concrete pier support",
        "Central bridge span, roadway asphalt and river channel",
        "North pier structure, under-deck geometry and water edge",
        "North bridge abutment, railing terminus and bank transition",
        "Exit road continuity, north terrain and peripheral vegetation",
        "Corridor exit perspective, outer terrain and northern boundary"
    ]

    from src.video.opencv5_frame_analyzer import compute_single_frame_metrics

    for i in range(target_num_frames):
        win_f0 = f_start + i * window_frames
        win_f1 = min(f_start + (i + 1) * window_frames, f_end)
        
        # Candidate search in window (stride 15 frames)
        best_cand = None
        best_q = -1e9
        best_gray = None

        for f_idx in range(win_f0, win_f1, 15):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                continue
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            ts = f_idx / fps
            m, _, _ = compute_single_frame_metrics(gray, f_idx, ts, analyzer.orb)
            if m.composite_quality_score > best_q:
                best_q = m.composite_quality_score
                best_cand = m
                best_gray = gray
                best_bgr = frame_bgr

        if best_cand is not None:
            ts = best_cand.timestamp_sec
            f_path = frames_dir / f"frame_{i:04d}_{ts:.3f}s.png"
            cv2.imwrite(str(f_path), best_bgr)
            
            d = {
                "selection_rank": i,
                "frame_index": best_cand.frame_index,
                "timestamp_sec": round(ts, 3),
                "file_path": str(f_path.resolve()),
                "metrics": best_cand.to_dict(),
                "pairwise_with_prev": None,
                "selection_reason": reasons[i],
                "uncertainty_note": None
            }
            frames_data.append(d)

    cap.release()

    with open(out_dir / "candidate_frames_metadata.json", "w") as f:
        json.dump(frames_data, f, indent=2)

    build_candidate_contact_sheet(
        frames_data,
        out_dir / "candidate_frames_contact_sheet.png",
        scene_title="River / Bridge"
    )


if __name__ == "__main__":
    analyze_campus()
    analyze_river_bridge()
    print("\n[+] Phase 2D candidate frame analysis complete.")
