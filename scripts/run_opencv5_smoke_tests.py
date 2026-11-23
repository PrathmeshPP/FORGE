"""F.O.R.G.E — Phase A: Isolated OpenCV 5 Smoke Tests Across All Four Benchmark Videos.

Performs deterministic OpenCV 5 quality and viewpoint diversity analysis on:
1. Campus 0049
2. River / Bridge
3. Desert Road
4. Forest Road

Saves reports to: outputs/opencv5_experiments/final_release_qa/opencv5_smoke_tests/
Strictly QA-only execution — ZERO reconstruction mutations.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer

OUT_DIR = PROJECT_ROOT / "outputs/opencv5_experiments/final_release_qa/opencv5_smoke_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = {
    "campus_0049": PROJECT_ROOT / "data/source_videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4",
    "river_bridge": PROJECT_ROOT / "data/source_videos/segment_52s_64s.mp4",
    "desert_road": PROJECT_ROOT / "data/source_videos/desert_road.mp4",
    "forest_road": PROJECT_ROOT / "data/source_videos/forest_road.mp4",
}


def main():
    print("=" * 75)
    print("OPENCV 5 ISOLATED SMOKE TESTS ACROSS ALL FOUR BENCHMARK VIDEOS")
    print("=" * 75)

    analyzer = OpenCV5FrameAnalyzer(orb_nfeatures=500)
    print(f"[*] Initialized OpenCV5FrameAnalyzer (OpenCV {analyzer.opencv_version})")

    all_results = {}

    for scene_key, vpath in VIDEOS.items():
        print(f"\n[*] Analyzing video for {scene_key.upper()}: {vpath.name}...")
        t0 = time.perf_counter()
        report = analyzer.analyze_video(
            video_path=str(vpath),
            target_num_frames=6,
            max_candidates=30
        )
        elapsed = time.perf_counter() - t0

        rep_dict = report.to_dict()
        out_json = OUT_DIR / f"{scene_key}_opencv5_curation_report.json"
        with open(out_json, "w") as f:
            json.dump(rep_dict, f, indent=2)

        print(f"    Selected {len(report.selected_frames)} frames in {elapsed:.2f}s:")
        for s in report.selected_frames:
            print(f"      Frame {s.frame_index:4d} @ {s.timestamp_sec:6.2f}s | Lap: {s.metrics.laplacian_sharpness:7.1f} | RMS: {s.metrics.rms_contrast:5.1f} | Q: {s.metrics.composite_quality_score:5.1f} | Reason: {s.selection_reason}")

        all_results[scene_key] = {
            "video_path": str(vpath.resolve()),
            "video_duration_sec": report.duration_sec,
            "total_video_frames": report.total_frames,
            "resolution": report.native_resolution,
            "selected_frames": [
                {
                    "rank": s.selection_rank,
                    "frame_index": s.frame_index,
                    "timestamp_sec": s.timestamp_sec,
                    "laplacian_sharpness": round(s.metrics.laplacian_sharpness, 2),
                    "tenengrad_sharpness": round(s.metrics.tenengrad_sharpness, 2),
                    "rms_contrast": round(s.metrics.rms_contrast, 2),
                    "composite_quality_score": round(s.metrics.composite_quality_score, 2),
                    "selection_reason": s.selection_reason
                }
                for s in report.selected_frames
            ],
            "summary": report.selection_summary
        }

    summary_file = OUT_DIR / "opencv5_smoke_tests_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Saved all OpenCV 5 smoke test reports to: {OUT_DIR}")


if __name__ == "__main__":
    main()
