r"""
Run OpenCV5FrameAnalyzer on the new DJI video.
"""
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer

VIDEO_PATH = r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24"

def main():
    print("[1] Initializing OpenCV5FrameAnalyzer...")
    analyzer = OpenCV5FrameAnalyzer(orb_nfeatures=1000)
    print(f"    OpenCV runtime: {analyzer.opencv_version}")

    print("[2] Analyzing video across candidate pool (120 candidates)...")
    report = analyzer.analyze_video(
        video_path=VIDEO_PATH,
        target_num_frames=12,
        max_candidates=120
    )

    print("\n--- Analysis Summary ---")
    for k, v in report.selection_summary.items():
        print(f"  {k}: {v}")

    print(f"\n--- Selected {len(report.selected_frames)} Frames ---")
    for idx, s in enumerate(report.selected_frames):
        m = s.metrics
        print(f"  [{idx:02d}] Frame {s.frame_index:05d} | t={s.timestamp_sec:6.2f}s | LapVar={m.laplacian_sharpness:7.2f} | Tenengrad={m.tenengrad_sharpness:9.1f} | KPs={m.orb_keypoint_count:4d} | Reason={s.selection_reason}")

    # Save curation report
    report_path = OUTPUT_DIR / "reports/curation_report_full_video.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nSaved curation report to: {report_path}")

if __name__ == "__main__":
    main()
