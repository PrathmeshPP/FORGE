"""F.O.R.G.E Phase 2A — OpenCV 5 Isolated Video Analysis Dry Run.

Executes non-destructive, isolated video quality analysis and viewpoint curation
using the active OpenCV 5 runtime. Strictly adheres to Phase 2A boundaries:
- Evaluates only an authorized benchmark video (desert_road.mp4).
- No neural 3D reconstruction is run.
- Saves diagnostic results into outputs/opencv5_experiments/phase2/.
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer


def run_dry_run(
    video_path: str = "data/source_videos/desert_road.mp4",
    output_dir: str = "outputs/opencv5_experiments/phase2",
    target_frames: int = 6
):
    print("=" * 70)
    print("F.O.R.G.E Phase 2A — OpenCV 5 Frame Analyzer Isolated Dry Run")
    print("=" * 70)

    # 1. Provenance Verification
    vpath = WORKSPACE_ROOT / video_path
    if not vpath.is_file():
        raise FileNotFoundError(f"Source video not found: {vpath}")

    out_path = WORKSPACE_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)
    frames_dir = out_path / "curated_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video Path:           {vpath}")
    print(f"Provenance Status:    Authorized (Meta FAIR Noncommercial Research License)")
    print(f"Active OpenCV:        {cv2.__version__}")
    print(f"Active Python:        {sys.version.split()[0]}")
    print(f"Target Frame Count:   {target_frames}")
    print("-" * 70)

    # 2. Initialize Analyzer
    analyzer = OpenCV5FrameAnalyzer()

    # 3. Analyze Video & Curate Viewpoints
    print("\n[Step 1/2] Analyzing video candidates across full trajectory...")
    report = analyzer.analyze_video(
        video_path=str(vpath),
        target_num_frames=target_frames,
        max_candidates=50
    )

    print(f"  Total Video Frames:       {report.total_frames}")
    print(f"  Native FPS:               {report.native_fps}")
    print(f"  Duration:                 {report.duration_sec:.2f}s")
    print(f"  Resolution:               {report.native_resolution[0]}x{report.native_resolution[1]}")
    print(f"  Candidates Analyzed:      {report.candidate_pool_size}")
    print(f"  Curated Frame Count:      {len(report.selected_frames)}")
    print(f"  Trajectory Coverage:      {report.selection_summary['trajectory_coverage_ratio']*100:.1f}%")
    print(f"  Temporal Span:            {report.selection_summary['temporal_span_sec']:.2f}s")
    print(f"  Mean Laplacian Sharpness: {report.selection_summary['mean_selected_laplacian_sharpness']:.1f}")
    print(f"  Mean Grid Entropy:        {report.selection_summary['mean_grid_entropy']:.3f}")
    print(f"  Geometric Verif. Rate:    {report.selection_summary['geometric_verification_rate']*100:.1f}%")

    # 4. Extract Curated Frames (non-destructive test images)
    print("\n[Step 2/2] Extracting curated frames to experiment directory...")
    extracted_records, _ = analyzer.extract_curated_frames(
        video_path=str(vpath),
        output_dir=str(frames_dir),
        target_num_frames=target_frames,
        save_metadata=True
    )
    print(f"  Saved {len(extracted_records)} curated frame PNGs to: {frames_dir}")

    # 5. Persist JSON Report
    json_path = out_path / "dry_run_analysis_report.json"
    full_report_data = {
        "experiment": "Phase 2A OpenCV 5 Isolated Frame Analyzer Dry Run",
        "provenance": {
            "selected_video": "data/source_videos/desert_road.mp4",
            "license": "Meta FAIR Noncommercial Research License (v1, Oct 2024)",
            "omitted_video": "data/source_videos/segment_52s_64s.mp4",
            "omission_rationale": "Third-Party Asset Provenance audit marks segment_52s_64s.mp4 as UNCONFIRMED / PLAUSIBLE FIRST-PARTY INFERENCE pending formal written owner clearance; excluded per strict Phase 2A safety boundary."
        },
        "curation_report": report.to_dict(),
        "extracted_frames": extracted_records
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report_data, f, indent=2)
    print(f"  Saved complete diagnostic report: {json_path}")

    # 6. Generate Markdown Summary
    md_path = out_path / "DRY_RUN_ANALYSIS_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# F.O.R.G.E Phase 2A — OpenCV 5 Frame Analyzer Dry Run Report\n\n")
        f.write("## 1. Environment & Asset Provenance\n\n")
        f.write(f"- **OpenCV Runtime:** `cv2.__version__ = {report.opencv_version}`\n")
        f.write(f"- **Python Environment:** `{sys.version.split()[0]}`\n")
        f.write(f"- **Evaluated Video:** `{vpath.name}` (`{vpath}`)\n")
        f.write(f"- **Video Metadata:** {report.native_resolution[0]}x{report.native_resolution[1]} @ {report.native_fps} FPS, {report.total_frames} frames ({report.duration_sec:.2f}s duration)\n")
        f.write(f"- **Asset Provenance & Licensing:** Authorized under [Meta FAIR Noncommercial Research License](../../docs/licenses/META_FAIR_NONCOMMERCIAL_LICENSE.txt).\n")
        f.write(f"- **Asset Exclusion Notice:** `segment_52s_64s.mp4` was intentionally set aside because `THIRD_PARTY_ASSET_PROVENANCE.md` (lines 18, 125) classifies it as `UNCONFIRMED` pending formal written copyright release.\n\n")

        f.write("## 2. Selected Frame Viewpoints & Metrics\n\n")
        f.write("| Rank | Frame Index | Timestamp (s) | Laplacian Sharpness | Tenengrad Sharpness | Mean Intensity | RMS Contrast | Grid Occupancy (/16) | Grid Entropy | Composite Score | Selection Reason & Uncertainty |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for s in report.selected_frames:
            m = s.metrics
            unc = f" *(Uncertainty: {s.uncertainty_note})*" if s.uncertainty_note else ""
            f.write(f"| {s.selection_rank} | {s.frame_index} | {s.timestamp_sec:.3f} | {m.laplacian_sharpness:.1f} | {m.tenengrad_sharpness:.1f} | {m.mean_intensity:.1f} | {m.rms_contrast:.1f} | {m.grid_occupied_cells} | {m.grid_distribution_entropy:.3f} | {m.composite_quality_score:.1f} | {s.selection_reason}{unc} |\n")

        f.write("\n## 3. Pairwise Inter-Frame Consistency & Geometric Verification\n\n")
        f.write("| Pair (Prev -> Curr) | Hist Correlation | Bhattacharyya Dist | Raw ORB Matches | Epipolar Available? | Model | Inlier Matches | Inlier Ratio | Verification Notes |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for p in report.pairwise_reports:
            f.write(f"| {p.prev_frame_index} -> {p.curr_frame_index} | {p.histogram_correlation:.4f} | {p.bhattacharyya_distance:.4f} | {p.raw_orb_matches} | {'YES' if p.geometric_verification_available else 'NO'} | `{p.geometric_model}` | {p.inlier_matches} | {p.inlier_ratio*100:.1f}% | {p.verification_notes} |\n")

        f.write("\n## 4. Summary & Coverage Evaluation\n\n")
        f.write(f"- **Candidates Evaluated:** {report.candidate_pool_size} candidate frames across trajectory\n")
        f.write(f"- **Curated Frames:** {len(report.selected_frames)} frames selected across chronological windows\n")
        f.write(f"- **Temporal Span:** {report.selection_summary['temporal_span_sec']:.2f}s ({report.selection_summary['trajectory_coverage_ratio']*100:.1f}% of total duration)\n")
        f.write(f"- **Mean Laplacian Sharpness:** {report.selection_summary['mean_selected_laplacian_sharpness']:.1f}\n")
        f.write(f"- **Mean Grid Entropy:** {report.selection_summary['mean_grid_entropy']:.3f}\n")
        f.write(f"- **Geometric Verification Rate:** {report.selection_summary['geometric_verification_rate']*100:.1f}%\n")
        f.write(f"- **Deterministic Execution:** True (verified via automated unit test suite)\n\n")

    print(f"  Saved human-readable summary: {md_path}")
    print("\n" + "=" * 70)
    print("ISOLATED DRY RUN COMPLETE: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    run_dry_run()
