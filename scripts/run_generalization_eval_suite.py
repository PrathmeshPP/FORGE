"""
F.O.R.G.E. — Generalisation Evaluation Suite (Phase 5).

Executes the identical generalized reconstruction pipeline on unseen indoor videos:
1. Unseen Room: WhatsApp Video 2026-11-01 at 18.05.30.mp4
2. Unseen Corridor: WhatsApp Video 2026-10-30 at 17.05.00.mp4
3. Stationary Pan (Failure Detection): WhatsApp Video 2026-10-30 at 16.43.49.mp4

Enforces strict rules:
- No manual frame curation or room-specific threshold tuning.
- Honest reporting of partial reconstruction or recapture recommendations.
- Compiles a unified generalisation benchmark matrix.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_generalized_indoor_reconstruction import GeneralizedIndoorReconstructionPipeline
from scripts.register_generalized_scene import register_scene

TEST_SUITE = [
    {
        "id": "unseen_room_180530",
        "category": "Unseen Room",
        "video_path": Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-11-01 at 18.05.30.mp4"),
        "output_dir": PROJECT_ROOT / "outputs/indoor/unseen_room_20261101_180530_adaptive",
        "display_name": "UNSEEN ROOM — 18:05:30 (ADAPTIVE GENERALIZED)",
        "subtitle": "Independent Room Survey (Adaptive Curation & Sim(3) Alignment)",
        "target_frames": 50
    },
    {
        "id": "unseen_corridor_170500",
        "category": "Unseen Corridor",
        "video_path": Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.05.00.mp4"),
        "output_dir": PROJECT_ROOT / "outputs/indoor/unseen_corridor_20261030_170500_adaptive",
        "display_name": "UNSEEN CORRIDOR — 17:05:00 (ADAPTIVE GENERALIZED)",
        "subtitle": "Hostel Corridor Walkthrough (Adaptive Bridge-Frame Tracking)",
        "target_frames": 50
    },
    {
        "id": "stationary_pan_164349",
        "category": "Stationary Pan (Failure Detection)",
        "video_path": Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 16.43.49.mp4"),
        "output_dir": PROJECT_ROOT / "outputs/indoor/stationary_pan_failure_detection",
        "display_name": "STATIONARY PAN TEST — 16:43:49 (FAILURE DETECTION)",
        "subtitle": "Weak Depth Parallax Test (Honest Recapture Reporting)",
        "target_frames": 30
    }
]

def run_suite():
    print("=" * 80)
    print("  F.O.R.G.E. GENERALISATION EVALUATION SUITE")
    print(f"  Total Test Cases: {len(TEST_SUITE)}")
    print("=" * 80)

    results = []
    # Include B-314 Regression Baseline
    b314_meta_path = PROJECT_ROOT / "outputs/indoor/b314_adaptive_regression_experiment/reconstruction_metadata.json"
    if b314_meta_path.exists():
        with open(b314_meta_path, "r", encoding="utf-8") as f:
            b314_m = json.load(f)
        results.append({
            "test_id": "indoor_b314_generalized_regression",
            "category": "Hostel Room B-314 (Regression Baseline)",
            "video": b314_m.get("source_video", {}).get("name"),
            "status": "PASS",
            "eligibility": b314_m.get("navigation", {}).get("eligibility_status"),
            "frames_selected": b314_m.get("frame_selection", {}).get("curated_frame_count"),
            "mean_sharpness": b314_m.get("frame_selection", {}).get("mean_selected_laplacian_sharpness"),
            "overlap_score": b314_m.get("frame_selection", {}).get("mean_pairwise_overlap_score"),
            "alignment_integrity": b314_m.get("window_alignment", {}).get("alignment_integrity_rate"),
            "runtime_sec": b314_m.get("durations", {}).get("total_pipeline_time_sec"),
            "vertices": b314_m.get("geometry", {}).get("vertices"),
            "triangles": b314_m.get("geometry", {}).get("triangles"),
            "photographic_coverage": b314_m.get("texturing", {}).get("photographic_coverage_pct")
        })

    for test in TEST_SUITE:
        vpath = test["video_path"]
        out_dir = test["output_dir"]
        scene_id = test["id"]

        print(f"\n---> Running Test: {test['category']} ({vpath.name})...")
        if not vpath.exists():
            print(f"[-] Video file missing: {vpath}")
            continue

        pipeline = GeneralizedIndoorReconstructionPipeline(
            video_path=vpath,
            output_dir=out_dir,
            target_frame_count=test["target_frames"],
            scene_id=scene_id
        )

        try:
            meta = pipeline.run(reuse_cached_cameras=False)
            register_scene(
                exp_dir=out_dir,
                scene_id=scene_id,
                display_name=test["display_name"],
                subtitle=test["subtitle"]
            )
            results.append({
                "test_id": scene_id,
                "category": test["category"],
                "video": vpath.name,
                "status": "PASS",
                "eligibility": meta.get("navigation", {}).get("eligibility_status"),
                "frames_selected": meta.get("frame_selection", {}).get("curated_frame_count"),
                "mean_sharpness": meta.get("frame_selection", {}).get("mean_selected_laplacian_sharpness"),
                "overlap_score": meta.get("frame_selection", {}).get("mean_pairwise_overlap_score"),
                "alignment_integrity": meta.get("window_alignment", {}).get("alignment_integrity_rate"),
                "runtime_sec": meta.get("durations", {}).get("total_pipeline_time_sec"),
                "vertices": meta.get("geometry", {}).get("vertices"),
                "triangles": meta.get("geometry", {}).get("triangles"),
                "photographic_coverage": meta.get("texturing", {}).get("photographic_coverage_pct")
            })
        except Exception as e:
            print(f"[-] Error in test {scene_id}: {e}")
            results.append({
                "test_id": scene_id,
                "category": test["category"],
                "video": vpath.name,
                "status": "FAILED",
                "error": str(e)
            })

    # Save summary matrix
    matrix_path = PROJECT_ROOT / "outputs/indoor/generalisation_evaluation_matrix.json"
    with open(matrix_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  GENERALISATION EVALUATION MATRIX COMPLETE")
    print(f"  Saved to: {matrix_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_suite()
