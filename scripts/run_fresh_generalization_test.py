import os
import sys
import json
import time
from pathlib import Path

# Add project root
root = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
sys.path.insert(0, str(root))

from scripts.run_generalized_indoor_reconstruction import GeneralizedIndoorReconstructionPipeline
from scripts.register_generalized_scene import register_scene

fresh_tests = [
    {
        "id": "fresh_room_185144",
        "category": "Fresh Room Evaluation",
        "video": Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-11-01 at 18.51.44.mp4"),
        "output_dir": root / "outputs/indoor/fresh_room_20261101_185144_eval",
        "display_name": "NOVEMBER 1 — 18:51:44 FRESH ROOM (GENERALIZATION EVAL)",
        "subtitle": "Independent Unseen Room (Zero Manual Tuning, Depth 10 Surface)",
        "target_frames": 50
    },
    {
        "id": "fresh_corridor_233142",
        "category": "Fresh Corridor Evaluation",
        "video": Path(r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 23.31.42.mp4"),
        "output_dir": root / "outputs/indoor/fresh_corridor_20261030_233142_eval",
        "display_name": "OCTOBER 30 — 23:31:42 FRESH CORRIDOR (GENERALIZATION EVAL)",
        "subtitle": "Independent Unseen Corridor (Zero Manual Tuning, 100% Floor Grounding)",
        "target_frames": 50
    }
]

def main():
    print("=" * 80)
    print("F.O.R.G.E. — Milestone 4: Generalisation on Fresh Room & Fresh Corridor")
    print("=" * 80)

    results = []

    for item in fresh_tests:
        v_path = item["video"]
        out_dir = item["output_dir"]
        s_id = item["id"]
        cat = item["category"]

        print(f"\n---> Starting {cat}: {v_path.name}")
        if not v_path.exists():
            print(f"[-] Video file missing: {v_path}")
            continue

        pipeline = GeneralizedIndoorReconstructionPipeline(
            video_path=v_path,
            output_dir=out_dir,
            target_frame_count=item["target_frames"],
            scene_id=s_id
        )

        res = pipeline.run()
        print(f"[+] Reconstruction finished for {s_id} in {res.get('durations', {}).get('total_pipeline_time_sec', 0)}s")

        # Register in viewer manifest
        register_scene(
            exp_dir=out_dir,
            scene_id=s_id,
            display_name=item["display_name"],
            subtitle=item["subtitle"]
        )

        results.append({
            "test_id": s_id,
            "category": cat,
            "source_video": v_path.name,
            "video_size_bytes": v_path.stat().st_size,
            "pipeline_time_sec": res.get("durations", {}).get("total_pipeline_time_sec"),
            "frames_selected": res.get("frame_selection", {}).get("curated_frame_count"),
            "sim3_integrity_pct": res.get("window_alignment", {}).get("alignment_integrity_rate"),
            "vertices": res.get("geometry", {}).get("vertices"),
            "triangles": res.get("geometry", {}).get("triangles"),
            "floor_continuity_pct": res.get("navigation", {}).get("floor_continuity_pct"),
            "obstacle_clearance_pct": res.get("navigation", {}).get("obstacle_clearance_pct"),
            "eligibility_status": res.get("navigation", {}).get("eligibility_status"),
            "actionable_notes": res.get("navigation", {}).get("actionable_notes")
        })

    out_summary_p = root / "outputs/indoor/fresh_generalization_summary.json"
    with open(out_summary_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Fresh Generalization Summary saved to: {out_summary_p}")

if __name__ == "__main__":
    main()
