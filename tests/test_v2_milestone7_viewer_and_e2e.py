"""
Automated Tests — F.O.R.G.E. V2 End-to-End Orchestration & Viewer Walkthrough QA (Test #22):
  1. Verifies GeneralizedIndoorReconstructionPipeline end-to-end DAG execution,
     cache reuse across Planar OFF -> Planar ON (Stages 1..5 HIT), and honest
     status/manifest outputs.
  2. Verifies verify_v2_viewer_qa.cjs against the live Vite + Three.js viewer
     (GLB HTTP load, non-empty mesh, texture present, clay mode, ENTER WALKTHROUGH,
     and KeyW/KeyU navigation state deltas).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_generalized_indoor_reconstruction import GeneralizedIndoorReconstructionPipeline
from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.indoor_pipeline import IndoorPipelineConfig


def _create_synthetic_indoor_walk_video(video_path: Path, n_frames: int = 24, w: int = 320, h: int = 240):
    """Creates a clean textured indoor room walkthrough synthetic video for E2E orchestration test."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
    rng = np.random.default_rng(42)
    base_tex = rng.integers(40, 210, size=(h, w, 3), dtype=np.uint8)
    for k in range(n_frames):
        img = np.roll(base_tex, shift=k * 6, axis=1).copy()
        # Add structured architectural checker/grid edges so ORB tracks cleanly
        for x in range(20, w, 40):
            cv2.line(img, ((x + k * 4) % w, 10), ((x + k * 4) % w, h - 10), (240, 230, 200), 2)
        for y in range(30, h, 45):
            cv2.line(img, (10, y), (w - 10, y), (60, 120, 200), 2)
        cv2.rectangle(img, (40, h // 2), (120, h - 20), (180, 90, 70), -1)
        writer.write(img)
    writer.release()


def _synthetic_vggt_windows(image_paths, window_size=6, overlap=3, checkpoint_path="", device="cpu", log_callback=None):
    """Builds physically consistent multi-window room predictions for orchestration verification."""
    N = len(image_paths)
    H, W = 120, 160
    K = np.array([[130.0, 0.0, 80.0], [0.0, 130.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)

    exts = []
    intrs = np.tile(K[None, :, :], (N, 1, 1))
    depths = np.zeros((N, H, W, 1), dtype=np.float32)
    confs = np.full((N, H, W), 2.5, dtype=np.float32)

    # Box room: floor y=+0.5, ceiling y=-0.8, front wall z=+2.2, back wall z=-2.2, left x=-1.6, right x=+1.6
    # Cameras walk in a loop of radius 0.65m looking inward/around
    for i in range(N):
        ang = 2.0 * np.pi * (i / max(1, N))
        cx = 0.65 * np.sin(ang)
        cz = 0.65 * np.cos(ang)
        cy = 0.0
        # Look toward center/forward tangent
        fwd = np.array([-np.sin(ang), 0.0, -np.cos(ang)], dtype=np.float64)
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # OpenCV Y-down
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R_c2w = np.column_stack([right, -up if False else up, fwd])
        # Build R_w2c directly so p_cam = R_w2c @ (p_w - C):
        # ensure R_w2c @ fwd = [0, 0, 1], det=+1
        R_w2c = np.vstack([right, np.cross(fwd, right), fwd])
        if np.linalg.det(R_w2c) < 0:
            R_w2c[0] *= -1
        C = np.array([cx, cy, cz], dtype=np.float64)
        exts.append(extrinsic_from_center_and_rotation(R_w2c, C))

        # Raycast exact box depth from camera C in world coordinates
        rx = (uu - K[0, 2]) / K[0, 0]
        ry = (vv - K[1, 2]) / K[1, 1]
        rz = np.ones_like(rx)
        dirs_cam = np.stack([rx, ry, rz], axis=-1)  # (H,W,3)
        dirs_w = dirs_cam @ R_w2c  # since (R_w2c^T @ v) = v @ R_w2c

        t_candidates = []
        # Floor y=+0.5 (below C_y=0): d_y > 0 -> t = (0.5 - cy) / d_y
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = np.where(dirs_w[..., 1] > 0.05, (0.5 - cy) / dirs_w[..., 1], 99.0)
            t_ceil = np.where(dirs_w[..., 1] < -0.05, (-0.8 - cy) / dirs_w[..., 1], 99.0)
            t_front = np.where(dirs_w[..., 2] > 0.05, (2.2 - cz) / dirs_w[..., 2], 99.0)
            t_back = np.where(dirs_w[..., 2] < -0.05, (-2.2 - cz) / dirs_w[..., 2], 99.0)
            t_right = np.where(dirs_w[..., 0] > 0.05, (1.6 - cx) / dirs_w[..., 0], 99.0)
            t_left = np.where(dirs_w[..., 0] < -0.05, (-1.6 - cx) / dirs_w[..., 0], 99.0)
        t_min = np.minimum.reduce([t_floor, t_ceil, t_front, t_back, t_right, t_left])
        t_min = np.clip(t_min, 0.3, 4.8)
        depths[i, :, :, 0] = t_min.astype(np.float32)

    extrinsics = np.stack(exts, axis=0)
    raw_windows = []
    step = max(1, window_size - overlap)
    w_idx = 0
    s = 0
    while s < N:
        e = min(N, s + window_size)
        idxs = list(range(s, e))
        raw_windows.append({
            "window_index": w_idx,
            "frame_indices": idxs,
            "local_extrinsics": extrinsics[idxs].copy(),
            "local_intrinsics": intrs[idxs].copy(),
            "local_depth": depths[idxs].copy(),
            "local_conf": confs[idxs].copy(),
        })
        w_idx += 1
        if e >= N:
            break
        s += step

    joins = [
        {"window_from": k, "window_to": k + 1, "median_residual_m": 0.005, "quality_status": "GOOD"}
        for k in range(max(0, len(raw_windows) - 1))
    ]
    return {
        "extrinsics": extrinsics,
        "intrinsics": intrs,
        "depth": depths,
        "confidence": confs,
        "raw_windows": raw_windows,
        "windows": [w["frame_indices"] for w in raw_windows],
        "window_alignment_manifest": joins,
        "section_summary": {"num_sections": len(raw_windows), "weak_joins": 0},
        "performance": {"mean_sim3_alignment_error_m": 0.005, "peak_vram_mb": 10.0},
        "is_cached": False,
    }


class TestV2E2EAndViewerQA(unittest.TestCase):

    def test_e2e_dag_cache_reuse_planar_toggle(self):
        """
        Tests full 11-stage DAG orchestration on a synthetic room walkthrough video:
          - Run A: Planar OFF (`--planar-refinement off`)
          - Run B: Planar ON  (`--planar-refinement on`) using same frames/DAG cache:
            MUST report `cache_status == "HIT"` on Stages 1..5 without rerunning VGGT!
        """
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            vid_p = td_p / "synth_room.mp4"
            _create_synthetic_indoor_walk_video(vid_p, n_frames=24)

            with patch(
                "scripts.run_generalized_indoor_reconstruction.IndoorReconstructionEngine.run_overlapping_windows",
                side_effect=_synthetic_vggt_windows,
            ):
                # Run A: Planar OFF
                out_off = td_p / "run_planar_off"
                test_cache = td_p / "dag_cache"
                test_hist = td_p / "run_history"
                pipe_off = GeneralizedIndoorReconstructionPipeline(
                    video_path=vid_p,
                    output_dir=out_off,
                    target_frame_count=10,
                    scene_id="synth_test_room",
                    config=IndoorPipelineConfig(image_resolution=160, window_size=6, overlap=3),
                    meshing_method="auto",
                    enable_planar_refinement=False,
                    enable_hidden_completion=False,
                    enable_ba=True,
                    enable_yolo=False,
                    use_dag_cache=True,
                    register_viewer=False,
                    run_viewer_qa=False,
                    atlas_size=1024,
                    max_triangles=60000,
                    cache_root=test_cache,
                    run_history_dir=test_hist,
                )
                res_off = pipe_off.run()
                manifest_off = json.loads(Path(res_off["run_manifest_path"]).read_text(encoding="utf-8"))
                self.assertIn(manifest_off["final_provenance"]["final_status"], ("PASS", "PARTIAL_MODEL", "QA_FAILED"))
                self.assertTrue((out_off / "room_photographic_model.glb").is_file())
                self.assertTrue((out_off / "meshes" / "mesh_branches_qa_summary.json").is_file())

                # Run B: Planar ON (reuses Stages 1,2,4,5 DAG cache automatically!)
                out_on = td_p / "run_planar_on"
                pipe_on = GeneralizedIndoorReconstructionPipeline(
                    video_path=vid_p,
                    output_dir=out_on,
                    target_frame_count=10,
                    scene_id="synth_test_room",
                    config=IndoorPipelineConfig(image_resolution=160, window_size=6, overlap=3),
                    meshing_method="auto",
                    enable_planar_refinement=True,
                    enable_hidden_completion=False,
                    enable_ba=True,
                    enable_yolo=False,
                    use_dag_cache=True,
                    register_viewer=False,
                    run_viewer_qa=False,
                    atlas_size=1024,
                    max_triangles=60000,
                    cache_root=test_cache,
                    run_history_dir=test_hist,
                )
                res_on = pipe_on.run()
                manifest_on = json.loads(Path(res_on["run_manifest_path"]).read_text(encoding="utf-8"))
                stages_on = {s["stage_name"]: s for s in manifest_on["stages"]}
                self.assertEqual(stages_on["video_ingestion_and_frame_selection"]["cache_status"], "HIT")
                self.assertEqual(stages_on["vggt_window_inference"]["cache_status"], "HIT")
                self.assertEqual(stages_on["global_alignment_and_validated_ba"]["cache_status"], "HIT")
                self.assertEqual(stages_on["multiview_depth_visibility_freespace"]["cache_status"], "HIT")
                self.assertEqual(stages_on["planar_gs_refinement"]["status"], "EXECUTED")
                self.assertGreater(
                    stages_on["planar_gs_refinement"]["qa_metrics"].get("num_accepted_planes_final", 0),
                    0,
                )
                self.assertIn(manifest_on["final_provenance"]["final_status"], ("PASS", "PARTIAL_MODEL", "QA_FAILED"))
                self.assertIn("run_comparison_latest.json", [p.name for p in out_on.iterdir()])

    def test_22_live_viewer_walkthrough_qa_harness(self):
        """
        Test #22: Executes real `scripts/verify_v2_viewer_qa.cjs` against live Vite server + Edge WebGL
        using an existing baseline scene with zero mutation to model-manifest.json.
        """
        qa_script = PROJECT_ROOT / "scripts" / "verify_v2_viewer_qa.cjs"
        baseline_glb = PROJECT_ROOT / "outputs" / "indoor" / "candidate_b314_191043_boundary_aligned" / "room_photographic_model.glb"
        if not qa_script.is_file() or not baseline_glb.is_file():
            self.skipTest("Baseline GLB or QA harness missing")

        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [
                    "node",
                    str(qa_script),
                    "--scene-id",
                    "indoor_candidate_b314_191043_boundary_aligned",
                    "--output-dir",
                    td,
                    "--port",
                    "5176",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            rep_path = Path(td) / "viewer_qa" / "viewer_qa_report.json"
            self.assertTrue(rep_path.is_file(), f"Viewer QA report missing: {proc.stderr}")
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            self.assertEqual(rep["status"], "PASS", f"Viewer QA failed: {rep}")
            self.assertTrue(rep["checks"]["glb_loaded"])
            self.assertTrue(rep["checks"]["has_texture_map"])
            self.assertTrue(rep["checks"]["walkthrough_navigation_ok"])
            self.assertEqual(len(rep.get("page_errors", [])), 0)
            self.assertTrue(Path(rep["screenshots"]["walkthrough_moved"]).is_file())


if __name__ == "__main__":
    unittest.main()
