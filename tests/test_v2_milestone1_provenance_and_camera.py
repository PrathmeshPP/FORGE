"""
Automated Tests — F.O.R.G.E. V2 Milestone 1 (Traceability, DAG Cache, Canonical Camera).
Covers Part M Tests #1–#8, #11, #12 and Phase 1 Acceptance Tests #1–#7 on synthetic inputs.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import (
    CanonicalCameraFrame,
    CanonicalCameraRig,
    apply_sim3_to_extrinsic,
    apply_sim3_to_points,
    backproject_pixels,
    camera_center_from_extrinsic,
    compute_preprocessing_geometry,
    gltf_canonical_to_world,
    infer_to_native_intrinsics,
    native_to_infer_intrinsics,
    project_world_points,
    world_to_gltf_canonical,
)
from src.reconstruction.run_provenance import (
    DependencyCacheStore,
    RunProvenanceTracker,
    StageExecutionRecord,
    compare_run_manifests,
    compute_file_sha256,
    utc_now_iso,
)


class TestCanonicalCameraAndProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_1_camera_coordinate_round_trip(self):
        """World <-> Pixel <-> Camera <-> glTF Canonical transforms must round-trip cleanly."""
        K = np.array([[220.0, 0.0, 120.0], [0.0, 220.0, 216.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ang = 0.25
        R_cam = np.array([
            [np.cos(ang), 0.0, np.sin(ang)],
            [0.0, 1.0, 0.0],
            [-np.sin(ang), 0.0, np.cos(ang)],
        ])
        ext = np.hstack([R_cam, np.array([[0.3], [-0.15], [1.2]])])

        rng = np.random.default_rng(7)
        pts_w = rng.uniform([-1.5, -0.5, 1.0], [1.5, 0.8, 4.0], size=(40, 3))

        u, v, z = project_world_points(pts_w, K, ext)
        pts_back = backproject_pixels(u, v, z, K, ext)
        np.testing.assert_allclose(pts_back, pts_w, atol=1e-10)

        # Sim(3) consistency: transforming both world points and extrinsic keeps pixel projection invariant
        s_w = 1.37
        phi = -0.42
        R_w = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
        t_w = np.array([0.8, -0.4, 1.1])
        pts_w2 = apply_sim3_to_points(pts_w, s_w, R_w, t_w)
        ext2 = apply_sim3_to_extrinsic(ext, s_w, R_w, t_w)
        u2, v2, z2 = project_world_points(pts_w2, K, ext2)
        np.testing.assert_allclose(u2, u, atol=1e-9)
        np.testing.assert_allclose(v2, v, atol=1e-9)
        np.testing.assert_allclose(z2, z * s_w, atol=1e-9)

        # Canonical glTF upright round-trip
        pts_gltf, ctr = world_to_gltf_canonical(pts_w2)
        pts_w2_rec = gltf_canonical_to_world(pts_gltf, ctr)
        np.testing.assert_allclose(pts_w2_rec, pts_w2, atol=1e-11)

        # Center consistency
        c = camera_center_from_extrinsic(ext2)
        np.testing.assert_allclose(c, s_w * (R_w @ camera_center_from_extrinsic(ext)) + t_w, atol=1e-10)

    def test_2_intrinsics_transform_after_resize_and_crop(self):
        """Landscape, portrait, and extreme aspect ratio crop + balanced resize round-trip."""
        # Landscape electronics_lab-style 848x478
        pre_land = compute_preprocessing_geometry(848, 478, image_resolution=320, patch_size=16)
        self.assertFalse(pre_land["was_cropped"])
        self.assertEqual((pre_land["infer_width"], pre_land["infer_height"]), (432, 240))

        # Portrait B-314-style 480x854
        pre_port = compute_preprocessing_geometry(480, 854, image_resolution=320, patch_size=16)
        self.assertFalse(pre_port["was_cropped"])
        self.assertEqual((pre_port["infer_width"], pre_port["infer_height"]), (240, 432))

        # Extreme wide 1200x400 -> cropped to 800x400 (left=200)
        pre_wide = compute_preprocessing_geometry(1200, 400, image_resolution=320, patch_size=16)
        self.assertTrue(pre_wide["was_cropped"])
        self.assertEqual(pre_wide["crop_box_xywh"], [200, 0, 800, 400])

        K_infer = np.array([[200.0, 0.0, pre_wide["infer_width"] * 0.5], [0.0, 200.0, pre_wide["infer_height"] * 0.5], [0.0, 0.0, 1.0]])
        K_nat = infer_to_native_intrinsics(K_infer, pre_wide)
        # Principal point should map back to native image center x=600
        self.assertAlmostEqual(K_nat[0, 2], 600.0, places=6)
        K_infer_round = native_to_infer_intrinsics(K_nat, pre_wide)
        np.testing.assert_allclose(K_infer_round, K_infer, atol=1e-9)

    def test_3_to_12_provenance_dag_cache_invalidation_and_comparison(self):
        """
        Tests input/frame hashing, run_manifest creation, cache reuse, invalidation on
        config/frame/pose/calibration changes, GLB ancestor traceability, and run comparator.
        """
        vid_path = self.root / "synthetic.mp4"
        vid_path.write_bytes(b"FAKE_MP4_STREAM_DATA_V1")
        frame_p = self.root / "f000.png"
        frame_p.write_bytes(b"PNG_CONTENT_A")
        ckpt_p = self.root / "fake_vggt.pt"
        ckpt_p.write_bytes(b"VGGT_CKPT_V1")
        glb_p = self.root / "out_a" / "model.glb"
        glb_p.parent.mkdir(parents=True, exist_ok=True)
        glb_p.write_bytes(b"GLB_V1_BYTES")

        def _run_sim(cfg_val: str, frame_bytes: bytes, pose_ver: str, calib_ver: str, glb_bytes: bytes, out_sub: str):
            frame_p.write_bytes(frame_bytes)
            out_dir = self.root / out_sub
            out_dir.mkdir(parents=True, exist_ok=True)
            tracker = RunProvenanceTracker(
                repo_root=PROJECT_ROOT,
                output_dir=out_dir,
                video_path=vid_path,
                scene_id="synthetic_scene",
                resolved_config={"meshing": cfg_val, "voxel": 0.012},
                checkpoint_path=ckpt_p,
                cache_root=self.root / "cache",
                run_history_dir=self.root / "history",
                code_files_override=["src/reconstruction/canonical_camera.py"],
            )
            tracker.set_video_metadata(848, 478, 30.0, 10.0, 300)
            fsha = compute_file_sha256(frame_p)
            tracker.set_frame_provenance(
                [{"frame_index": 0, "timestamp_sec": 0.0, "dimensions_wh": [848, 478], "file_path": str(frame_p), "frame_sha256": fsha, "decision": "ACCEPTED_KEYFRAME"}],
                windows=[[0]],
            )

            rig = CanonicalCameraRig(
                run_id=tracker.run_id,
                scene_id="synthetic_scene",
                video_sha256=tracker.video_sha256,
                calibration_version=calib_ver,
                pose_version=pose_ver,
                frames=[
                    CanonicalCameraFrame(
                        selected_index=0, orig_frame_index=0, timestamp_sec=0.0, frame_path=str(frame_p),
                        frame_sha256=fsha or "", orig_width=848, orig_height=478, infer_width=432, infer_height=240,
                        crop_box_xywh=[0, 0, 848, 478], sx_crop_to_infer=432/848, sy_crop_to_infer=240/478,
                        intrinsics_infer=[[200, 0, 216], [0, 200, 120], [0, 0, 1]],
                        intrinsics_native=[[392, 0, 424], [0, 392, 239], [0, 0, 1]],
                        extrinsic_w2c=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                    )
                ],
            )
            rig_sha = rig.save_json(out_dir / "camera_rig.json")
            tracker.set_calibration_provenance(rig.to_dict())

            # Stage 1: VGGT depends ONLY on frames, resolution, ckpt — NOT meshing cfg or downstream pose_ver
            fp_vggt = tracker.cache_store.build_dependency_fingerprint(
                "vggt_window_inference", "2.0",
                {"frames_digest": tracker.frame_prov["frames_digest_sha256"]},
                {"image_resolution": 320, "window_size": 6, "overlap": 3},
                tracker.code_fingerprint,
                {"vggt_ckpt": tracker.checkpoint_sha256},
                calibration_version="vggt_native", pose_version="raw_window",
            )
            hit_v = tracker.cache_store.lookup("vggt_window_inference", fp_vggt)
            if hit_v:
                rec_v = StageExecutionRecord(
                    "vggt_window_inference", "2.0", fp_vggt, ["frame_selection"],
                    start_time_utc=utc_now_iso(), end_time_utc=utc_now_iso(), duration_sec=0.01,
                    status="REUSED", cache_status="HIT", reuse_or_skip_reason="identical_vggt_dependencies",
                )
            else:
                tracker.cache_store.store("vggt_window_inference", "2.0", fp_vggt, json_payload={"ok": True})
                rec_v = StageExecutionRecord(
                    "vggt_window_inference", "2.0", fp_vggt, ["frame_selection"],
                    start_time_utc=utc_now_iso(), end_time_utc=utc_now_iso(), duration_sec=0.05,
                    status="EXECUTED", cache_status="MISS",
                )
            tracker.record_stage(rec_v)

            # Stage 2: World backprojection depends on vggt + pose_ver + calib_ver
            fp_world = tracker.cache_store.build_dependency_fingerprint(
                "multiview_depth_validation", "2.0",
                {"vggt": fp_vggt, "rig": rig_sha},
                {"min_conf": 0.65},
                tracker.code_fingerprint,
                calibration_version=calib_ver, pose_version=pose_ver,
            )
            hit_w = tracker.cache_store.lookup("multiview_depth_validation", fp_world)
            if hit_w:
                rec_w = StageExecutionRecord("multiview_depth_validation", "2.0", fp_world, ["vggt_window_inference"], status="REUSED", cache_status="HIT", reuse_or_skip_reason="cached")
            else:
                tracker.cache_store.store("multiview_depth_validation", "2.0", fp_world, json_payload={"pts": 100})
                rec_w = StageExecutionRecord("multiview_depth_validation", "2.0", fp_world, ["vggt_window_inference"], status="EXECUTED", cache_status="MISS")
            tracker.record_stage(rec_w)

            # Stage 3: Meshing depends on world + meshing cfg_val
            fp_mesh = tracker.cache_store.build_dependency_fingerprint(
                "mesh_selection_qa", "2.0",
                {"mv_depth": fp_world},
                {"meshing": cfg_val, "voxel": 0.012},
                tracker.code_fingerprint,
            )
            hit_m = tracker.cache_store.lookup("mesh_selection_qa", fp_mesh)
            if hit_m:
                rec_m = StageExecutionRecord("mesh_selection_qa", "2.0", fp_mesh, ["multiview_depth_validation"], status="REUSED", cache_status="HIT", reuse_or_skip_reason="cached")
            else:
                tracker.cache_store.store("mesh_selection_qa", "2.0", fp_mesh, json_payload={"method": cfg_val})
                rec_m = StageExecutionRecord("mesh_selection_qa", "2.0", fp_mesh, ["multiview_depth_validation"], status="EXECUTED", cache_status="MISS")
            tracker.record_stage(rec_m)

            out_glb = out_dir / "model.glb"
            out_glb.write_bytes(glb_bytes)
            m_path, manifest = tracker.finalize_and_write({"glb": str(out_glb)}, {"ok": True}, "PASS")
            return manifest, m_path

        # Run 1 (Cold)
        m1, _ = _run_sim("poisson", b"PNG_CONTENT_A", "pose_v1", "cal_v1", b"GLB_V1", "run1")
        self.assertEqual([s["cache_status"] for s in m1["stages"]], ["MISS", "MISS", "MISS"])

        # Run 2 (Identical -> All cache hits, expected identical GLB)
        m2, _ = _run_sim("poisson", b"PNG_CONTENT_A", "pose_v1", "cal_v1", b"GLB_V1", "run2")
        self.assertNotEqual(m1["run_id"], m2["run_id"])
        self.assertEqual([s["status"] for s in m2["stages"]], ["REUSED", "REUSED", "REUSED"])
        self.assertTrue(m2["final_provenance"]["run_comparison"]["glb_identical_expected"])
        self.assertIsNone(m2["final_provenance"]["run_comparison"]["diagnostic_warning"])

        # Run 3 (Meshing config changed -> VGGT & MV cache HIT, only meshing invalidated! Plus test identical-GLB diagnostic if GLB didn't change)
        m3, _ = _run_sim("tsdf", b"PNG_CONTENT_A", "pose_v1", "cal_v1", b"GLB_V1", "run3")
        self.assertEqual([s["cache_status"] for s in m3["stages"]], ["HIT", "HIT", "MISS"])
        self.assertIsNotNone(m3["final_provenance"]["run_comparison"]["diagnostic_warning"])

        # Run 4 (Pose changed -> VGGT still HIT, world & mesh invalidated!)
        m4, _ = _run_sim("tsdf", b"PNG_CONTENT_A", "pose_v2_ba", "cal_v1", b"GLB_V2_POSE", "run4")
        self.assertEqual([s["cache_status"] for s in m4["stages"]], ["HIT", "MISS", "MISS"])
        self.assertIsNone(m4["final_provenance"]["run_comparison"]["diagnostic_warning"])

        # Run 5 (Calibration changed -> world & mesh invalidated while VGGT raw HIT!)
        m5, _ = _run_sim("tsdf", b"PNG_CONTENT_A", "pose_v2_ba", "cal_v2_intr", b"GLB_V3_CAL", "run5")
        self.assertEqual([s["cache_status"] for s in m5["stages"]], ["HIT", "MISS", "MISS"])

        # Run 6 (Frame content changed -> VGGT + all descendants invalidated!)
        m6, m6_path = _run_sim("tsdf", b"PNG_CONTENT_MODIFIED", "pose_v2_ba", "cal_v2_intr", b"GLB_V4_FRM", "run6")
        self.assertEqual([s["cache_status"] for s in m6["stages"]], ["MISS", "MISS", "MISS"])
        self.assertEqual(m6["input_provenance"]["video_sha256"], compute_file_sha256(vid_path))
        self.assertTrue(m6_path.exists())


if __name__ == "__main__":
    unittest.main()
