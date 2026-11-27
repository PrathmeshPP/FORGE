"""
F.O.R.G.E. V2 — Evidence-Validated Indoor 3D Reconstruction & Digital Twin CLI

Upgraded unified entry point executing the complete 12-stage V2 DAG pipeline:
  1.  video_ingestion_and_frame_selection (OpenCV5 sharpness/pair-overlap/bridge curation + SHA-256)
  2.  vggt_window_inference               (Frozen VGGT-Omega windows + GPU cache + cleanup)
  3.  yolo_dynamic_segmentation           (Sequential 8 GB VRAM YOLOv8n-seg dynamic masking)
  4.  global_alignment_and_validated_ba   (Within-window QC + Pose Graph + Held-out BA acceptance/retry)
  5.  multiview_depth_visibility_freespace(5-state V-FUSE consistency + floor leveling + FreeSpaceVoxelEvidence)
  6.  planar_gs_refinement                (Boundary-preserving PlanarGS ON/OFF experiment toggle)
  7.  canonical_geometry_and_depth_sync   (Single canonical representation shared across mesh/texture)
  8.  mesh_branches_and_evidence_qa       (Separate Poisson & TSDF branches + Raycasting QA, no forced winner)
  9.  hidden_surface_completion           (Behind-the-Veil honest audit + opt-in free-space-safe completion)
  10. texture_baking_and_export           (Visibility-aware z-buffered baking + 8K atlas + GLB reload QA)
  11. scene_navigation_and_manifest       (Navigability & walkthrough bounds + viewer manifest registration)
  12. viewer_walkthrough_qa               (Real headless Edge/Three.js GLB reload & WASD/U navigation check)
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
import argparse
import dataclasses
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
import torch

# Enforce project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer
from src.reconstruction.canonical_camera import (
    CanonicalCameraFrame,
    CanonicalCameraRig,
    camera_center_from_extrinsic,
    compute_preprocessing_geometry,
    infer_to_native_intrinsics,
)
from src.reconstruction.run_provenance import (
    RunProvenanceTracker,
    StageArtifactRef,
    StageExecutionRecord,
    compare_run_manifests,
    compute_arrays_sha256,
    compute_file_sha256,
    compute_json_sha256,
    utc_now_iso,
)
from src.reconstruction.indoor_pipeline import (
    IndoorPipelineConfig,
    IndoorReconstructionEngine,
    compute_floor_leveling_rotation,
)
from src.reconstruction.global_alignment_ba import (
    AlignmentQAThresholds,
    GlobalAlignmentReport,
    run_global_alignment_and_validated_ba,
)
from src.reconstruction.depth_consistency_filter import (
    DepthConsistencyConfig,
    FreeSpaceVoxelEvidence,
    MultiViewDepthConsistencyFilter,
)
from src.reconstruction.planar_refinement import (
    PROTECTED_NON_PLANAR_NAMES,
    PlanarRefinementConfig,
    PlanarRegionRecord,
    associate_yolo_semantics_to_3d_points,
    project_canonical_points_to_depth_maps,
    refine_scene_geometry_planar_gs,
    run_yolo_indoor_segmentation_scheduled,
)
from src.reconstruction.mesh_branches_and_qa import (
    MeshBranchQAConfig,
    _clean_mesh_topology,
    build_canonical_scene_geometry,
    run_global_tsdf_and_conditional_repair,
)
from src.reconstruction.hidden_surface_completion import (
    HiddenCompletionConfig,
    run_optional_hidden_surface_completion,
)
from src.reconstruction.visibility_texture_baker import VisibilityAwareTextureBaker
from src.reconstruction.scene_navigation_generator import derive_scene_navigation


DEFAULT_VGGT_CHECKPOINT = Path(r"C:\Users\25beevdt025\Desktop\SIH\models\vggt_omega_1b_512.pt")
DEFAULT_YOLO_CHECKPOINT = Path(r"C:\Users\25beevdt025\Desktop\SIH\yolov8n-seg.pt")
DEFAULT_VIEWER_MANIFEST = PROJECT_ROOT / "viewer" / "src" / "data" / "model-manifest.json"
DEFAULT_VIEWER_QA_SCRIPT = PROJECT_ROOT / "scripts" / "verify_v2_viewer_qa.cjs"


def get_gpu_vram_mb() -> Tuple[float, float]:
    """Returns (allocated_mb, reserved_mb) on CUDA device 0 if available."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return round(alloc, 2), round(res, 2)
    return 0.0, 0.0


def _artifact_ref(artifact_id: str, path: Optional[Path], kind: str = "file") -> Dict[str, Any]:
    """Helper constructing a SHA-256-verified StageArtifactRef dict."""
    if path is None:
        return dataclasses.asdict(StageArtifactRef(artifact_id=artifact_id, path=None, sha256=None, kind=kind))
    p = Path(path)
    sha = compute_file_sha256(p) if p.is_file() else None
    return dataclasses.asdict(StageArtifactRef(
        artifact_id=artifact_id,
        path=str(p.resolve()) if p.exists() else str(p),
        sha256=sha,
        kind=kind,
    ))


def _load_images_rgb_at_infer(
    image_paths: List[str],
    infer_hw: Tuple[int, int],
) -> List[np.ndarray]:
    """Loads curated frames as RGB uint8 arrays resized strictly to VGGT infer (H, W)."""
    h_inf, w_inf = int(infer_hw[0]), int(infer_hw[1])
    frames_rgb: List[np.ndarray] = []
    for p in image_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            frames_rgb.append(np.zeros((h_inf, w_inf, 3), dtype=np.uint8))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != h_inf or rgb.shape[1] != w_inf:
            rgb = cv2.resize(rgb, (w_inf, h_inf), interpolation=cv2.INTER_AREA)
        frames_rgb.append(rgb)
    return frames_rgb


def _load_images_bgr_native(image_paths: List[str]) -> List[np.ndarray]:
    """Loads curated frames at native resolution as BGR uint8 arrays."""
    out: List[np.ndarray] = []
    for p in image_paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        out.append(bgr)
    return out


class GeneralizedIndoorReconstructionPipeline:
    """
    F.O.R.G.E. V2 Unified, Evidence-Validated Indoor 3D Reconstruction Pipeline.
    Preserves clean separation across observable observations vs inferred priors,
    uses SHA-256 DAG stage caching, enforces held-out BA acceptance and physical
    RaycastingScene QA, and validates real Three.js viewer GLB rendering/navigation.
    """

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        target_frame_count: int = 90,
        scene_id: Optional[str] = None,
        config: Optional[IndoorPipelineConfig] = None,
        meshing_method: str = "auto",
        enable_planar_refinement: bool = True,
        enable_hidden_completion: bool = False,
        enable_ba: bool = True,
        enable_yolo: bool = True,
        use_dag_cache: bool = True,
        register_viewer: bool = True,
        run_viewer_qa: bool = True,
        atlas_size: int = 8192,
        max_triangles: int = 450000,
        checkpoint_path: Path = DEFAULT_VGGT_CHECKPOINT,
        yolo_path: Path = DEFAULT_YOLO_CHECKPOINT,
        viewer_manifest_path: Optional[Path] = DEFAULT_VIEWER_MANIFEST,
        cache_root: Optional[Path] = None,
        run_history_dir: Optional[Path] = None,
    ):
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.target_frame_count = int(target_frame_count)
        self.scene_id = scene_id or self.output_dir.name
        self.config = config or IndoorPipelineConfig()

        # Mandatory Global TSDF architecture (auto/global_tsdf/conditional_tsdf/tsdf/poisson normalized safely)
        m_norm = str(meshing_method).strip().lower()
        if m_norm in ("auto", "conditional_tsdf", "global_tsdf", ""):
            m_norm = "global_tsdf"
        elif m_norm not in ("tsdf", "poisson", "global_tsdf", "global_tsdf_no_repair"):
            m_norm = "global_tsdf"
        self.meshing_method = m_norm
        self.config.meshing_method = self.meshing_method
        self.config.use_vggt_cache = bool(use_dag_cache)

        self.enable_planar_refinement = bool(enable_planar_refinement)
        self.enable_hidden_completion = bool(enable_hidden_completion)
        self.enable_ba = bool(enable_ba)
        self.enable_yolo = bool(enable_yolo)
        self.use_dag_cache = bool(use_dag_cache)
        self.register_viewer = bool(register_viewer)
        self.run_viewer_qa = bool(run_viewer_qa)
        self.atlas_size = int(atlas_size)
        self.max_triangles = int(max_triangles)
        self.checkpoint_path = Path(checkpoint_path)
        self.yolo_path = Path(yolo_path)
        self.viewer_manifest_path = Path(viewer_manifest_path) if viewer_manifest_path else None
        self.cache_root = Path(cache_root) if cache_root else None
        self.run_history_dir = Path(run_history_dir) if run_history_dir else None

        self.frames_dir = self.output_dir / "frames"
        self.camera_dir = self.output_dir / "camera_geometry"
        self.mesh_dir = self.output_dir / "meshes"
        self.renders_dir = self.output_dir / "renders"

        for d in [self.output_dir, self.frames_dir, self.camera_dir, self.mesh_dir, self.renders_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / "reconstruction.log"
        self.log_f = open(self.log_path, "w", encoding="utf-8")

    def log(self, msg: str):
        t_str = time.strftime("%H:%M:%S")
        line = f"[{t_str}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def _resolved_global_config(self) -> Dict[str, Any]:
        return {
            "pipeline_version": "2.2.0",
            "target_frame_count": self.target_frame_count,
            "image_resolution": int(self.config.image_resolution),
            "window_size": int(self.config.window_size),
            "overlap": int(self.config.overlap),
            "min_depth": float(self.config.min_depth),
            "max_depth": float(self.config.max_depth),
            "min_confidence": float(self.config.min_confidence),
            "voxel_size": float(self.config.voxel_size),
            "level_floor": bool(self.config.level_floor),
            "meshing_method": self.meshing_method,
            "enable_planar_refinement": self.enable_planar_refinement,
            "enable_hidden_completion": self.enable_hidden_completion,
            "enable_ba": self.enable_ba,
            "enable_yolo": self.enable_yolo,
            "atlas_size": self.atlas_size,
            "max_triangles": self.max_triangles,
        }

    def _web_url_for_path(self, target_path: Path) -> str:
        """Resolves `/outputs/...` HTTP URL cleanly for Vite viewer."""
        try:
            rel = Path(target_path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            return "/" + rel
        except ValueError:
            return f"/outputs/indoor/{self.output_dir.name}/{Path(target_path).name}"

    def _register_in_viewer_manifest(
        self,
        manifest_scene_id: str,
        deliverables: Dict[str, Optional[str]],
        nav_dict: Dict[str, Any],
        deliverable_mesh: o3d.geometry.TriangleMesh,
        pre_viewer_status: str,
        selected_method: Optional[str],
        num_frames: int,
        run_id: str,
        video_sha256: str,
    ) -> Optional[Path]:
        """Safely registers or updates ONLY this V2 scene ID inside viewer/src/data/model-manifest.json."""
        if not self.register_viewer or self.viewer_manifest_path is None:
            return None
        if not self.viewer_manifest_path.is_file():
            self.log(f"  [!] Viewer manifest not found at {self.viewer_manifest_path}; skipping registration.")
            return None

        glb_path_str = deliverables.get("glb")
        if not glb_path_str or not Path(glb_path_str).is_file():
            self.log("  [!] No valid GLB deliverable to register in viewer manifest.")
            return None

        glb_p = Path(glb_path_str)
        glb_size = glb_p.stat().st_size
        glb_sha = compute_file_sha256(glb_p) or ""
        glb_url = self._web_url_for_path(glb_p)

        reps: Dict[str, Any] = {
            "gltf_glb": {
                "type": "glb",
                "label": (
                    f"V2 {selected_method.upper() if selected_method else 'MESH'} GLB "
                    f"({glb_size / (1024 * 1024):.1f} MB, 0 OOR UVs)"
                ),
                "glb_url": glb_url,
                "file_size_bytes": int(glb_size),
                "sha256": glb_sha,
                "available": True,
            }
        }
        obj_p_str = deliverables.get("obj")
        mtl_p_str = deliverables.get("mtl")
        if obj_p_str and Path(obj_p_str).is_file() and mtl_p_str and Path(mtl_p_str).is_file():
            reps["textured_obj"] = {
                "type": "obj",
                "label": "V2 Textured Wavefront OBJ + MTL",
                "obj_url": self._web_url_for_path(Path(obj_p_str)),
                "mtl_url": self._web_url_for_path(Path(mtl_p_str)),
                "file_size_bytes": int(Path(obj_p_str).stat().st_size),
                "available": True,
            }
        comp_glb_str = deliverables.get("completed_glb")
        if comp_glb_str and Path(comp_glb_str).is_file():
            reps["completed_optional_glb"] = {
                "type": "glb",
                "label": "V2 Observed + Inferred Completion GLB (Separate Material)",
                "glb_url": self._web_url_for_path(Path(comp_glb_str)),
                "file_size_bytes": int(Path(comp_glb_str).stat().st_size),
                "available": True,
            }

        spawn = nav_dict.get("spawn_position", [0.0, 0.25, 0.0])
        init_pos = [
            round(float(spawn[0]) + 1.4, 3),
            round(float(spawn[1]) + 0.9, 3),
            round(float(spawn[2]) + 1.6, 3),
        ]

        planar_tag = "Planar ON" if self.enable_planar_refinement else "Planar OFF"
        entry = {
            "id": manifest_scene_id,
            "scene_name": manifest_scene_id,
            "display_name": f"{self.scene_id.replace('_', ' ').upper()} (F.O.R.G.E. V2)",
            "subtitle": (
                f"{num_frames} Frames | {planar_tag} | "
                f"Meshing={selected_method.upper() if selected_method else 'NONE'} | {pre_viewer_status}"
            ),
            "category": "Indoor Architecture",
            "acceptance_status": pre_viewer_status,
            "acceptance_label": f"F.O.R.G.E. V2 ({planar_tag})",
            "description": (
                f"Evidence-validated F.O.R.G.E. V2 digital twin ({planar_tag}, "
                f"BA={'ON' if self.enable_ba else 'OFF'}, "
                f"HiddenCompletion={'ON' if self.enable_hidden_completion else 'OFF'})."
            ),
            "vertex_count": int(len(deliverable_mesh.vertices)),
            "triangle_count": int(len(deliverable_mesh.triangles)),
            "default_representation": "gltf_glb",
            "representations": reps,
            "initial_camera": {
                "position": init_pos,
                "target": [0.0, 0.0, 0.0],
            },
            "reconstruction_run_id": run_id,
            "source_video_provenance": {
                "path": str(self.video_path.resolve()),
                "sha256": video_sha256,
            },
            "navigation_metadata": nav_dict,
        }

        manifest = json.loads(self.viewer_manifest_path.read_text(encoding="utf-8"))
        scenes = manifest.get("scenes", [])
        replaced = False
        for i, existing in enumerate(scenes):
            if existing.get("id") == manifest_scene_id:
                scenes[i] = entry
                replaced = True
                break
        if not replaced:
            scenes.append(entry)
        manifest["scenes"] = scenes

        tmp_p = self.viewer_manifest_path.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_p.replace(self.viewer_manifest_path)
        self.log(f"  [+] Registered `{manifest_scene_id}` in {self.viewer_manifest_path.name} (preserves all baseline entries).")
        return self.viewer_manifest_path

    def run(self, reuse_cached_cameras: bool = False) -> Dict[str, Any]:
        """Executes the complete F.O.R.G.E. V2 reconstruction and evidence verification pipeline."""
        if getattr(self, "log_f", None) is None or self.log_f.closed:
            self.log_f = open(self.log_path, "a", encoding="utf-8")
        try:
            return self._run_impl(reuse_cached_cameras=reuse_cached_cameras)
        finally:
            if getattr(self, "log_f", None) and not self.log_f.closed:
                self.log_f.close()

    def _run_impl(self, reuse_cached_cameras: bool = False) -> Dict[str, Any]:
        """Internal V2 pipeline execution wrapped for leak-free log closing."""
        if reuse_cached_cameras:
            self.use_dag_cache = True
            self.config.use_vggt_cache = True

        self.log("=" * 80)
        self.log(f"F.O.R.G.E. V2 PIPELINE — SCENE: {self.scene_id}")
        self.log(f"Input Video: {self.video_path}")
        self.log(f"Output Dir:  {self.output_dir}")
        self.log(
            f"Config:      target_frames={self.target_frame_count}, meshing={self.meshing_method}, "
            f"planar={'ON' if self.enable_planar_refinement else 'OFF'}, "
            f"ba={'ON' if self.enable_ba else 'OFF'}, yolo={'ON' if self.enable_yolo else 'OFF'}, "
            f"hidden_comp={'ON' if self.enable_hidden_completion else 'OFF'}, "
            f"dag_cache={'ON' if self.use_dag_cache else 'OFF'}"
        )
        self.log("=" * 80)

        t_pipeline_start = time.perf_counter()
        resolved_global_cfg = self._resolved_global_config()

        tracker = RunProvenanceTracker(
            repo_root=PROJECT_ROOT,
            output_dir=self.output_dir,
            video_path=self.video_path,
            scene_id=self.scene_id,
            resolved_config=resolved_global_cfg,
            checkpoint_path=self.checkpoint_path if self.checkpoint_path.is_file() else None,
            extra_asset_paths=(
                {"yolov8n_seg": self.yolo_path} if self.yolo_path.is_file() else None
            ),
            cache_root=self.cache_root,
            run_history_dir=self.run_history_dir,
        )
        cache_store = tracker.cache_store

        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        vid_dur = (vid_total / vid_fps) if vid_fps > 0 else 0.0
        tracker.set_video_metadata(vid_w, vid_h, vid_fps, vid_dur, vid_total)

        preproc_geom = compute_preprocessing_geometry(
            orig_width=max(1, vid_w),
            orig_height=max(1, vid_h),
            image_resolution=int(self.config.image_resolution),
            patch_size=16,
            mode="balanced",
        )
        infer_hw = (int(preproc_geom["infer_height"]), int(preproc_geom["infer_width"]))

        # ======================================================================
        # STAGE 1: Video Ingestion & Adaptive Frame Selection (OpenCV5 + DAG)
        # ======================================================================
        s1_t0 = time.perf_counter()
        s1_start_utc = utc_now_iso()
        s1_cfg = {
            "target_frame_count": self.target_frame_count,
            "orb_nfeatures": 1000,
        }
        s1_fp = cache_store.build_dependency_fingerprint(
            stage_name="video_ingestion_and_frame_selection",
            stage_version="2.0.0",
            parent_hashes={"video_sha256": tracker.video_sha256},
            resolved_config=s1_cfg,
            code_fingerprint=tracker.code_fingerprint,
        )
        s1_hit = cache_store.lookup("video_ingestion_and_frame_selection", s1_fp) if self.use_dag_cache else None

        manifest_sel_path = self.output_dir / "adaptive_frame_selection_manifest.json"
        if s1_hit is not None and "payload_json" in s1_hit["files"]:
            payload = json.loads(Path(s1_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
            curated_records = payload["curated_records"]
            curation_dict = payload["curation_dict"]
            for i, rec in enumerate(curated_records):
                fkey = f"frame_{i:04d}"
                src_cached = Path(s1_hit["files"][fkey])
                dst_frame = self.frames_dir / Path(rec["file_path"]).name
                shutil.copy2(src_cached, dst_frame)
                rec["file_path"] = str(dst_frame.resolve())
                rec["frame_sha256"] = compute_file_sha256(dst_frame) or rec.get("frame_sha256", "")
            manifest_sel_path.write_text(json.dumps(curation_dict, indent=2), encoding="utf-8")
            s1_cache_status = "HIT"
            s1_status = "REUSED"
            s1_reason = f"Matched DAG fingerprint {s1_fp[:16]}"
            self.log(f"\n[STAGE 1] Video & Frame Selection: CACHE HIT ({len(curated_records)} frames, {s1_fp[:12]})")
        else:
            self.log("\n[STAGE 1] Video Ingestion & OpenCV5 Adaptive Frame Selection...")
            analyzer = OpenCV5FrameAnalyzer(orb_nfeatures=1000)
            curated_records, curation_report = analyzer.extract_curated_frames(
                video_path=str(self.video_path),
                output_dir=str(self.frames_dir),
                target_num_frames=self.target_frame_count,
                prefix="frame",
            )
            curation_dict = curation_report.to_dict()
            manifest_sel_path.write_text(json.dumps(curation_dict, indent=2), encoding="utf-8")
            file_srcs = {f"frame_{i:04d}": r["file_path"] for i, r in enumerate(curated_records)}
            cache_store.store(
                stage_name="video_ingestion_and_frame_selection",
                stage_version="2.0.0",
                fingerprint=s1_fp,
                file_sources=file_srcs,
                json_payload={"curated_records": curated_records, "curation_dict": curation_dict},
            )
            s1_cache_status = "MISS"
            s1_status = "EXECUTED"
            s1_reason = None

        curation_summary = curation_dict.get("selection_summary", {})
        image_paths = [r["file_path"] for r in curated_records]
        num_frames = len(image_paths)
        tracker.set_frame_provenance(curated_records)
        frames_digest = tracker.frame_prov.get("frames_digest_sha256", s1_fp)

        s1_dur = round(time.perf_counter() - s1_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="video_ingestion_and_frame_selection",
            stage_version="2.0.0",
            dependency_fingerprint=s1_fp,
            parent_stages=[],
            input_artifacts=[_artifact_ref("source_video", self.video_path, "video")],
            output_artifacts=[_artifact_ref("selection_manifest", manifest_sel_path, "json")],
            resolved_config=s1_cfg,
            start_time_utc=s1_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s1_dur,
            status=s1_status,
            cache_status=s1_cache_status,
            reuse_or_skip_reason=s1_reason,
            qa_metrics={
                "selected_frames": num_frames,
                "bridges_inserted": curation_summary.get("bridge_frames_inserted", 0),
                "mean_sharpness": curation_summary.get("mean_selected_laplacian_sharpness", 0.0),
                "mean_overlap_score": curation_summary.get("mean_pairwise_overlap_score", 0.0),
            },
        ))
        self.log(f"  [+] Stage 1 done in {s1_dur}s: {num_frames} frames, bridges={curation_summary.get('bridge_frames_inserted', 0)}")

        # ======================================================================
        # STAGE 2: VGGT-Omega Window Inference (8 GB VRAM Sequential + DAG Cache)
        # ======================================================================
        s2_t0 = time.perf_counter()
        s2_start_utc = utc_now_iso()
        s2_cfg = {
            "image_resolution": int(self.config.image_resolution),
            "window_size": int(self.config.window_size),
            "overlap": int(self.config.overlap),
        }
        s2_fp = cache_store.build_dependency_fingerprint(
            stage_name="vggt_window_inference",
            stage_version="2.0.0",
            parent_hashes={"frames_digest_sha256": frames_digest},
            resolved_config=s2_cfg,
            code_fingerprint=tracker.code_fingerprint,
            model_hashes={"vggt_omega": tracker.checkpoint_sha256},
        )
        s2_hit = cache_store.lookup("vggt_window_inference", s2_fp) if self.use_dag_cache else None

        engine = IndoorReconstructionEngine(self.config)
        vggt_perf_path = self.camera_dir / "vggt_performance.json"
        win_manifest_path = self.camera_dir / "vggt_window_alignment_manifest.json"

        if s2_hit is not None and "arrays_npz" in s2_hit["files"] and "payload_json" in s2_hit["files"]:
            npz = np.load(s2_hit["files"]["arrays_npz"])
            pjson = json.loads(Path(s2_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
            extrinsics_base = npz["extrinsics"].astype(np.float64)
            intrinsics_infer = npz["intrinsics"].astype(np.float64)
            depth_base = npz["depth"].astype(np.float32)
            conf_base = npz["confidence"].astype(np.float32)
            raw_windows: List[Dict[str, Any]] = []
            for w_idx, f_idxs in enumerate(pjson["raw_window_frames"]):
                kp = f"w{w_idx}_"
                raw_windows.append({
                    "window_index": int(w_idx),
                    "frame_indices": [int(x) for x in f_idxs],
                    "local_extrinsics": npz[kp + "ext"].astype(np.float64),
                    "local_intrinsics": npz[kp + "intr"].astype(np.float64),
                    "local_depth": npz[kp + "depth"].astype(np.float32),
                    "local_conf": npz[kp + "conf"].astype(np.float32),
                })
            window_joins_base = pjson["window_alignment_manifest"]
            vggt_perf = pjson["performance"]
            section_summary = pjson["section_summary"]
            s2_cache_status = "HIT"
            s2_status = "REUSED"
            s2_reason = f"Loaded VGGT raw windows & baseline predictions from DAG cache ({s2_fp[:12]})"
            self.log(f"\n[STAGE 2] VGGT-Omega Inference: CACHE HIT ({len(raw_windows)} windows)")
        else:
            self.log("\n[STAGE 2] VGGT-Omega Window Inference (8 GB VRAM sequential scheduling)...")
            window_results = engine.run_overlapping_windows(
                image_paths=image_paths,
                window_size=self.config.window_size,
                overlap=self.config.overlap,
                checkpoint_path=str(self.checkpoint_path),
                log_callback=self.log,
            )
            extrinsics_base = np.asarray(window_results["extrinsics"], dtype=np.float64)
            intrinsics_infer = np.asarray(window_results["intrinsics"], dtype=np.float64)
            depth_base = np.asarray(window_results["depth"], dtype=np.float32)
            conf_base = np.asarray(window_results["confidence"], dtype=np.float32)
            raw_windows = window_results["raw_windows"]
            window_joins_base = window_results["window_alignment_manifest"]
            vggt_perf = window_results["performance"]
            section_summary = window_results["section_summary"]

            arrays_to_cache: Dict[str, np.ndarray] = {
                "extrinsics": extrinsics_base,
                "intrinsics": intrinsics_infer,
                "depth": depth_base,
                "confidence": conf_base,
            }
            win_frames_meta: List[List[int]] = []
            for w in raw_windows:
                wi = int(w["window_index"])
                kp = f"w{wi}_"
                arrays_to_cache[kp + "ext"] = np.asarray(w["local_extrinsics"], dtype=np.float64)
                arrays_to_cache[kp + "intr"] = np.asarray(w["local_intrinsics"], dtype=np.float64)
                arrays_to_cache[kp + "depth"] = np.asarray(w["local_depth"], dtype=np.float32)
                arrays_to_cache[kp + "conf"] = np.asarray(w["local_conf"], dtype=np.float32)
                win_frames_meta.append([int(x) for x in w["frame_indices"]])

            cache_store.store(
                stage_name="vggt_window_inference",
                stage_version="2.0.0",
                fingerprint=s2_fp,
                arrays_npz=arrays_to_cache,
                json_payload={
                    "raw_window_frames": win_frames_meta,
                    "window_alignment_manifest": window_joins_base,
                    "performance": vggt_perf,
                    "section_summary": section_summary,
                },
            )
            s2_cache_status = "MISS"
            s2_status = "REUSED" if window_results.get("is_cached") else "EXECUTED"
            s2_reason = "VGGT model cache hit" if window_results.get("is_cached") else None

        # Always release any residual CUDA blocks before YOLO
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        actual_infer_h, actual_infer_w = depth_base[0].squeeze().shape[:2]
        infer_hw = (int(actual_infer_h), int(actual_infer_w))
        preproc_geom["infer_height"] = int(actual_infer_h)
        preproc_geom["infer_width"] = int(actual_infer_w)
        preproc_geom["sx_crop_to_infer"] = float(actual_infer_w) / max(1.0, float(preproc_geom["crop_box_xywh"][2]))
        preproc_geom["sy_crop_to_infer"] = float(actual_infer_h) / max(1.0, float(preproc_geom["crop_box_xywh"][3]))

        vggt_perf_path.write_text(json.dumps(vggt_perf, indent=2), encoding="utf-8")
        win_manifest_path.write_text(json.dumps(window_joins_base, indent=2), encoding="utf-8")
        tracker.set_frame_provenance(curated_records, [w["frame_indices"] for w in raw_windows])

        # Build canonical camera rig anchored strictly to preprocessed geometry
        win_membership: Dict[int, List[int]] = {i: [] for i in range(num_frames)}
        for w in raw_windows:
            for fi in w["frame_indices"]:
                win_membership.setdefault(int(fi), []).append(int(w["window_index"]))

        cam_frames: List[CanonicalCameraFrame] = []
        for i in range(num_frames):
            rec = curated_records[i]
            k_native = infer_to_native_intrinsics(intrinsics_infer[i], preproc_geom)
            cam_frames.append(CanonicalCameraFrame(
                selected_index=i,
                orig_frame_index=int(rec.get("frame_index", i)),
                timestamp_sec=float(rec.get("timestamp_sec", i * 0.5)),
                frame_path=str(rec["file_path"]),
                frame_sha256=str(rec.get("frame_sha256", "")),
                orig_width=int(preproc_geom["orig_width"]),
                orig_height=int(preproc_geom["orig_height"]),
                infer_width=int(infer_hw[1]),
                infer_height=int(infer_hw[0]),
                crop_box_xywh=[int(x) for x in preproc_geom["crop_box_xywh"]],
                sx_crop_to_infer=float(preproc_geom["sx_crop_to_infer"]),
                sy_crop_to_infer=float(preproc_geom["sy_crop_to_infer"]),
                intrinsics_infer=[[round(float(v), 6) for v in row] for row in intrinsics_infer[i]],
                intrinsics_native=[[round(float(v), 6) for v in row] for row in k_native],
                extrinsic_w2c=[[round(float(v), 7) for v in row] for row in extrinsics_base[i]],
                window_ids=win_membership.get(i, []),
                is_bridge_frame=bool(rec.get("is_bridge", False)),
                selection_score=float(rec.get("selection_score", 0.0)),
            ))

        rig = CanonicalCameraRig(
            run_id=tracker.run_id,
            scene_id=self.scene_id,
            video_sha256=tracker.video_sha256,
            frames=cam_frames,
        )
        rig.save_json(self.camera_dir / "canonical_camera_rig_initial.json")
        tracker.set_calibration_provenance(rig.to_dict())

        s2_dur = round(time.perf_counter() - s2_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="vggt_window_inference",
            stage_version="2.0.0",
            dependency_fingerprint=s2_fp,
            parent_stages=["video_ingestion_and_frame_selection"],
            input_artifacts=[_artifact_ref("selection_manifest", manifest_sel_path, "json")],
            output_artifacts=[
                _artifact_ref("vggt_performance", vggt_perf_path, "json"),
                _artifact_ref("canonical_rig_initial", self.camera_dir / "canonical_camera_rig_initial.json", "json"),
            ],
            resolved_config=s2_cfg,
            start_time_utc=s2_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s2_dur,
            status=s2_status,
            cache_status=s2_cache_status,
            reuse_or_skip_reason=s2_reason,
            qa_metrics={
                "num_windows": len(raw_windows),
                "mean_sim3_alignment_error_m": vggt_perf.get("mean_sim3_alignment_error_m", 0.0),
                "peak_vram_mb": vggt_perf.get("peak_vram_mb", 0.0),
            },
        ))
        self.log(f"  [+] Stage 2 done in {s2_dur}s: {len(raw_windows)} windows, infer_hw={infer_hw}")

        # Load RGB inference-resolution images once for stages 3, 4, 5
        frames_rgb_infer = _load_images_rgb_at_infer(image_paths, infer_hw)

        # ======================================================================
        # STAGE 3: Sequential YOLOv8n-seg Dynamic & Indoor Furniture Segmentation
        # ======================================================================
        s3_t0 = time.perf_counter()
        s3_start_utc = utc_now_iso()
        s3_cfg = {
            "enable_yolo": self.enable_yolo,
            "infer_hw": list(infer_hw),
            "return_furniture_masks": True,
        }
        s3_fp = cache_store.build_dependency_fingerprint(
            stage_name="yolo_dynamic_segmentation",
            stage_version="2.1.0",
            parent_hashes={"frames_digest_sha256": frames_digest},
            resolved_config=s3_cfg,
            code_fingerprint=tracker.code_fingerprint,
            model_hashes={"yolov8n_seg": compute_file_sha256(self.yolo_path) if self.yolo_path.is_file() else None},
        )
        yolo_stats_path = self.camera_dir / "dynamic_masks_stats.json"

        if not self.enable_yolo:
            dynamic_masks_infer: List[Optional[np.ndarray]] = [None] * num_frames
            furn_cls_masks_infer: List[np.ndarray] = [np.full(infer_hw, -1, dtype=np.int16) for _ in range(num_frames)]
            furn_inst_masks_infer: List[np.ndarray] = [np.zeros(infer_hw, dtype=np.int32) for _ in range(num_frames)]
            yolo_info = {"enabled": False, "status": "SKIPPED_BY_FLAG", "frames_with_dynamic_pixels": 0}
            s3_status, s3_cache_status, s3_reason = "SKIPPED", "MISS", "Disabled by --enable-yolo off"
        else:
            s3_hit = cache_store.lookup("yolo_dynamic_segmentation", s3_fp) if self.use_dag_cache else None
            if s3_hit is not None and "arrays_npz" in s3_hit["files"] and "payload_json" in s3_hit["files"]:
                npz_s3 = np.load(s3_hit["files"]["arrays_npz"])
                dyn_arr = npz_s3["dynamic_masks"]
                cls_arr = npz_s3["furn_cls_masks"] if "furn_cls_masks" in npz_s3 else np.full((num_frames, *infer_hw), -1, dtype=np.int16)
                inst_arr = npz_s3["furn_inst_masks"] if "furn_inst_masks" in npz_s3 else np.zeros((num_frames, *infer_hw), dtype=np.int32)
                dynamic_masks_infer = [dyn_arr[i].astype(bool) if np.any(dyn_arr[i]) else None for i in range(num_frames)]
                furn_cls_masks_infer = [cls_arr[i].astype(np.int16) for i in range(num_frames)]
                furn_inst_masks_infer = [inst_arr[i].astype(np.int32) for i in range(num_frames)]
                yolo_info = json.loads(Path(s3_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
                s3_status, s3_cache_status, s3_reason = "REUSED", "HIT", f"Loaded YOLO masks from DAG cache ({s3_fp[:12]})"
                self.log(
                    f"\n[STAGE 3] Dynamic + Furniture Segmentation: CACHE HIT "
                    f"(person_frames={yolo_info.get('frames_with_dynamic_person', 0)}, "
                    f"furniture_frames={yolo_info.get('frames_with_furniture_masks', 0)})"
                )
            else:
                self.log("\n[STAGE 3] Sequential YOLOv8n-seg Dynamic & Furniture Masking...")
                dynamic_masks_infer, furn_cls_masks_infer, furn_inst_masks_infer, yolo_info = run_yolo_indoor_segmentation_scheduled(
                    frames_rgb=frames_rgb_infer,
                    infer_hw=infer_hw,
                    model_path=self.yolo_path,
                    device="auto",
                    log_callback=self.log,
                    return_furniture_masks=True,
                )
                dyn_stack = np.stack(
                    [m.astype(np.uint8) if m is not None else np.zeros(infer_hw, dtype=np.uint8) for m in dynamic_masks_infer],
                    axis=0,
                )
                cls_stack = np.stack([m.astype(np.int16) for m in furn_cls_masks_infer], axis=0)
                inst_stack = np.stack([m.astype(np.int32) for m in furn_inst_masks_infer], axis=0)
                cache_store.store(
                    stage_name="yolo_dynamic_segmentation",
                    stage_version="2.1.0",
                    fingerprint=s3_fp,
                    arrays_npz={
                        "dynamic_masks": dyn_stack,
                        "furn_cls_masks": cls_stack,
                        "furn_inst_masks": inst_stack,
                    },
                    json_payload=yolo_info,
                )
                s3_status, s3_cache_status, s3_reason = "EXECUTED", "MISS", None

        yolo_stats_path.write_text(json.dumps(yolo_info, indent=2), encoding="utf-8")
        s3_dur = round(time.perf_counter() - s3_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="yolo_dynamic_segmentation",
            stage_version="2.1.0",
            dependency_fingerprint=s3_fp,
            parent_stages=["video_ingestion_and_frame_selection"],
            output_artifacts=[_artifact_ref("dynamic_masks_stats", yolo_stats_path, "json")],
            resolved_config=s3_cfg,
            start_time_utc=s3_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s3_dur,
            status=s3_status,
            cache_status=s3_cache_status,
            reuse_or_skip_reason=s3_reason,
            qa_metrics=yolo_info,
        ))
        self.log(f"  [+] Stage 3 done in {s3_dur}s: status={s3_status}, masked_frames={yolo_info.get('frames_with_dynamic_person', 0)}")

        # ======================================================================
        # STAGE 4: Global Pose-Graph Alignment + Held-out Validated BA (with bounded retry)
        # ======================================================================
        s4_t0 = time.perf_counter()
        s4_start_utc = utc_now_iso()
        s4_cfg = {
            "enable_pose_graph": True,
            "enable_ba": self.enable_ba,
        }
        s4_fp = cache_store.build_dependency_fingerprint(
            stage_name="global_alignment_and_validated_ba",
            stage_version="2.0.0",
            parent_hashes={"vggt_fp": s2_fp, "yolo_fp": s3_fp},
            resolved_config=s4_cfg,
            code_fingerprint=tracker.code_fingerprint,
            calibration_version=rig.calibration_version,
            pose_version=rig.pose_version,
        )
        align_report_path = self.camera_dir / "global_alignment_report.json"
        s4_hit = cache_store.lookup("global_alignment_and_validated_ba", s4_fp) if self.use_dag_cache else None

        s4_retries = 0
        if s4_hit is not None and "arrays_npz" in s4_hit["files"] and "payload_json" in s4_hit["files"]:
            npz = np.load(s4_hit["files"]["arrays_npz"])
            extrinsics = npz["extrinsics"].astype(np.float64)
            depth_maps = npz["depth"].astype(np.float32)
            confidence_maps = npz["confidence"].astype(np.float32)
            align_report_dict = json.loads(Path(s4_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
            s4_cache_status, s4_status, s4_reason = "HIT", "REUSED", f"Matched DAG fingerprint {s4_fp[:16]}"
            self.log(f"\n[STAGE 4] Global Alignment & BA: CACHE HIT (status={align_report_dict.get('status')})")
        else:
            self.log("\n[STAGE 4] Global Pose-Graph Loop Closure & Held-Out Validated BA...")
            best_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, GlobalAlignmentReport]] = None

            def _score_align(rep: GlobalAlignmentReport) -> Tuple[int, float]:
                rank = {"ALIGNMENT_PASS": 3, "ALIGNMENT_RECOVERABLE": 2, "PARTIAL_ALIGNMENT": 1, "ALIGNMENT_FAILED": 0}.get(rep.status, 0)
                med = rep.accepted_metrics.get("reproj_median_px", 99.0) if rep.accepted_metrics else 99.0
                return (rank, -float(med))

            acc_ext, acc_dep, acc_conf, align_report = run_global_alignment_and_validated_ba(
                raw_windows=raw_windows,
                images_rgb=frames_rgb_infer,
                baseline_extrinsics=extrinsics_base,
                baseline_intrinsics_infer=intrinsics_infer,
                baseline_depths=depth_base,
                baseline_confs=conf_base,
                window_joins_baseline=window_joins_base,
                dynamic_masks_infer=dynamic_masks_infer,
                enable_pose_graph=True,
                enable_ba=self.enable_ba,
                thresholds=AlignmentQAThresholds(),
                log_callback=self.log,
            )
            best_tuple = (acc_ext, acc_dep, acc_conf, align_report)

            # Bounded automatic retries (max 3 attempts total) when ALIGNMENT_RECOVERABLE
            retry_specs = [
                ("relaxed_loops", True, self.enable_ba, AlignmentQAThresholds(min_loop_inliers=16, min_loop_grid_cells=4, max_loop_sim3_median_m=0.085)),
                ("pose_graph_safe_fallback", True, False, AlignmentQAThresholds(min_loop_inliers=16, min_loop_grid_cells=4)),
            ]
            for label, pg_en, ba_en, thresh in retry_specs:
                if best_tuple[3].status != "ALIGNMENT_RECOVERABLE":
                    break
                s4_retries += 1
                self.log(f"  [!] Alignment status {best_tuple[3].status}; executing bounded recovery retry #{s4_retries} ({label})...")
                cand_ext, cand_dep, cand_conf, cand_rep = run_global_alignment_and_validated_ba(
                    raw_windows=raw_windows,
                    images_rgb=frames_rgb_infer,
                    baseline_extrinsics=extrinsics_base,
                    baseline_intrinsics_infer=intrinsics_infer,
                    baseline_depths=depth_base,
                    baseline_confs=conf_base,
                    window_joins_baseline=window_joins_base,
                    dynamic_masks_infer=dynamic_masks_infer,
                    enable_pose_graph=pg_en,
                    enable_ba=ba_en,
                    thresholds=thresh,
                    log_callback=self.log,
                )
                if _score_align(cand_rep) > _score_align(best_tuple[3]):
                    best_tuple = (cand_ext, cand_dep, cand_conf, cand_rep)

            extrinsics, depth_maps, confidence_maps, align_report = best_tuple
            align_report_dict = align_report.to_dict()
            cache_store.store(
                stage_name="global_alignment_and_validated_ba",
                stage_version="2.0.0",
                fingerprint=s4_fp,
                arrays_npz={"extrinsics": extrinsics, "depth": depth_maps, "confidence": confidence_maps},
                json_payload=align_report_dict,
            )
            s4_cache_status = "MISS"
            s4_status = "EXECUTED" if align_report_dict["status"] != "ALIGNMENT_FAILED" else "FAILED"
            s4_reason = None

        align_report_path.write_text(json.dumps(align_report_dict, indent=2), encoding="utf-8")
        rig.update_extrinsics(extrinsics, pose_version="global_pose_graph_ba_v2", world_frame_state="vggt_opencv_y_down")
        rig.save_json(self.camera_dir / "canonical_camera_rig_aligned.json")

        traj_assessment = engine.assess_indoor_trajectory(extrinsics)
        traj_report_path = self.camera_dir / "trajectory_assessment.json"
        traj_report_path.write_text(json.dumps(traj_assessment, indent=2), encoding="utf-8")

        s4_dur = round(time.perf_counter() - s4_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="global_alignment_and_validated_ba",
            stage_version="2.0.0",
            dependency_fingerprint=s4_fp,
            parent_stages=["vggt_window_inference", "yolo_dynamic_segmentation"],
            output_artifacts=[
                _artifact_ref("global_alignment_report", align_report_path, "json"),
                _artifact_ref("trajectory_assessment", traj_report_path, "json"),
            ],
            resolved_config=s4_cfg,
            start_time_utc=s4_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s4_dur,
            status=s4_status,
            cache_status=s4_cache_status,
            reuse_or_skip_reason=s4_reason,
            retry_count=s4_retries,
            qa_metrics={
                "status": align_report_dict.get("status"),
                "ba_decision": align_report_dict.get("ba_decision"),
                "loop_closure_edges": align_report_dict.get("loop_closure_edges", 0),
                "accepted_median_reproj_px": align_report_dict.get("accepted_metrics", {}).get("reproj_median_px"),
                "is_stationary_pan": traj_assessment.get("is_stationary_pan", False),
            },
        ))
        self.log(
            f"  [+] Stage 4 done in {s4_dur}s: status={align_report_dict['status']}, "
            f"ba={align_report_dict['ba_decision']}, loops={align_report_dict['loop_closure_edges']}"
        )

        # ======================================================================
        # STAGE 5: Multi-View Depth Validation, Free-Space Evidence & Floor Leveling
        # ======================================================================
        s5_t0 = time.perf_counter()
        s5_start_utc = utc_now_iso()
        s5_cfg = {
            "min_depth": float(self.config.min_depth),
            "max_depth": float(self.config.max_depth),
            "min_confidence": float(self.config.min_confidence),
            "voxel_size": float(self.config.voxel_size),
            "level_floor": bool(self.config.level_floor),
        }
        s5_fp = cache_store.build_dependency_fingerprint(
            stage_name="multiview_depth_visibility_freespace",
            stage_version="2.0.0",
            parent_hashes={"align_fp": s4_fp, "yolo_fp": s3_fp},
            resolved_config=s5_cfg,
            code_fingerprint=tracker.code_fingerprint,
            pose_version=rig.pose_version,
        )
        depth_val_report_path = self.output_dir / "depth_validation_report.json"
        planes_path = self.output_dir / "detected_room_planes.json"
        s5_hit = cache_store.lookup("multiview_depth_visibility_freespace", s5_fp) if self.use_dag_cache else None

        if s5_hit is not None and "arrays_npz" in s5_hit["files"] and "payload_json" in s5_hit["files"]:
            npz = np.load(s5_hit["files"]["arrays_npz"])
            pjson = json.loads(Path(s5_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
            clean_pts = npz["clean_pts"].astype(np.float32)
            clean_colors = npz["clean_colors"].astype(np.uint8)
            clean_conf = npz["clean_conf"].astype(np.float32)
            validated_depth_maps = npz["validated_depth_maps"].astype(np.float32)
            validated_weight_maps = npz["validated_weight_maps"].astype(np.float32)
            extrinsics = npz["extrinsics_leveled"].astype(np.float64)
            R_floor = npz["R_floor"].astype(np.float32)
            depth_val_report_dict = pjson["depth_validation_report"]
            plane_results = pjson["plane_results"]

            # Rebuild lightweight leveled FreeSpaceVoxelEvidence deterministically
            free_space_ev = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
            free_space_ev.build_from_depth_maps(
                validated_depth_maps,
                confidence_maps,
                extrinsics,
                intrinsics_infer,
                min_depth=float(self.config.min_depth),
                max_depth=float(self.config.max_depth),
                min_confidence=float(self.config.min_confidence),
                pixel_stride=4,
            )
            s5_cache_status, s5_status, s5_reason = "HIT", "REUSED", f"Matched DAG fingerprint {s5_fp[:16]}"
            self.log(f"\n[STAGE 5] Multi-View Depth & FreeSpace Validation: CACHE HIT ({len(clean_pts):,} clean pts)")
        else:
            self.log("\n[STAGE 5] Multi-View Depth Consistency, Free-Space Evidence & Floor Leveling...")
            dc_cfg = DepthConsistencyConfig(
                min_confidence=float(self.config.min_confidence),
                min_depth_m=float(self.config.min_depth),
                max_depth_m=float(self.config.max_depth),
                voxel_size_m=float(self.config.voxel_size),
                free_space_voxel_m=0.04,
            )
            depth_filter = MultiViewDepthConsistencyFilter(dc_cfg)
            val_res = depth_filter.validate_and_fuse(
                frames_rgb=frames_rgb_infer,
                depth_maps=depth_maps,
                confidence_maps=confidence_maps,
                extrinsics=extrinsics,
                intrinsics=intrinsics_infer,
                dynamic_masks_infer=dynamic_masks_infer,
                log_callback=self.log,
            )
            clean_pts = np.asarray(val_res["clean_points"], dtype=np.float32)
            clean_colors = np.asarray(val_res["clean_colors"], dtype=np.uint8)
            clean_conf = np.asarray(val_res["clean_confidences"], dtype=np.float32)
            validated_depth_maps = np.asarray(val_res["validated_depth_maps"], dtype=np.float32)
            validated_weight_maps = np.asarray(val_res["validated_weight_maps"], dtype=np.float32)
            depth_val_report_dict = val_res["report"].to_dict()

            cam_centers_pre = [camera_center_from_extrinsic(e) for e in extrinsics]
            plane_results = engine.detect_room_planes(
                clean_pts,
                distance_threshold=0.04,
                cam_centers=cam_centers_pre,
                extrinsics=extrinsics,
            )

            R_floor = np.eye(3, dtype=np.float32)
            horiz_floors = [p for p in plane_results.get("detected_planes", []) if p.get("type") == "horizontal_floor"]
            if horiz_floors and self.config.level_floor:
                best_floor = horiz_floors[0]
                R_floor = compute_floor_leveling_rotation(np.array(best_floor["normal"], dtype=np.float64)).astype(np.float32)
                clean_pts = (R_floor @ clean_pts.T).T.astype(np.float32)
                for i in range(len(extrinsics)):
                    extrinsics[i, :3, :3] = extrinsics[i, :3, :3] @ R_floor.T
                cam_centers_lvl = [camera_center_from_extrinsic(e) for e in extrinsics]
                plane_results = engine.detect_room_planes(
                    clean_pts,
                    distance_threshold=0.04,
                    cam_centers=cam_centers_lvl,
                    extrinsics=extrinsics,
                )
                self.log(f"  [+] Floor leveling applied: rotated floor normal -> [0, 1, 0]")

            free_space_ev = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
            free_space_ev.build_from_depth_maps(
                validated_depth_maps,
                confidence_maps,
                extrinsics,
                intrinsics_infer,
                min_depth=float(self.config.min_depth),
                max_depth=float(self.config.max_depth),
                min_confidence=float(self.config.min_confidence),
                pixel_stride=4,
            )

            cache_store.store(
                stage_name="multiview_depth_visibility_freespace",
                stage_version="2.0.0",
                fingerprint=s5_fp,
                arrays_npz={
                    "clean_pts": clean_pts,
                    "clean_colors": clean_colors,
                    "clean_conf": clean_conf,
                    "validated_depth_maps": validated_depth_maps,
                    "validated_weight_maps": validated_weight_maps,
                    "extrinsics_leveled": extrinsics,
                    "R_floor": R_floor,
                },
                json_payload={
                    "depth_validation_report": depth_val_report_dict,
                    "plane_results": plane_results,
                },
            )
            s5_cache_status, s5_status, s5_reason = "MISS", "EXECUTED", None

        depth_val_report_path.write_text(json.dumps(depth_val_report_dict, indent=2), encoding="utf-8")
        planes_path.write_text(json.dumps(plane_results, indent=2), encoding="utf-8")
        rig.floor_leveling_rotation = [[round(float(v), 7) for v in row] for row in R_floor]
        rig.update_extrinsics(
            extrinsics,
            pose_version="floor_leveled_world_v2",
            world_frame_state="leveled_opencv_y_down",
        )
        rig.save_json(self.camera_dir / "canonical_camera_rig.json")

        cam_centers = [camera_center_from_extrinsic(e) for e in extrinsics]
        clean_ply_path = self.mesh_dir / "clean_filtered_points.ply"
        pcd_clean_vis = o3d.geometry.PointCloud()
        pcd_clean_vis.points = o3d.utility.Vector3dVector(clean_pts.astype(np.float64))
        pcd_clean_vis.colors = o3d.utility.Vector3dVector(clean_colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(clean_ply_path), pcd_clean_vis)

        s5_dur = round(time.perf_counter() - s5_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="multiview_depth_visibility_freespace",
            stage_version="2.0.0",
            dependency_fingerprint=s5_fp,
            parent_stages=["global_alignment_and_validated_ba", "yolo_dynamic_segmentation"],
            output_artifacts=[
                _artifact_ref("depth_validation_report", depth_val_report_path, "json"),
                _artifact_ref("detected_planes", planes_path, "json"),
            ],
            resolved_config=s5_cfg,
            start_time_utc=s5_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s5_dur,
            status=s5_status,
            cache_status=s5_cache_status,
            reuse_or_skip_reason=s5_reason,
            qa_metrics={
                "clean_points": int(len(clean_pts)),
                "contradicted_pruned": depth_val_report_dict.get("contradicted_pruned", 0),
                "known_free_space_voxels": len(free_space_ev.free_keys),
            },
        ))
        self.log(f"  [+] Stage 5 done in {s5_dur}s: {len(clean_pts):,} clean points, free_voxels={len(free_space_ev.free_keys):,}")

        # Associate YOLO static furniture semantics onto cleaned 3D points cleanly before PlanarGS
        if self.enable_yolo and yolo_info.get("available"):
            point_sem_cls, point_furn_inst, sem_assoc_stats = associate_yolo_semantics_to_3d_points(
                points=clean_pts,
                extrinsics=extrinsics,
                intrinsics=intrinsics_infer,
                validated_depth_maps=validated_depth_maps,
                furn_cls_masks=furn_cls_masks_infer,
                furn_inst_masks=furn_inst_masks_infer,
                dynamic_masks=dynamic_masks_infer,
                class_id_to_name=yolo_info.get("class_id_to_name"),
                min_depth=float(self.config.min_depth),
                max_depth=float(self.config.max_depth),
            )
            self.log(
                f"  [+] 3D YOLO Semantic Association: {sem_assoc_stats['labeled_points']:,} pts "
                f"({sem_assoc_stats['labeled_fraction_pct']}%), "
                f"{sem_assoc_stats['unique_3d_furniture_instances']} 3D furniture instances, "
                f"classes={sem_assoc_stats['class_point_counts']}"
            )
        else:
            point_sem_cls = np.full(len(clean_pts), -1, dtype=np.int32)
            point_furn_inst = np.zeros(len(clean_pts), dtype=np.int32)
            sem_assoc_stats = {
                "labeled_points": 0,
                "labeled_fraction_pct": 0.0,
                "unique_3d_furniture_instances": 0,
                "class_point_counts": {},
            }
        (self.camera_dir / "semantic_association_report.json").write_text(
            json.dumps(sem_assoc_stats, indent=2), encoding="utf-8"
        )

        # ======================================================================
        # STAGE 6: Boundary-Preserving PlanarGS Refinement (OFF / ON Ablation)
        # ======================================================================
        s6_t0 = time.perf_counter()
        s6_start_utc = utc_now_iso()
        planar_cfg = PlanarRefinementConfig(
            enabled=self.enable_planar_refinement,
            voxel_size=float(self.config.voxel_size),
        )
        s6_cfg = {
            "enabled": self.enable_planar_refinement,
            "planar_config": planar_cfg.to_dict(),
        }
        s6_fp = cache_store.build_dependency_fingerprint(
            stage_name="planar_gs_refinement",
            stage_version="2.1.0",
            parent_hashes={"depth_val_fp": s5_fp, "yolo_fp": s3_fp},
            resolved_config=s6_cfg,
            code_fingerprint=tracker.code_fingerprint,
            pose_version=rig.pose_version,
        )
        planar_report_path = self.mesh_dir / "planar_refinement_report.json"

        planar_regions: List[PlanarRegionRecord] = []
        thin_structure_points = np.zeros((0, 3), dtype=np.float32)
        if not self.enable_planar_refinement:
            refined_pts = clean_pts.copy()
            refined_colors = clean_colors.copy()
            c_map_raw = yolo_info.get("class_id_to_name") or {}
            prot_cids = [int(k) for k, v in c_map_raw.items() if str(v) in PROTECTED_NON_PLANAR_NAMES]
            if prot_cids:
                thin_structure_points = refined_pts[np.isin(point_sem_cls, prot_cids)].astype(np.float32)
            planar_metrics: Dict[str, Any] = {
                "enabled": False,
                "status": "SKIPPED",
                "num_accepted_planes_final": 0,
                "points_corrected": 0,
                "furniture_protected_points": int(len(thin_structure_points)),
                "input_points": int(len(clean_pts)),
                "output_points": int(len(clean_pts)),
            }
            s6_status, s6_cache_status, s6_reason = "SKIPPED", "MISS", "Disabled by experiment toggle (--planar-refinement off)"
            self.log("\n[STAGE 6] PlanarGS Refinement: SKIPPED (--planar-refinement off)")
        else:
            s6_hit = cache_store.lookup("planar_gs_refinement", s6_fp) if self.use_dag_cache else None
            if s6_hit is not None and "arrays_npz" in s6_hit["files"] and "payload_json" in s6_hit["files"]:
                npz = np.load(s6_hit["files"]["arrays_npz"])
                pjson = json.loads(Path(s6_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
                refined_pts = npz["refined_pts"].astype(np.float32)
                refined_colors = npz["refined_colors"].astype(np.uint8)
                thin_structure_points = npz["thin_structure_points"].astype(np.float32) if "thin_structure_points" in npz else np.zeros((0, 3), dtype=np.float32)
                planar_metrics = pjson["metrics"]
                for rdict in pjson.get("regions_summary", []):
                    planar_regions.append(PlanarRegionRecord(
                        region_id=int(rdict["region_id"]),
                        semantic_label=str(rdict["semantic_label"]),
                        point_indices=np.zeros(0, dtype=np.int64),
                        normal=np.array(rdict["normal"], dtype=np.float64),
                        d_coeff=float(rdict["d_coeff"]),
                        centroid=np.array(rdict["centroid"], dtype=np.float64),
                        bbox_min=np.array(rdict["bbox_min"], dtype=np.float64),
                        bbox_max=np.array(rdict["bbox_max"], dtype=np.float64),
                        mad_before_m=float(rdict["mad_before_m"]),
                        mad_after_m=float(rdict["mad_after_m"]),
                        crossview_support_ratio=float(rdict["crossview_support_ratio"]),
                        supporting_views_count=int(rdict["supporting_views_count"]),
                        free_space_conflict_ratio=float(rdict["free_space_conflict_ratio"]),
                        accepted=bool(rdict["accepted"]),
                        reject_reason=rdict.get("reject_reason"),
                        furniture_instance_id=int(rdict.get("furniture_instance_id", 0)),
                    ))
                s6_status, s6_cache_status, s6_reason = "REUSED", "HIT", f"Loaded planar refinement from DAG cache ({s6_fp[:12]})"
                self.log(f"\n[STAGE 6] PlanarGS Refinement: CACHE HIT ({planar_metrics.get('num_accepted_planes_final', 0)} accepted planes)")
            else:
                self.log("\n[STAGE 6] Boundary-Preserving PlanarGS Refinement (depth/visibility/free-space + YOLO furniture gated)...")
                p_res = refine_scene_geometry_planar_gs(
                    points=clean_pts,
                    colors=clean_colors,
                    cam_centers=cam_centers,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics_infer,
                    depth_maps=validated_depth_maps,
                    confidence_maps=confidence_maps,
                    free_space_evidence=free_space_ev,
                    config=planar_cfg,
                    log_callback=self.log,
                    point_semantic_classes=point_sem_cls,
                    point_furniture_instance_ids=point_furn_inst,
                    class_id_to_name=yolo_info.get("class_id_to_name"),
                )
                refined_pts = np.asarray(p_res["refined_points"], dtype=np.float32)
                refined_colors = np.asarray(p_res["refined_colors"], dtype=np.uint8)
                planar_regions = p_res["regions"]
                thin_structure_points = np.asarray(p_res.get("thin_structure_points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
                planar_metrics = dict(p_res["metrics"])
                planar_metrics["enabled"] = True
                cache_store.store(
                    stage_name="planar_gs_refinement",
                    stage_version="2.1.0",
                    fingerprint=s6_fp,
                    arrays_npz={
                        "refined_pts": refined_pts,
                        "refined_colors": refined_colors,
                        "thin_structure_points": thin_structure_points,
                    },
                    json_payload={
                        "metrics": planar_metrics,
                        "regions_summary": [r.to_summary_dict() for r in planar_regions],
                    },
                )
                s6_status, s6_cache_status, s6_reason = "EXECUTED", "MISS", None

        planar_report_path.write_text(
            json.dumps({
                "metrics": planar_metrics,
                "regions": [r.to_summary_dict() for r in planar_regions],
            }, indent=2),
            encoding="utf-8",
        )
        refined_ply_path = self.mesh_dir / "refined_canonical_points.ply"
        pcd_ref_vis = o3d.geometry.PointCloud()
        pcd_ref_vis.points = o3d.utility.Vector3dVector(refined_pts.astype(np.float64))
        pcd_ref_vis.colors = o3d.utility.Vector3dVector(refined_colors.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(str(refined_ply_path), pcd_ref_vis)

        s6_dur = round(time.perf_counter() - s6_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="planar_gs_refinement",
            stage_version="2.1.0",
            dependency_fingerprint=s6_fp,
            parent_stages=["multiview_depth_visibility_freespace", "yolo_dynamic_segmentation"],
            output_artifacts=[_artifact_ref("planar_refinement_report", planar_report_path, "json")],
            resolved_config=s6_cfg,
            start_time_utc=s6_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s6_dur,
            status=s6_status,
            cache_status=s6_cache_status,
            reuse_or_skip_reason=s6_reason,
            qa_metrics=planar_metrics,
        ))

        # ======================================================================
        # STAGE 7: Canonical Scene Geometry & Depth Synchronization
        # ======================================================================
        s7_t0 = time.perf_counter()
        s7_start_utc = utc_now_iso()
        s7_cfg = {"voxel_size": float(self.config.voxel_size), "max_depth_delta_m": 0.06, "splat_radius_px": 1}
        s7_fp = cache_store.build_dependency_fingerprint(
            stage_name="canonical_geometry_and_depth_sync",
            stage_version="2.2.0",
            parent_hashes={"planar_fp": s6_fp, "depth_val_fp": s5_fp},
            resolved_config=s7_cfg,
            code_fingerprint=tracker.code_fingerprint,
        )
        synced_depths, synced_weights, sync_stats = project_canonical_points_to_depth_maps(
            canonical_points=refined_pts,
            extrinsics=extrinsics,
            intrinsics=intrinsics_infer,
            validated_depth_maps=validated_depth_maps,
            validated_weight_maps=validated_weight_maps,
            max_depth_delta_m=0.06,
            min_depth=float(self.config.min_depth),
            max_depth=float(self.config.max_depth),
            splat_radius_px=1,
        )
        geom = build_canonical_scene_geometry(
            points=refined_pts,
            colors=refined_colors,
            cam_centers=cam_centers,
            extrinsics=extrinsics,
            intrinsics=intrinsics_infer,
            synchronized_depth_maps=synced_depths,
            validated_weight_maps=synced_weights,
            confidences=clean_conf if len(clean_conf) == len(refined_pts) else None,
            voxel_size=float(self.config.voxel_size),
            planar_refinement_applied=self.enable_planar_refinement,
            free_space_evidence=free_space_ev,
            point_semantic_classes=point_sem_cls if len(point_sem_cls) == len(refined_pts) else None,
            point_furniture_instance_ids=point_furn_inst if len(point_furn_inst) == len(refined_pts) else None,
            thin_structure_points=thin_structure_points if len(thin_structure_points) > 0 else None,
        )
        s7_dur = round(time.perf_counter() - s7_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="canonical_geometry_and_depth_sync",
            stage_version="2.2.0",
            dependency_fingerprint=s7_fp,
            parent_stages=["planar_gs_refinement", "multiview_depth_visibility_freespace"],
            output_artifacts=[_artifact_ref("canonical_points_ply", refined_ply_path, "ply")],
            resolved_config=s7_cfg,
            start_time_utc=s7_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s7_dur,
            status="EXECUTED",
            cache_status="MISS",
            qa_metrics=sync_stats,
        ))
        self.log(f"\n[STAGE 7] Canonical Geometry & Depth Sync done in {s7_dur}s (synced_pct={sync_stats.get('update_fraction_pct', 0):.1f}%)")

        # ======================================================================
        # STAGE 8: Mandatory Global TSDF + Conditional Localized Poisson Repair & QA
        # ======================================================================
        s8_t0 = time.perf_counter()
        s8_start_utc = utc_now_iso()
        enable_cond_repair = (self.meshing_method not in ("tsdf", "global_tsdf_no_repair"))
        mesh_qa_cfg = MeshBranchQAConfig(
            voxel_size=float(self.config.voxel_size),
            poisson_depth=int(self.config.poisson_depth),
            density_trim_quantile=float(self.config.density_trim_quantile),
            min_cluster_faces=max(25, int(self.config.min_cluster_faces)),
            tsdf_voxel_length=float(self.config.tsdf_voxel_length),
            tsdf_trunc=float(self.config.tsdf_trunc),
            min_depth=float(self.config.min_depth),
            max_depth=float(self.config.max_depth),
            min_confidence=float(self.config.min_confidence),
            enable_conditional_repair=enable_cond_repair,
        )
        s8_cfg = {
            "meshing_method": self.meshing_method,
            "enable_conditional_repair": enable_cond_repair,
            "max_triangles": self.max_triangles,
            "qa_cfg": mesh_qa_cfg.to_dict(),
        }
        s8_fp = cache_store.build_dependency_fingerprint(
            stage_name="mesh_branches_and_evidence_qa",
            stage_version="2.2.0",
            parent_hashes={"sync_fp": s7_fp, "planar_fp": s6_fp},
            resolved_config=s8_cfg,
            code_fingerprint=tracker.code_fingerprint,
        )
        mesh_qa_summary_path = self.mesh_dir / "mesh_branches_qa_summary.json"
        cond_poisson_path = self.mesh_dir / "conditional_poisson_report.json"
        gap_cls_path = self.mesh_dir / "gap_classifications.json"
        s8_hit = cache_store.lookup("mesh_branches_and_evidence_qa", s8_fp) if self.use_dag_cache else None

        frames_bgr_native = _load_images_bgr_native(image_paths)
        if s8_hit is not None and "payload_json" in s8_hit["files"]:
            branch_summary = json.loads(Path(s8_hit["files"]["payload_json"]).read_text(encoding="utf-8"))
            selection_status = str(branch_summary["selection_status"])
            selected_method = branch_summary.get("selected_method")
            sel_src = s8_hit["files"].get("selected_mesh")
            obs_src = s8_hit["files"].get("mesh_observed")
            rep_src = s8_hit["files"].get("mesh_repaired")
            gap_src = s8_hit["files"].get("gap_classifications")
            if obs_src and Path(obs_src).is_file():
                obs_m = o3d.io.read_triangle_mesh(obs_src)
                o3d.io.write_triangle_mesh(str(self.mesh_dir / "mesh_global_tsdf_observed.ply"), obs_m)
            if rep_src and Path(rep_src).is_file():
                rep_m = o3d.io.read_triangle_mesh(rep_src)
                o3d.io.write_triangle_mesh(str(self.mesh_dir / "mesh_global_tsdf_repaired.ply"), rep_m)
            if gap_src and Path(gap_src).is_file():
                gap_cls_path.write_text(Path(gap_src).read_text(encoding="utf-8"), encoding="utf-8")
            selected_mesh = o3d.io.read_triangle_mesh(sel_src) if sel_src and Path(sel_src).is_file() else None
            if selected_mesh is not None:
                o3d.io.write_triangle_mesh(str(self.mesh_dir / "selected_mesh.ply"), selected_mesh)
            cond_poisson_path.write_text(
                json.dumps(branch_summary.get("conditional_poisson_repair", {}), indent=2),
                encoding="utf-8",
            )
            s8_status, s8_cache_status, s8_reason = "REUSED", "HIT", f"Loaded Global TSDF mesh & QA scorecard from DAG cache ({s8_fp[:12]})"
            self.log(f"\n[STAGE 8] Global TSDF & Conditional Repair: CACHE HIT (status={selection_status}, selected={selected_method})")
        else:
            self.log(f"\n[STAGE 8] Running Mandatory Global TSDF + Conditional Localized Repair (repair={enable_cond_repair})...")
            mres = run_global_tsdf_and_conditional_repair(
                geom=geom,
                frames_bgr=frames_bgr_native,
                meshes_dir=self.mesh_dir,
                cfg=mesh_qa_cfg,
                renders_dir=self.renders_dir,
                log_callback=self.log,
            )
            branch_summary = mres["summary"]
            selection_status = mres["selection_status"]
            selected_method = branch_summary.get("selected_method")
            selected_mesh = mres["selected_mesh"]

            # Preserve observed TSDF strictly for diagnostics if QA rejected all candidates
            if selected_mesh is None:
                cand_obs = mres.get("mesh_observed")
                if cand_obs is not None and len(cand_obs.triangles) > 0:
                    selected_mesh = cand_obs
                    selected_method = "global_tsdf_observed_unaccepted_diagnostic"

            if selected_mesh is not None and len(selected_mesh.triangles) > self.max_triangles:
                orig_t = len(selected_mesh.triangles)
                selected_mesh = selected_mesh.simplify_quadric_decimation(target_number_of_triangles=self.max_triangles)
                selected_mesh = _clean_mesh_topology(selected_mesh, min_cluster_faces=mesh_qa_cfg.min_cluster_faces)
                branch_summary["decimated_from_triangles"] = int(orig_t)
                branch_summary["decimated_to_triangles"] = int(len(selected_mesh.triangles))
                self.log(f"  [+] Decimated selected mesh cleanly: {orig_t:,} -> {len(selected_mesh.triangles):,} tris")

            sel_ply = self.mesh_dir / "selected_mesh.ply"
            if selected_mesh is not None:
                o3d.io.write_triangle_mesh(str(sel_ply), selected_mesh)

            cache_files = {}
            if sel_ply.is_file():
                cache_files["selected_mesh"] = sel_ply
            obs_ply = self.mesh_dir / "mesh_global_tsdf_observed.ply"
            if obs_ply.is_file():
                cache_files["mesh_observed"] = obs_ply
            rep_ply = self.mesh_dir / "mesh_global_tsdf_repaired.ply"
            if rep_ply.is_file():
                cache_files["mesh_repaired"] = rep_ply
            if gap_cls_path.is_file():
                cache_files["gap_classifications"] = gap_cls_path
            cache_store.store(
                stage_name="mesh_branches_and_evidence_qa",
                stage_version="2.2.0",
                fingerprint=s8_fp,
                file_sources=cache_files,
                json_payload=branch_summary,
            )
            cond_poisson_path.write_text(
                json.dumps(branch_summary.get("conditional_poisson_repair", {}), indent=2),
                encoding="utf-8",
            )
            s8_status = "FAILED" if selection_status == "QA_FAILED" else "EXECUTED"
            s8_cache_status = "MISS"
            s8_reason = None

        mesh_qa_summary_path.write_text(json.dumps(branch_summary, indent=2), encoding="utf-8")
        s8_dur = round(time.perf_counter() - s8_t0, 3)
        s8_artifacts = [
            _artifact_ref("mesh_qa_summary", mesh_qa_summary_path, "json"),
            _artifact_ref("conditional_poisson_report", cond_poisson_path, "json"),
            _artifact_ref("gap_classifications", gap_cls_path, "json"),
        ]
        if (self.renders_dir / "gap_diagnostics.png").is_file():
            s8_artifacts.append(_artifact_ref("gap_diagnostics_png", self.renders_dir / "gap_diagnostics.png", "png"))
        if (self.renders_dir / "before_after_repair_comparison.png").is_file():
            s8_artifacts.append(_artifact_ref("before_after_comparison_png", self.renders_dir / "before_after_repair_comparison.png", "png"))
        tracker.record_stage(StageExecutionRecord(
            stage_name="mesh_branches_and_evidence_qa",
            stage_version="2.2.0",
            dependency_fingerprint=s8_fp,
            parent_stages=["canonical_geometry_and_depth_sync"],
            output_artifacts=s8_artifacts,
            resolved_config=s8_cfg,
            start_time_utc=s8_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s8_dur,
            status=s8_status,
            cache_status=s8_cache_status,
            reuse_or_skip_reason=s8_reason,
            qa_metrics=branch_summary,
        ))
        self.log(f"  [+] Stage 8 done in {s8_dur}s: status={selection_status}, method={selected_method}")

        # ======================================================================
        # STAGE 9: Hidden-Surface Completion Audit & Opt-in Free-Space-Safe Pass
        # ======================================================================
        s9_t0 = time.perf_counter()
        s9_start_utc = utc_now_iso()
        hc_cfg = HiddenCompletionConfig(enabled=self.enable_hidden_completion)
        s9_cfg = hc_cfg.to_dict()
        s9_fp = cache_store.build_dependency_fingerprint(
            stage_name="hidden_surface_completion",
            stage_version="2.0.0",
            parent_hashes={"mesh_fp": s8_fp, "planar_fp": s6_fp, "depth_val_fp": s5_fp},
            resolved_config=s9_cfg,
            code_fingerprint=tracker.code_fingerprint,
        )
        hc_res = run_optional_hidden_surface_completion(
            observed_mesh=selected_mesh if selected_mesh is not None else o3d.geometry.TriangleMesh(),
            planar_regions=planar_regions,
            observed_points=refined_pts,
            cam_centers=cam_centers,
            free_space_evidence=free_space_ev,
            out_dir=self.mesh_dir,
            parent_hashes={"mesh_fp": s8_fp, "freespace_fp": s5_fp},
            config=hc_cfg,
            log_callback=self.log,
        )
        inferred_mesh = hc_res.get("inferred_mesh")
        hc_report_path = Path(hc_res["report_path"])
        s9_dur = round(time.perf_counter() - s9_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="hidden_surface_completion",
            stage_version="2.0.0",
            dependency_fingerprint=s9_fp,
            parent_stages=["mesh_branches_and_evidence_qa", "planar_gs_refinement"],
            output_artifacts=[_artifact_ref("hidden_completion_report", hc_report_path, "json")],
            resolved_config=s9_cfg,
            start_time_utc=s9_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s9_dur,
            status="EXECUTED" if self.enable_hidden_completion else "SKIPPED",
            cache_status="MISS",
            reuse_or_skip_reason=None if self.enable_hidden_completion else "Opt-in completion disabled by default (--hidden-completion off)",
            qa_metrics=hc_res.get("report", {}),
        ))

        # ======================================================================
        # STAGE 10: Visibility-Aware Texture Baking, 8K Atlas & GLB Export
        # ======================================================================
        s10_t0 = time.perf_counter()
        s10_start_utc = utc_now_iso()
        s10_cfg = {"atlas_size": self.atlas_size, "min_cos_theta": 0.22, "exposure_consistency": True}
        s10_fp = cache_store.build_dependency_fingerprint(
            stage_name="texture_baking_and_export",
            stage_version="2.2.0",
            parent_hashes={"mesh_fp": s8_fp, "hidden_fp": s9_fp, "yolo_fp": s3_fp},
            resolved_config=s10_cfg,
            code_fingerprint=tracker.code_fingerprint,
        )
        deliverables: Dict[str, Optional[str]] = {}
        tex_qa: Dict[str, Any] = {"glb_reload_valid": False, "out_of_range_uv_count": 0}
        rendered_previews: List[str] = []

        if selected_mesh is not None and len(selected_mesh.triangles) > 0:
            self.log(f"\n[STAGE 10] Visibility-Aware Texture Baking & Canonical GLB Export (atlas={self.atlas_size})...")
            cond_rep_meta = branch_summary.get("conditional_poisson_repair", {})
            rep_start_idx: Optional[int] = None
            if (
                branch_summary.get("selected_candidate") == "global_tsdf_repaired"
                and not branch_summary.get("decimated_to_triangles")
            ):
                obs_f_cnt = int(cond_rep_meta.get("observed_triangle_count", 0))
                if 0 < obs_f_cnt < len(selected_mesh.triangles):
                    rep_start_idx = obs_f_cnt

            baker = VisibilityAwareTextureBaker(
                mesh=selected_mesh,
                extrinsics=extrinsics,
                intrinsics_infer=intrinsics_infer,
                frame_paths=[Path(p) for p in image_paths],
                depth_maps=synced_depths,
                min_cos_theta=0.22,
                dynamic_masks_infer=dynamic_masks_infer,
            )
            frame_sharpness = [float(r.get("sharpness", 1.0)) for r in curated_records]
            best_cam_per_face, cam_scores, vis_stats = baker.compute_visibility_and_best_cameras(
                frame_sharpness=frame_sharpness,
                log_callback=self.log,
            )
            atlas_path = self.output_dir / "atlas_texture.png"
            tile_rects, atlas_stats = baker.synthesize_atlas(
                best_cam_per_face=best_cam_per_face,
                out_atlas_path=atlas_path,
                atlas_size=self.atlas_size,
            )
            deliverables, tex_qa = baker.export_canonical_deliverables(
                output_dir=self.output_dir,
                base_name="room_photographic_model",
                best_cam_per_face=best_cam_per_face,
                tile_rects=tile_rects,
                atlas_path=atlas_path,
                atlas_size=self.atlas_size,
                inferred_mesh=inferred_mesh,
                repaired_face_start_idx=rep_start_idx,
            )
            tex_qa["visibility_stats"] = vis_stats
            tex_qa["atlas_stats"] = atlas_stats

            rig.gltf_canonical_center = tex_qa["canonical_center"]
            rig.save_json(self.camera_dir / "canonical_camera_rig.json")
            tracker.set_calibration_provenance(rig.to_dict())

            # Generate visual untextured shaded + wireframe QA renders
            rendered_previews = IndoorReconstructionEngine.render_visual_previews(
                selected_mesh, self.renders_dir, prefix="selected_mesh"
            )
            s10_status = "EXECUTED" if tex_qa.get("glb_reload_valid") else "FAILED"
        else:
            s10_status = "FAILED"
            self.log("\n[STAGE 10] Texture Baking SKIPPED: no non-empty mesh available.")

        tex_qa_path = self.output_dir / "texture_baking_qa.json"
        tex_qa_path.write_text(json.dumps(tex_qa, indent=2), encoding="utf-8")
        s10_dur = round(time.perf_counter() - s10_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="texture_baking_and_export",
            stage_version="2.2.0",
            dependency_fingerprint=s10_fp,
            parent_stages=["mesh_branches_and_evidence_qa", "hidden_surface_completion"],
            output_artifacts=[
                _artifact_ref("deliverable_glb", Path(deliverables["glb"]) if deliverables.get("glb") else None, "glb"),
                _artifact_ref("texture_qa", tex_qa_path, "json"),
            ],
            resolved_config=s10_cfg,
            start_time_utc=s10_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s10_dur,
            status=s10_status,
            cache_status="MISS",
            qa_metrics={k: v for k, v in tex_qa.items() if k != "visibility_stats"},
        ))

        # ======================================================================
        # STAGE 11: Scene-Specific Navigation Metadata & Viewer Registration
        # ======================================================================
        s11_t0 = time.perf_counter()
        s11_start_utc = utc_now_iso()
        manifest_scene_id = self.scene_id if self.scene_id.startswith("v2_") else f"v2_{self.scene_id}"
        glb_rel_url = (
            self._web_url_for_path(Path(deliverables["glb"]))
            if deliverables.get("glb")
            else f"/outputs/indoor/{self.output_dir.name}/room_photographic_model.glb"
        )
        canon_ply_path = Path(deliverables["ply"]) if deliverables.get("ply") else (self.mesh_dir / "selected_mesh.ply")
        deliverable_mesh = (
            o3d.io.read_triangle_mesh(str(canon_ply_path))
            if canon_ply_path.is_file()
            else (selected_mesh if selected_mesh is not None else o3d.geometry.TriangleMesh())
        )
        canonical_center_vec = np.array(tex_qa.get("canonical_center", [0.0, 0.0, 0.0]), dtype=np.float64)

        nav_dict: Dict[str, Any] = {}
        route_status = "RECAPTURE_NEEDED"
        nav_path = self.output_dir / "scene_navigation_metadata.json"

        if len(deliverable_mesh.vertices) > 0:
            nav_metadata = derive_scene_navigation(
                mesh=deliverable_mesh,
                extrinsics=extrinsics,
                detected_planes=plane_results,
                scene_id=manifest_scene_id,
                model_glb_url=glb_rel_url,
                player_radius=0.08,
                frame_manifest=curated_records,
                floor_rotation=np.eye(3, dtype=np.float64),
                canonical_center=canonical_center_vec,
            )
            nav_dict = nav_metadata.to_dict()
            route_status = nav_metadata.route_validation.eligibility_status
            nav_path.write_text(json.dumps(nav_dict, indent=2), encoding="utf-8")

        pre_viewer_status = self._determine_honest_status(
            traj_assessment=traj_assessment,
            align_report_dict=align_report_dict,
            selection_status=selection_status,
            tex_qa=tex_qa,
            route_status=route_status,
            viewer_qa=None,
        )

        reg_path = self._register_in_viewer_manifest(
            manifest_scene_id=manifest_scene_id,
            deliverables=deliverables,
            nav_dict=nav_dict,
            deliverable_mesh=deliverable_mesh,
            pre_viewer_status=pre_viewer_status,
            selected_method=selected_method,
            num_frames=num_frames,
            run_id=tracker.run_id,
            video_sha256=tracker.video_sha256,
        )
        s11_dur = round(time.perf_counter() - s11_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="scene_navigation_and_manifest",
            stage_version="2.0.0",
            dependency_fingerprint=compute_json_sha256({"s10": s10_fp, "scene_id": manifest_scene_id}),
            parent_stages=["texture_baking_and_export"],
            output_artifacts=[_artifact_ref("scene_navigation_metadata", nav_path, "json")],
            resolved_config={"manifest_scene_id": manifest_scene_id, "register_viewer": self.register_viewer},
            start_time_utc=s11_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s11_dur,
            status="EXECUTED" if nav_dict else "FAILED",
            cache_status="MISS",
            qa_metrics={
                "route_eligibility": route_status,
                "viewer_registered": reg_path is not None,
            },
        ))

        # ======================================================================
        # STAGE 12: Live Headless Viewer Reload & Walkthrough QA
        # ======================================================================
        s12_t0 = time.perf_counter()
        s12_start_utc = utc_now_iso()
        viewer_qa_res: Dict[str, Any] = {"status": "SKIPPED"}
        viewer_qa_path = self.output_dir / "viewer_qa" / "viewer_qa_report.json"

        can_run_vqa = (
            self.run_viewer_qa
            and self.register_viewer
            and reg_path is not None
            and deliverables.get("glb")
            and DEFAULT_VIEWER_QA_SCRIPT.is_file()
        )
        if can_run_vqa:
            self.log(f"\n[STAGE 12] Running Live Headless Viewer GLB Reload & Walkthrough QA (`?scene={manifest_scene_id}`)...")
            try:
                proc = subprocess.run(
                    [
                        "node",
                        str(DEFAULT_VIEWER_QA_SCRIPT),
                        "--scene-id",
                        manifest_scene_id,
                        "--output-dir",
                        str(self.output_dir),
                        "--port",
                        "5175",
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if viewer_qa_path.is_file():
                    viewer_qa_res = json.loads(viewer_qa_path.read_text(encoding="utf-8"))
                else:
                    viewer_qa_res = {
                        "status": "FAIL",
                        "error": proc.stderr[-500:] if proc.stderr else f"exit code {proc.returncode}",
                    }
                self.log(
                    f"  [+] Viewer QA status={viewer_qa_res.get('status')} | "
                    f"walkthrough_move_m={viewer_qa_res.get('checks', {}).get('walkthrough_move_delta_m', 0)}"
                )
            except Exception as e:
                viewer_qa_res = {"status": "FAIL", "error": str(e)}
                self.log(f"  [!] Viewer QA failed to execute: {e}")
            s12_status = "EXECUTED" if viewer_qa_res.get("status") == "PASS" else "FAILED"
            s12_reason = None
        else:
            s12_status = "SKIPPED"
            s12_reason = "Skipped via --skip-viewer-qa or unregistered viewer manifest"
            self.log("\n[STAGE 12] Viewer Walkthrough QA: SKIPPED.")

        s12_dur = round(time.perf_counter() - s12_t0, 3)
        tracker.record_stage(StageExecutionRecord(
            stage_name="viewer_walkthrough_qa",
            stage_version="2.0.0",
            dependency_fingerprint=compute_json_sha256({"manifest_id": manifest_scene_id, "glb_sha": compute_file_sha256(deliverables["glb"]) if deliverables.get("glb") else None}),
            parent_stages=["scene_navigation_and_manifest"],
            output_artifacts=[_artifact_ref("viewer_qa_report", viewer_qa_path, "json")],
            resolved_config={"run_viewer_qa": self.run_viewer_qa},
            start_time_utc=s12_start_utc,
            end_time_utc=utc_now_iso(),
            duration_sec=s12_dur,
            status=s12_status,
            cache_status="MISS",
            reuse_or_skip_reason=s12_reason,
            qa_metrics=viewer_qa_res.get("checks", {"status": viewer_qa_res.get("status")}),
        ))

        # ======================================================================
        # FINAL HONEST VERDICT & PROVENANCE MANIFEST
        # ======================================================================
        final_status = self._determine_honest_status(
            traj_assessment=traj_assessment,
            align_report_dict=align_report_dict,
            selection_status=selection_status,
            tex_qa=tex_qa,
            route_status=route_status,
            viewer_qa=viewer_qa_res if can_run_vqa else None,
        )

        # Keep viewer manifest acceptance_status aligned with final honest status
        if reg_path is not None and final_status != pre_viewer_status:
            self._register_in_viewer_manifest(
                manifest_scene_id=manifest_scene_id,
                deliverables=deliverables,
                nav_dict=nav_dict,
                deliverable_mesh=deliverable_mesh,
                pre_viewer_status=final_status,
                selected_method=selected_method,
                num_frames=num_frames,
                run_id=tracker.run_id,
                video_sha256=tracker.video_sha256,
            )

        qa_summary = {
            "final_status": final_status,
            "alignment": align_report_dict,
            "trajectory": traj_assessment,
            "yolo_segmentation": yolo_info,
            "semantic_association": sem_assoc_stats,
            "depth_validation": depth_val_report_dict,
            "planar_refinement": planar_metrics,
            "mesh_branches": branch_summary,
            "hidden_completion": hc_res.get("report", {}),
            "texture_qa": {k: v for k, v in tex_qa.items() if k != "visibility_stats"},
            "route_eligibility": route_status,
            "viewer_qa": viewer_qa_res,
        }

        prev_manifest = tracker.latest_preceding_manifest()
        manifest_path, run_manifest = tracker.finalize_and_write(
            output_paths={
                **{k: v for k, v in deliverables.items() if v},
                "mesh_observed_ply": str((self.mesh_dir / "mesh_global_tsdf_observed.ply").resolve()),
                "mesh_repaired_ply": str((self.mesh_dir / "mesh_global_tsdf_repaired.ply").resolve()) if (self.mesh_dir / "mesh_global_tsdf_repaired.ply").is_file() else None,
                "gap_classifications": str(gap_cls_path.resolve()) if gap_cls_path.is_file() else None,
                "gap_diagnostics_png": str((self.renders_dir / "gap_diagnostics.png").resolve()) if (self.renders_dir / "gap_diagnostics.png").is_file() else None,
                "before_after_comparison_png": str((self.renders_dir / "before_after_repair_comparison.png").resolve()) if (self.renders_dir / "before_after_repair_comparison.png").is_file() else None,
                "canonical_camera_rig": str((self.camera_dir / "canonical_camera_rig.json").resolve()),
                "navigation_metadata": str(nav_path.resolve()) if nav_path.is_file() else None,
                "viewer_qa_report": str(viewer_qa_path.resolve()) if viewer_qa_path.is_file() else None,
            },
            qa_summary=qa_summary,
            final_status=final_status,
        )
        run_cmp = compare_run_manifests(run_manifest, prev_manifest)
        (self.output_dir / "run_comparison_latest.json").write_text(json.dumps(run_cmp, indent=2), encoding="utf-8")

        total_time = round(time.perf_counter() - t_pipeline_start, 2)
        master_meta = {
            "pipeline": "F.O.R.G.E. V2 Evidence-Validated Indoor Reconstruction",
            "run_id": tracker.run_id,
            "scene_id": self.scene_id,
            "manifest_scene_id": manifest_scene_id,
            "final_status": final_status,
            "total_duration_sec": total_time,
            "qa_summary": qa_summary,
            "deliverables": deliverables,
            "rendered_previews": rendered_previews,
            "run_manifest_path": str(manifest_path),
        }
        (self.output_dir / "reconstruction_metadata.json").write_text(json.dumps(master_meta, indent=2), encoding="utf-8")
        self._write_markdown_report(master_meta, run_manifest, run_cmp)

        self.log("=" * 80)
        self.log(f"[+] V2 PIPELINE FINISHED in {total_time}s | FINAL STATUS: {final_status}")
        self.log(f"    Run Manifest: {manifest_path}")
        self.log(f"    Deliverable GLB: {deliverables.get('glb')}")
        self.log("=" * 80)
        return master_meta

    @staticmethod
    def _determine_honest_status(
        traj_assessment: Dict[str, Any],
        align_report_dict: Dict[str, Any],
        selection_status: str,
        tex_qa: Dict[str, Any],
        route_status: str,
        viewer_qa: Optional[Dict[str, Any]],
    ) -> str:
        """Maps evidence measurements honestly to PASS | PARTIAL_MODEL | QA_FAILED | RECAPTURE_NEEDED."""
        if traj_assessment.get("is_stationary_pan"):
            return "RECAPTURE_NEEDED"
        al_status = align_report_dict.get("status", "ALIGNMENT_FAILED")
        if al_status == "ALIGNMENT_FAILED":
            return "QA_FAILED"
        if selection_status == "QA_FAILED":
            return "QA_FAILED"
        if not tex_qa.get("glb_reload_valid", False) or int(tex_qa.get("out_of_range_uv_count", 0)) > 0:
            return "QA_FAILED"
        if viewer_qa is not None and viewer_qa.get("status") != "PASS":
            return "QA_FAILED"
        if (
            selection_status == "PARTIAL_MODEL"
            or al_status in ("ALIGNMENT_RECOVERABLE", "PARTIAL_ALIGNMENT")
            or route_status != "WALKTHROUGH_READY"
        ):
            return "PARTIAL_MODEL"
        return "PASS"

    def _write_markdown_report(
        self,
        meta: Dict[str, Any],
        manifest: Dict[str, Any],
        run_cmp: Dict[str, Any],
    ) -> None:
        qa = meta["qa_summary"]
        al = qa["alignment"]
        pl = qa["planar_refinement"]
        mb = qa["mesh_branches"]
        sel_cand = mb.get("selected_candidate") or "global_tsdf_observed"
        sel_qa = mb.get("candidates", {}).get(sel_cand, {})
        cond_p = mb.get("conditional_poisson_repair", {})

        lines = [
            f"# F.O.R.G.E. V2 Reconstruction Report — {meta['scene_id']}",
            f"**Final Status:** `{meta['final_status']}` | **Run ID:** `{meta['run_id']}` | **Total Time:** {meta['total_duration_sec']}s",
            "",
            "## 1. Stage DAG Cache & Timing Summary",
            "| Stage | Status | Cache | Duration (s) |",
            "| :--- | :--- | :--- | ---: |",
        ]
        for st in manifest.get("stages", []):
            lines.append(f"| `{st['stage_name']}` | `{st['status']}` | `{st['cache_status']}` | {st['duration_sec']:.2f} |")

        lines.extend([
            "",
            "## 2. Alignment, Loop Closure & Held-Out BA Gate",
            f"- **Alignment Status:** `{al.get('status')}` | **BA Decision:** `{al.get('ba_decision')}` ({al.get('ba_reason')})",
            f"- **Loop Closures:** {al.get('loop_closure_edges', 0)} edges | **Held-out Tracks:** {al.get('heldout_tracks', 0)}",
            f"- **Accepted Median Reproj Error:** {al.get('accepted_metrics', {}).get('reproj_median_px', 'N/A')} px",
            "",
            "## 3. PlanarGS Refinement & Multi-View Depth Validation",
            f"- **Clean Fused Points:** {qa['depth_validation'].get('total_retained_points', 0):,} | **Contradicted Pruned:** {qa['depth_validation'].get('contradicted_pruned', 0):,}",
            f"- **Planar Refinement:** enabled=`{pl.get('enabled', False)}`, accepted_planes={pl.get('num_accepted_planes_final', 0)}, refined_points={pl.get('points_corrected', 0):,}, furniture_protected={pl.get('furniture_protected_points', 0):,}",
            "",
            "## 4. Global TSDF & Conditional Localized Poisson QA",
            f"- **Selection Status:** `{mb.get('selection_status')}` | **Winner:** `{sel_cand}` (`{mb.get('selected_method')}`)",
            f"- **Rationale:** {mb.get('selection_rationale')}",
            f"- **Triangles:** {sel_qa.get('triangle_count', 0):,} | **Obs Support:** {sel_qa.get('obs_support_ratio', 0):.3f} | **Free-Space Violations:** {sel_qa.get('free_space_violation_ratio', 0):.4f} | **Double Shell:** {sel_qa.get('double_shell_ratio', 0):.4f} | **Local Normal Median:** {sel_qa.get('local_normal_median_deg', 0):.2f} deg",
            f"- **Flatness MAD:** wall={sel_qa.get('wall_flatness_mad_mm', 'N/A')} mm, floor={sel_qa.get('floor_flatness_mad_mm', 'N/A')} mm | **Holes:** {sel_qa.get('hole_count', 0)} ({sel_qa.get('hole_area_m2', 0):.3f} m2)",
            f"- **Conditional Repair:** B2_poisson=`{cond_p.get('localized_poisson_executed', False)}`, B1_planar=`{cond_p.get('b1_planar_triggered', False)}`, decision=`{cond_p.get('repair_acceptance_decision', 'BYPASSED')}`, b1_accepted={cond_p.get('b1_patches_accepted', 0)}, b2_accepted={cond_p.get('b2_poisson_patches_accepted', 0)} | small_filled={cond_p.get('small_holes_repaired', 0)} | preserved={len(cond_p.get('case_c_preserved_gaps', []))} | rejected_patch_qa={cond_p.get('rejected_per_patch_qa_count', 0)}",
            "",
            "## 5. Texture, Route Validation & Viewer QA",
            f"- **GLB Reload Valid:** `{qa['texture_qa'].get('glb_reload_valid')}` | **OOR UVs:** {qa['texture_qa'].get('out_of_range_uv_count', 0)} | **Exposure Range:** {qa['texture_qa'].get('atlas_stats', {}).get('exposure_consistency', {}).get('gain_min', 1.0)}–{qa['texture_qa'].get('atlas_stats', {}).get('exposure_consistency', {}).get('gain_max', 1.0)}",
            f"- **Route Eligibility:** `{qa['route_eligibility']}`",
            f"- **Viewer Walkthrough QA:** `{qa['viewer_qa'].get('status')}`",
            f"- **Latest Preceding Run Diff:** `{run_cmp.get('status', 'N/A')}`",
        ])
        (self.output_dir / "reconstruction_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="F.O.R.G.E. V2 Evidence-Validated Indoor Reconstruction")
    parser.add_argument("--video", type=str, required=True, help="Path to continuous indoor video")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--target-frames", type=int, default=90, help="Target curated frame count (default: 90)")
    parser.add_argument("--scene-id", type=str, default=None, help="Scene ID")
    parser.add_argument(
        "--meshing",
        type=str,
        default="global_tsdf",
        choices=["auto", "global_tsdf", "global_tsdf_no_repair", "conditional_tsdf", "tsdf", "poisson"],
        help="Global TSDF + conditional localized Poisson repair mode (default: global_tsdf)",
    )
    parser.add_argument("--planar-refinement", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--hidden-completion", type=str, default="off", choices=["on", "off"])
    parser.add_argument("--enable-ba", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--enable-yolo", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--register-viewer", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--atlas-size", type=int, default=8192)
    parser.add_argument("--max-triangles", type=int, default=450000)
    parser.add_argument("--reuse-cached-cameras", action="store_true", help="Backward-compatible flag keeping DAG cache enabled")
    parser.add_argument("--no-cache", action="store_true", help="Disable DAG stage & VGGT cache lookups")
    parser.add_argument("--skip-viewer-qa", action="store_true", help="Skip headless Edge/Three.js viewer QA")
    args = parser.parse_args()

    pipeline = GeneralizedIndoorReconstructionPipeline(
        video_path=Path(args.video),
        output_dir=Path(args.output),
        target_frame_count=args.target_frames,
        scene_id=args.scene_id,
        meshing_method=args.meshing,
        enable_planar_refinement=(args.planar_refinement == "on"),
        enable_hidden_completion=(args.hidden_completion == "on"),
        enable_ba=(args.enable_ba == "on"),
        enable_yolo=(args.enable_yolo == "on"),
        use_dag_cache=(False if args.no_cache else True),
        register_viewer=(args.register_viewer == "on"),
        run_viewer_qa=(not args.skip_viewer_qa),
        atlas_size=args.atlas_size,
        max_triangles=args.max_triangles,
    )
    pipeline.run(reuse_cached_cameras=args.reuse_cached_cameras)


if __name__ == "__main__":
    main()
