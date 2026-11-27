"""
F.O.R.G.E. V2 — Run Traceability, Dependency-Aware DAG Caching & Run Comparator.

Guarantees:
1. Globally unique `run_id` independent of filename or timestamp.
2. Complete machine-readable `run_manifest.json` capturing input, frame, camera,
   implementation (git commit + uncommitted diff + code hashes + checkpoints),
   hardware (GPU VRAM + system RAM), and per-stage execution records.
3. Dependency-aware stage cache (`DependencyCacheStore`) keyed by parent artifact
   hashes, frame hashes, calibration/pose versions, resolved config, and code/model
   fingerprints with SHA-256 verification on reload.
4. Automated run-to-run comparison diagnosing unchanged vs regenerated deliverables.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


DEFAULT_CODE_FILES_FOR_FINGERPRINT = [
    "scripts/run_generalized_indoor_reconstruction.py",
    "scripts/verify_v2_viewer_qa.cjs",
    "src/video/opencv5_frame_analyzer.py",
    "src/vggt/inference.py",
    "src/vggt/vggt_cache.py",
    "src/reconstruction/canonical_camera.py",
    "src/reconstruction/run_provenance.py",
    "src/reconstruction/indoor_pipeline.py",
    "src/reconstruction/window_alignment_lba.py",
    "src/reconstruction/global_alignment_ba.py",
    "src/reconstruction/depth_consistency_filter.py",
    "src/reconstruction/planar_refinement.py",
    "src/reconstruction/mesh_branches_and_qa.py",
    "src/reconstruction/hidden_surface_completion.py",
    "src/reconstruction/visibility_texture_baker.py",
    "src/reconstruction/hybrid_mesher.py",
    "src/reconstruction/point_cloud.py",
    "src/reconstruction/scene_navigation_generator.py",
]


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def generate_unique_run_id(prefix: str = "forge_v2") -> str:
    """Generates a globally unique run_id independent of output filename or second-level collisions."""
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{ts}_{uid}"


def compute_file_sha256(filepath: Union[str, Path], chunk_size: int = 4 * 1024 * 1024) -> Optional[str]:
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def compute_json_sha256(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_arrays_sha256(arrays: Dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays.keys()):
        arr = np.ascontiguousarray(arrays[k])
        h.update(k.encode("utf-8"))
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.view(np.uint8).tobytes()[:1024 * 1024] if arr.nbytes > 64 * 1024 * 1024 else arr.tobytes())
        if arr.nbytes > 64 * 1024 * 1024:
            # Also hash trailing + strided sample bytes + byte size for huge arrays without OOM
            h.update(str(arr.nbytes).encode("utf-8"))
            h.update(arr.view(np.uint8).reshape(-1)[::max(1, arr.nbytes // (512 * 1024))].tobytes())
    return h.hexdigest()


def collect_git_and_code_provenance(
    repo_root: Union[str, Path],
    code_files: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Records commit SHA, dirty status, diff hash, and per-file code hashes."""
    root = Path(repo_root)
    files = list(code_files or DEFAULT_CODE_FILES_FOR_FINGERPRINT)

    def _run_git(args: List[str]) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=8,
            )
            if res.returncode == 0:
                return res.stdout
        except Exception:
            return None
        return None

    commit_hash = (_run_git(["rev-parse", "HEAD"]) or "").strip() or None
    branch_name = (_run_git(["rev-parse", "--abbrev-ref", "HEAD"]) or "").strip() or None
    status_out = _run_git(["status", "--porcelain"])
    is_dirty = bool(status_out and status_out.strip()) if status_out is not None else None
    diff_out = _run_git(["diff", "HEAD"])
    diff_sha256 = hashlib.sha256(diff_out.encode("utf-8", errors="replace")).hexdigest() if diff_out else None

    file_hashes: Dict[str, Optional[str]] = {}
    h_code = hashlib.sha256()
    for rel in sorted(files):
        fp = root / rel
        fh = compute_file_sha256(fp)
        file_hashes[rel] = fh
        h_code.update(f"{rel}:{fh or 'MISSING'}\n".encode("utf-8"))

    # Note: viewer data manifest edits (model-manifest.json) must not invalidate upstream DAG caches;
    # per-file SHA-256 hashes of all source modules above deterministically capture tracked and untracked code changes.
    code_diff_out = _run_git(["diff", "HEAD", "--", *files]) if files else None
    code_diff_sha256 = hashlib.sha256(code_diff_out.encode("utf-8", errors="replace")).hexdigest() if code_diff_out else None
    if code_diff_sha256:
        h_code.update(f"code_diff:{code_diff_sha256}".encode("utf-8"))

    return {
        "repo_root": str(root.resolve()),
        "git_commit": commit_hash,
        "git_branch": branch_name,
        "working_tree_dirty": is_dirty,
        "uncommitted_diff_sha256": diff_sha256,
        "code_fingerprint": h_code.hexdigest(),
        "source_file_hashes": file_hashes,
    }


def collect_hardware_and_runtime_provenance() -> Dict[str, Any]:
    """Queries CPU, GPU VRAM, system RAM, Python, and library versions."""
    ram_total_gb: Optional[float] = None
    ram_avail_gb: Optional[float] = None
    try:
        class _MEM(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = _MEM()
        mem.dwLength = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            ram_total_gb = round(mem.ullTotalPhys / 1e9, 2)
            ram_avail_gb = round(mem.ullAvailPhys / 1e9, 2)
    except Exception:
        pass

    gpu_name = None
    vram_total_mb = 0.0
    vram_free_mb = 0.0
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_total_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
        try:
            free_b, _ = torch.cuda.mem_get_info(0)
            vram_free_mb = round(free_b / (1024 ** 2), 2)
        except Exception:
            pass

    deps: Dict[str, str] = {}
    for pkg in ("numpy", "torch", "cv2", "open3d", "scipy", "trimesh", "pycolmap", "ultralytics", "sklearn", "skimage"):
        try:
            m = __import__(pkg)
            deps[pkg] = getattr(m, "__version__", "unknown")
        except Exception:
            deps[pkg] = "not_installed"

    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "system_ram_total_gb": ram_total_gb,
        "system_ram_available_gb": ram_avail_gb,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "gpu_name": gpu_name,
        "gpu_vram_total_mb": vram_total_mb,
        "gpu_vram_free_at_start_mb": vram_free_mb,
        "dependency_versions": deps,
    }


@dataclass
class StageArtifactRef:
    artifact_id: str
    path: Optional[str] = None
    sha256: Optional[str] = None
    kind: str = "file"


@dataclass
class StageExecutionRecord:
    stage_name: str
    stage_version: str
    dependency_fingerprint: str
    parent_stages: List[str] = field(default_factory=list)
    input_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    output_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    resolved_config: Dict[str, Any] = field(default_factory=dict)
    start_time_utc: str = ""
    end_time_utc: str = ""
    duration_sec: float = 0.0
    status: str = "PENDING"  # EXECUTED | REUSED | SKIPPED | FAILED | REJECTED
    cache_status: str = "MISS"  # HIT | MISS | BYPASSED | SKIPPED | INVALIDATED
    reuse_or_skip_reason: Optional[str] = None
    retry_count: int = 0
    qa_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DependencyCacheStore:
    """
    Dependency-aware content-addressed artifact cache for F.O.R.G.E. V2 stages.
    Never returns a cached entry unless all parent hashes, frame hashes, calibration/pose
    versions, resolved config, model checkpoints, and stage code versions match.
    """

    def __init__(self, cache_root: Union[str, Path]):
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_dependency_fingerprint(
        stage_name: str,
        stage_version: str,
        parent_hashes: Dict[str, Any],
        resolved_config: Dict[str, Any],
        code_fingerprint: str,
        model_hashes: Optional[Dict[str, Optional[str]]] = None,
        calibration_version: str = "v2",
        pose_version: str = "v2",
    ) -> str:
        payload = {
            "stage_name": stage_name,
            "stage_version": stage_version,
            "parent_hashes": parent_hashes,
            "resolved_config": resolved_config,
            "code_fingerprint": code_fingerprint,
            "model_hashes": model_hashes or {},
            "calibration_version": calibration_version,
            "pose_version": pose_version,
        }
        return compute_json_sha256(payload)

    def entry_dir(self, stage_name: str, fingerprint: str) -> Path:
        return self.cache_root / stage_name / fingerprint[:24]

    def lookup(self, stage_name: str, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Loads cached stage metadata and verifies SHA-256 of every referenced payload file.
        Returns None on cache miss or integrity failure.
        """
        edir = self.entry_dir(stage_name, fingerprint)
        meta_path = edir / "stage_cache_meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dependency_fingerprint") != fingerprint:
                return None
            files = meta.get("cached_files", {})
            for name, info in files.items():
                fp = edir / info["rel_path"]
                actual_sha = compute_file_sha256(fp)
                if actual_sha is None or actual_sha != info.get("sha256"):
                    return None
            resolved_paths = {name: str((edir / info["rel_path"]).resolve()) for name, info in files.items()}
            return {
                "meta": meta,
                "files": resolved_paths,
                "entry_dir": str(edir.resolve()),
            }
        except Exception:
            return None

    def store(
        self,
        stage_name: str,
        stage_version: str,
        fingerprint: str,
        file_sources: Optional[Dict[str, Union[str, Path]]] = None,
        arrays_npz: Optional[Dict[str, np.ndarray]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stores immutable stage outputs with SHA-256 integrity records."""
        edir = self.entry_dir(stage_name, fingerprint)
        edir.mkdir(parents=True, exist_ok=True)
        cached_files: Dict[str, Dict[str, str]] = {}

        if file_sources:
            for key, src in file_sources.items():
                src_p = Path(src)
                if not src_p.is_file():
                    continue
                rel = src_p.name if key == src_p.stem else f"{key}{src_p.suffix}"
                dst_p = edir / rel
                if src_p.resolve() != dst_p.resolve():
                    shutil.copy2(src_p, dst_p)
                cached_files[key] = {
                    "rel_path": rel,
                    "sha256": compute_file_sha256(dst_p) or "",
                }

        if arrays_npz:
            npz_rel = "arrays.npz"
            npz_p = edir / npz_rel
            np.savez_compressed(npz_p, **arrays_npz)
            cached_files["arrays_npz"] = {
                "rel_path": npz_rel,
                "sha256": compute_file_sha256(npz_p) or "",
            }

        if json_payload is not None:
            jp_rel = "payload.json"
            jp_p = edir / jp_rel
            jp_p.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
            cached_files["payload_json"] = {
                "rel_path": jp_rel,
                "sha256": compute_file_sha256(jp_p) or "",
            }

        meta = {
            "stage_name": stage_name,
            "stage_version": stage_version,
            "dependency_fingerprint": fingerprint,
            "created_utc": utc_now_iso(),
            "cached_files": cached_files,
        }
        (edir / "stage_cache_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta


def compare_run_manifests(
    current_manifest: Dict[str, Any],
    previous_manifest: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compares the current run manifest against the most recent comparable run
    on the same source video or logical scene. Flags unexplained identical outputs.
    """
    if previous_manifest is None:
        return {
            "has_previous_run": False,
            "previous_run_id": None,
            "note": "No preceding comparable run found for this video/scene.",
            "diagnostic_warning": None,
        }

    curr_inp = current_manifest.get("input_provenance", {})
    prev_inp = previous_manifest.get("input_provenance", {})

    curr_frm = current_manifest.get("frame_provenance", {})
    prev_frm = previous_manifest.get("frame_provenance", {})

    curr_imp = current_manifest.get("implementation_provenance", {})
    prev_imp = previous_manifest.get("implementation_provenance", {})

    curr_exe = current_manifest.get("execution_provenance", {})
    prev_exe = previous_manifest.get("execution_provenance", {})

    curr_fin = current_manifest.get("final_provenance", {})
    prev_fin = previous_manifest.get("final_provenance", {})

    curr_frame_hashes = [f.get("frame_sha256") for f in curr_frm.get("frames", [])]
    prev_frame_hashes = [f.get("frame_sha256") for f in prev_frm.get("frames", [])]

    curr_glb_sha = (curr_fin.get("output_hashes") or {}).get("glb")
    prev_glb_sha = (prev_fin.get("output_hashes") or {}).get("glb")

    input_changed = curr_inp.get("video_sha256") != prev_inp.get("video_sha256")
    frames_changed = curr_frame_hashes != prev_frame_hashes
    config_changed = curr_exe.get("configuration_hash") != prev_exe.get("configuration_hash")
    code_changed = curr_imp.get("code_fingerprint") != prev_imp.get("code_fingerprint")
    ckpt_changed = curr_imp.get("vggt_checkpoint_sha256") != prev_imp.get("vggt_checkpoint_sha256")

    material_upstream_change = any([input_changed, frames_changed, config_changed, code_changed, ckpt_changed])
    glb_unchanged = (curr_glb_sha is not None) and (curr_glb_sha == prev_glb_sha)

    curr_stages = {s["stage_name"]: s for s in current_manifest.get("stages", [])}
    prev_stages = {s["stage_name"]: s for s in previous_manifest.get("stages", [])}
    stage_deltas = {}
    for sname in sorted(set(curr_stages) | set(prev_stages)):
        cs = curr_stages.get(sname, {})
        ps = prev_stages.get(sname, {})
        stage_deltas[sname] = {
            "curr_status": cs.get("status"),
            "curr_cache_status": cs.get("cache_status"),
            "curr_duration_sec": cs.get("duration_sec"),
            "prev_duration_sec": ps.get("duration_sec"),
            "fingerprint_changed": cs.get("dependency_fingerprint") != ps.get("dependency_fingerprint"),
        }

    diag_warn = None
    if glb_unchanged and material_upstream_change:
        diag_warn = (
            "DIAGNOSTIC_WARNING: Final GLB SHA-256 is identical to preceding run despite material "
            f"upstream change (input={input_changed}, frames={frames_changed}, config={config_changed}, "
            f"code={code_changed}, ckpt={ckpt_changed}). Investigate potential stale cache or non-operative parameter."
        )

    return {
        "has_previous_run": True,
        "previous_run_id": previous_manifest.get("run_id"),
        "input_sha256_changed": input_changed,
        "selected_frames_changed": frames_changed,
        "configuration_hash_changed": config_changed,
        "code_fingerprint_changed": code_changed,
        "checkpoint_sha256_changed": ckpt_changed,
        "curr_glb_sha256": curr_glb_sha,
        "prev_glb_sha256": prev_glb_sha,
        "glb_identical_to_previous": glb_unchanged,
        "glb_identical_expected": bool(glb_unchanged and not material_upstream_change),
        "cache_hits": curr_exe.get("cache_hits", 0),
        "cache_misses": curr_exe.get("cache_misses", 0),
        "curr_total_duration_sec": curr_exe.get("total_duration_sec"),
        "prev_total_duration_sec": prev_exe.get("total_duration_sec"),
        "stage_deltas": stage_deltas,
        "qa_comparison": {
            "current": curr_fin.get("qa_summary"),
            "previous": prev_fin.get("qa_summary"),
        },
        "diagnostic_warning": diag_warn,
    }


class RunProvenanceTracker:
    """Builds and persists the authoritative `run_manifest.json` and run history index."""

    def __init__(
        self,
        repo_root: Union[str, Path],
        output_dir: Union[str, Path],
        video_path: Union[str, Path],
        scene_id: str,
        resolved_config: Dict[str, Any],
        checkpoint_path: Optional[Union[str, Path]] = None,
        extra_asset_paths: Optional[Dict[str, Union[str, Path]]] = None,
        cache_root: Optional[Union[str, Path]] = None,
        run_history_dir: Optional[Union[str, Path]] = None,
        code_files_override: Optional[Sequence[str]] = None,
    ):
        self.repo_root = Path(repo_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = Path(video_path)
        self.scene_id = scene_id
        self.run_id = generate_unique_run_id()
        self.start_utc = utc_now_iso()
        self._t0 = time.perf_counter()

        default_cache = self.repo_root / "user_workspace" / "cache" / "v2_dag_cache"
        self.cache_store = DependencyCacheStore(cache_root or default_cache)
        self.run_history_dir = Path(run_history_dir or (self.repo_root / "user_workspace" / "run_history"))
        self.run_history_dir.mkdir(parents=True, exist_ok=True)

        self.resolved_config = resolved_config
        self.config_hash = compute_json_sha256(resolved_config)

        self.video_sha256 = compute_file_sha256(self.video_path) or "missing"
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = compute_file_sha256(checkpoint_path) if checkpoint_path else None

        extra_hashes: Dict[str, Optional[str]] = {}
        if extra_asset_paths:
            for k, p in extra_asset_paths.items():
                extra_hashes[k] = compute_file_sha256(p)

        self.impl_prov = collect_git_and_code_provenance(self.repo_root, code_files_override)
        self.impl_prov["vggt_model_version"] = "VGGTOmega"
        self.impl_prov["vggt_checkpoint_path"] = self.checkpoint_path
        self.impl_prov["vggt_checkpoint_sha256"] = self.checkpoint_sha256
        self.impl_prov["extra_asset_hashes"] = extra_hashes

        self.hw_prov = collect_hardware_and_runtime_provenance()
        self.input_prov: Dict[str, Any] = {
            "run_id": self.run_id,
            "scene_id": scene_id,
            "video_logical_id": self.video_path.stem,
            "video_path": str(self.video_path.resolve()) if self.video_path.exists() else str(self.video_path),
            "video_sha256": self.video_sha256,
            "resolution_wh": None,
            "fps": None,
            "duration_sec": None,
            "total_frames": None,
            "extra_asset_hashes": extra_hashes,
            "capture_metadata": {},
        }
        self.frame_prov: Dict[str, Any] = {"num_selected": 0, "frames": [], "windows": []}
        self.calibration_prov: Dict[str, Any] = {}
        self.stages: List[StageExecutionRecord] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    @property
    def code_fingerprint(self) -> str:
        return self.impl_prov["code_fingerprint"]

    def set_video_metadata(
        self,
        width: int,
        height: int,
        fps: float,
        duration_sec: float,
        total_frames: int,
        capture_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.input_prov["resolution_wh"] = [int(width), int(height)]
        self.input_prov["fps"] = round(float(fps), 4)
        self.input_prov["duration_sec"] = round(float(duration_sec), 3)
        self.input_prov["total_frames"] = int(total_frames)
        if capture_metadata:
            self.input_prov["capture_metadata"] = capture_metadata

    def set_frame_provenance(
        self,
        frame_records: List[Dict[str, Any]],
        windows: Optional[List[List[int]]] = None,
    ):
        """Records selected frame IDs, timestamps, dims, content SHA-256, scores, windows."""
        enriched = []
        win_map: Dict[int, List[int]] = {}
        if windows:
            for w_idx, idxs in enumerate(windows):
                for fi in idxs:
                    win_map.setdefault(fi, []).append(w_idx)

        for i, rec in enumerate(frame_records):
            fpath = rec.get("file_path")
            fsha = rec.get("frame_sha256") or (compute_file_sha256(fpath) if fpath else None)
            enriched.append({
                "selected_index": i,
                "orig_frame_index": rec.get("frame_index"),
                "timestamp_sec": rec.get("timestamp_sec"),
                "dimensions_wh": rec.get("dimensions_wh"),
                "file_path": fpath,
                "frame_sha256": fsha,
                "selection_score": rec.get("selection_score", rec.get("sharpness")),
                "decision": rec.get("decision"),
                "is_bridge": bool(rec.get("is_bridge", False)),
                "is_transition": bool(rec.get("is_transition", False)),
                "window_ids": win_map.get(i, []),
            })
        self.frame_prov = {
            "num_selected": len(enriched),
            "frames_digest_sha256": compute_json_sha256([e["frame_sha256"] for e in enriched]),
            "frames": enriched,
            "windows": windows or [],
        }

    def set_calibration_provenance(self, rig_dict: Dict[str, Any]):
        self.calibration_prov = {
            "camera_model": rig_dict.get("conventions", {}).get("camera_model", "PINHOLE"),
            "distortion_model": rig_dict.get("conventions", {}).get("distortion_model", "NONE"),
            "extrinsic_convention": rig_dict.get("conventions", {}).get("extrinsic_convention"),
            "depth_axis_and_sign": rig_dict.get("conventions", {}).get("depth_axis_and_sign"),
            "depth_units": rig_dict.get("conventions", {}).get("depth_units"),
            "world_frame_state": rig_dict.get("world_frame_state"),
            "metric_scale_classification": rig_dict.get("metric_scale_classification"),
            "metric_scale_note": rig_dict.get("metric_scale_note"),
            "calibration_version": rig_dict.get("calibration_version"),
            "pose_version": rig_dict.get("pose_version"),
            "floor_leveling_rotation": rig_dict.get("floor_leveling_rotation"),
            "gltf_canonical_center": rig_dict.get("gltf_canonical_center"),
        }

    def record_stage(self, rec: StageExecutionRecord):
        self.stages.append(rec)
        if rec.warnings:
            self.warnings.extend([f"[{rec.stage_name}] {w}" for w in rec.warnings])
        if rec.errors:
            self.errors.extend([f"[{rec.stage_name}] {e}" for e in rec.errors])

    def latest_preceding_manifest(self) -> Optional[Dict[str, Any]]:
        index_path = self.run_history_dir / f"{self.video_sha256[:16]}_history.json"
        if not index_path.is_file():
            return None
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8"))
            for item in reversed(entries):
                m_path = Path(item.get("manifest_path", ""))
                if item.get("run_id") != self.run_id and m_path.is_file():
                    return json.loads(m_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    def finalize_and_write(
        self,
        output_paths: Dict[str, Optional[str]],
        qa_summary: Dict[str, Any],
        final_status: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        end_utc = utc_now_iso()
        total_dur = round(time.perf_counter() - self._t0, 3)

        output_hashes: Dict[str, Optional[str]] = {}
        for k, p in output_paths.items():
            output_hashes[k] = compute_file_sha256(p) if p else None

        cache_hits = sum(1 for s in self.stages if s.cache_status == "HIT")
        cache_misses = sum(1 for s in self.stages if s.cache_status == "MISS")

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            try:
                peak_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
            except Exception:
                pass

        manifest: Dict[str, Any] = {
            "manifest_version": "forge_v2_manifest_1.0",
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "input_provenance": self.input_prov,
            "frame_provenance": self.frame_prov,
            "calibration_and_camera": self.calibration_prov,
            "implementation_provenance": self.impl_prov,
            "execution_provenance": {
                "hardware": self.hw_prov,
                "peak_gpu_vram_mb": peak_vram_mb,
                "resolved_config": self.resolved_config,
                "configuration_hash": self.config_hash,
                "start_time_utc": self.start_utc,
                "end_time_utc": end_utc,
                "total_duration_sec": total_dur,
                "per_stage_durations_sec": {s.stage_name: s.duration_sec for s in self.stages},
                "total_retries": sum(s.retry_count for s in self.stages),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "warnings": self.warnings,
                "errors": self.errors,
            },
            "stages": [s.to_dict() for s in self.stages],
            "final_provenance": {
                "final_status": final_status,
                "output_paths": output_paths,
                "output_hashes": output_hashes,
                "qa_summary": qa_summary,
                "run_comparison": None,
            },
        }

        prev = self.latest_preceding_manifest()
        comp = compare_run_manifests(manifest, prev)
        manifest["final_provenance"]["run_comparison"] = comp
        if comp.get("diagnostic_warning"):
            manifest["execution_provenance"]["warnings"].append(comp["diagnostic_warning"])

        manifest_path = self.output_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Append to per-video run history index
        index_path = self.run_history_dir / f"{self.video_sha256[:16]}_history.json"
        history = []
        if index_path.is_file():
            try:
                history = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                history = []
        history.append({
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "end_time_utc": end_utc,
            "final_status": final_status,
            "config_hash": self.config_hash,
            "code_fingerprint": self.code_fingerprint,
            "glb_sha256": output_hashes.get("glb"),
            "manifest_path": str(manifest_path.resolve()),
        })
        index_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        return manifest_path, manifest
