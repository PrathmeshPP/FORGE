"""
F.O.R.G.E. — VGGT-Omega Neural Inference Persistent Cache Manager.

Computes deterministic, cryptographically secure cache keys based on:
1. Video file content SHA-256 (or frame hashes)
2. Selected frame indices and timestamps
3. Model checkpoint path and parameter fingerprint
4. Inference configuration (resolution, window_size, overlap)

Stores and reloads:
- Camera extrinsics (N, 3, 4)
- Camera intrinsics (N, 3, 3)
- Depth predictions (N, H, W, 1)
- Confidence predictions (N, H, W)
- Window alignment manifests and section integrity metrics
- Performance telemetry (cold vs cached speedup)
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch


@dataclass
class VGGTInferenceCacheMetadata:
    cache_key: str
    video_path: str
    video_sha256: str
    num_frames: int
    selected_frame_indices: List[int]
    image_resolution: int
    window_size: int
    overlap: int
    checkpoint_path: str
    device: str
    created_at_utc: str
    cold_total_inference_sec: float
    cold_sustained_fps: float
    cold_peak_vram_mb: float
    sim3_mean_residual_m: float
    num_windows: int
    alignment_integrity_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VGGTInferenceCacheManager:
    """Manages persistent caching for VGGT-Omega inference results."""

    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            # Default to user_workspace/cache/vggt_inference
            project_root = Path(__file__).resolve().parent.parent.parent
            self.cache_dir = project_root / "user_workspace" / "cache" / "vggt_inference"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_sha256(filepath: Union[str, Path], block_size: int = 1024 * 1024 * 4) -> str:
        p = Path(filepath)
        if not p.exists():
            return "file_not_found"
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(block_size):
                h.update(chunk)
        return h.hexdigest()

    def generate_cache_key(
        self,
        video_sha256: str,
        frame_indices: List[int],
        image_resolution: int,
        window_size: int,
        overlap: int,
        checkpoint_path: str
    ) -> str:
        """Computes a unique SHA-256 cache key for the inference run configuration."""
        raw_sig = (
            f"vggt_v2;"
            f"vid:{video_sha256};"
            f"frames:{','.join(map(str, frame_indices))};"
            f"res:{image_resolution};"
            f"win:{window_size};"
            f"ovl:{overlap};"
            f"ckpt:{Path(checkpoint_path).name}"
        )
        return hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()

    def get_cache_paths(self, cache_key: str) -> Tuple[Path, Path]:
        npz_path = self.cache_dir / f"vggt_cache_{cache_key[:16]}.npz"
        json_path = self.cache_dir / f"vggt_cache_{cache_key[:16]}_meta.json"
        return npz_path, json_path

    def has_valid_cache(self, cache_key: str) -> bool:
        npz_p, json_p = self.get_cache_paths(cache_key)
        return npz_p.exists() and json_p.exists() and npz_p.stat().st_size > 1024

    def load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Loads cached VGGT predictions and alignment metadata."""
        npz_p, json_p = self.get_cache_paths(cache_key)
        if not self.has_valid_cache(cache_key):
            return None

        t0 = time.perf_counter()
        try:
            with open(json_p, "r", encoding="utf-8") as f:
                meta = json.load(f)

            npz_data = np.load(npz_p)
            extrinsics = npz_data["extrinsics"]
            intrinsics = npz_data["intrinsics"]
            depth = npz_data["depth"]
            confidence = npz_data["confidence"]

            load_sec = time.perf_counter() - t0
            raw_windows = []
            win_frames_meta = meta.get("raw_window_frames", [])
            for w_idx, f_idxs in enumerate(win_frames_meta):
                k_prefix = f"w{w_idx}_"
                if (k_prefix + "ext") in npz_data:
                    raw_windows.append({
                        "window_index": w_idx,
                        "frame_indices": f_idxs,
                        "local_extrinsics": npz_data[k_prefix + "ext"],
                        "local_intrinsics": npz_data[k_prefix + "intr"],
                        "local_depth": npz_data[k_prefix + "depth"].astype(np.float32),
                        "local_conf": npz_data[k_prefix + "conf"].astype(np.float32),
                    })

            cached_result = {
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
                "depth": depth,
                "confidence": confidence,
                "raw_windows": raw_windows,
                "windows": win_frames_meta,
                "window_alignment_manifest": meta.get("window_alignment_manifest", []),
                "section_summary": meta.get("section_summary", {}),
                "performance": meta.get("performance", {}),
                "cache_metadata": meta.get("cache_metadata", {}),
                "cache_load_time_sec": round(load_sec, 4),
                "is_cached": True
            }
            return cached_result
        except Exception as e:
            print(f"[!] Warning: failed to load VGGT cache {cache_key}: {e}")
            return None

    def save_cache(
        self,
        cache_key: str,
        results: Dict[str, Any],
        video_path: str,
        video_sha256: str,
        frame_indices: List[int],
        image_resolution: int,
        window_size: int,
        overlap: int,
        checkpoint_path: str,
        device: str
    ) -> Tuple[Path, Path]:
        """Saves VGGT predictions and metadata to disk."""
        npz_p, json_p = self.get_cache_paths(cache_key)

        extrinsics = results["extrinsics"]
        intrinsics = results["intrinsics"]
        depth = results["depth"]
        confidence = results["confidence"]

        save_kwargs = {
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "depth": depth,
            "confidence": confidence,
        }
        raw_windows = results.get("raw_windows", [])
        raw_win_frames = []
        for w_idx, w in enumerate(raw_windows):
            raw_win_frames.append([int(x) for x in w["frame_indices"]])
            k_prefix = f"w{w_idx}_"
            save_kwargs[k_prefix + "ext"] = np.asarray(w["local_extrinsics"], dtype=np.float32)
            save_kwargs[k_prefix + "intr"] = np.asarray(w["local_intrinsics"], dtype=np.float32)
            save_kwargs[k_prefix + "depth"] = np.asarray(w["local_depth"], dtype=np.float16)
            save_kwargs[k_prefix + "conf"] = np.asarray(w["local_conf"], dtype=np.float16)

        np.savez_compressed(npz_p, **save_kwargs)

        perf = results.get("performance", {})
        sec_sum = results.get("section_summary", {})
        meta = VGGTInferenceCacheMetadata(
            cache_key=cache_key,
            video_path=str(video_path),
            video_sha256=video_sha256,
            num_frames=len(frame_indices),
            selected_frame_indices=frame_indices,
            image_resolution=image_resolution,
            window_size=window_size,
            overlap=overlap,
            checkpoint_path=str(checkpoint_path),
            device=str(device),
            created_at_utc=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            cold_total_inference_sec=perf.get("total_vggt_inference_sec", 0.0),
            cold_sustained_fps=perf.get("sustained_inference_fps", 0.0),
            cold_peak_vram_mb=perf.get("peak_vram_mb", 0.0),
            sim3_mean_residual_m=perf.get("mean_sim3_alignment_error_m", 0.0),
            num_windows=perf.get("num_windows", 0),
            alignment_integrity_rate=sec_sum.get("alignment_integrity_rate", 1.0)
        )

        payload = {
            "cache_metadata": meta.to_dict(),
            "performance": perf,
            "section_summary": sec_sum,
            "window_alignment_manifest": results.get("window_alignment_manifest", []),
            "raw_window_frames": raw_win_frames,
        }

        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return npz_p, json_p
