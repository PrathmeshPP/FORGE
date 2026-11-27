"""
F.O.R.G.E. V2 — Canonical Camera, Coordinate System & Frame Record.

Single authoritative camera, calibration, and coordinate-convention contract used
across VGGT-Omega inference, window alignment, pose graph & bundle adjustment,
multi-view depth validation, PlanarGS-inspired refinement, Poisson/TSDF meshing,
hidden-surface completion, texture projection, and interactive walkthrough export.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


COORDINATE_CONVENTIONS: Dict[str, Any] = {
    "camera_model": "PINHOLE",
    "distortion_model": "NONE",
    "extrinsic_convention": "world_to_camera_opencv",
    "extrinsic_equation": "p_cam = R @ p_world + t; cam_center = -R.T @ t",
    "camera_axes": "+X right, +Y down, +Z forward (optical axis)",
    "depth_axis_and_sign": "+Z_cam_positive_forward",
    "depth_units": "model_units_unverified_nominal_m",
    "gltf_canonical_transform": "p_gltf = (diag(1, -1, -1) @ p_world) - canonical_center",
}


def compute_preprocessing_geometry(
    orig_width: int,
    orig_height: int,
    image_resolution: int = 320,
    patch_size: int = 16,
    mode: str = "balanced",
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Dict[str, Any]:
    """
    Mirrors VGGT-Omega `load_and_preprocess_images` exactly without touching disk:
    1. Center-crops extreme aspect ratios into [min_aspect_ratio, max_aspect_ratio].
    2. Computes target `(infer_height, infer_width)` aligned to `patch_size`.
    Returns explicit crop box and scale factors between native and inference pixels.
    """
    if orig_width <= 0 or orig_height <= 0:
        raise ValueError(f"Invalid frame dimensions: {orig_width}x{orig_height}")

    aspect = orig_height / float(orig_width)
    left, top, crop_w, crop_h = 0, 0, int(orig_width), int(orig_height)

    if aspect < min_aspect_ratio:
        crop_w = min(orig_width, max(1, int(round(orig_height / min_aspect_ratio))))
        left = max((orig_width - crop_w) // 2, 0)
    elif aspect > max_aspect_ratio:
        crop_h = min(orig_height, max(1, int(round(orig_width * max_aspect_ratio))))
        top = max((orig_height - crop_h) // 2, 0)

    crop_aspect = crop_h / max(float(crop_w), 1.0)
    if mode == "balanced":
        token_number = (image_resolution // patch_size) ** 2
        w_patches = math.sqrt(token_number / crop_aspect)
        h_patches = token_number / w_patches
        w_patches_i = max(1, int(round(w_patches)))
        h_patches_i = max(1, int(round(h_patches)))
        infer_h, infer_w = h_patches_i * patch_size, w_patches_i * patch_size
    elif mode == "max_size":
        if crop_aspect >= 1.0:
            infer_h = image_resolution
            infer_w = max(patch_size, int(round((image_resolution / crop_aspect) / patch_size)) * patch_size)
        else:
            infer_w = image_resolution
            infer_h = max(patch_size, int(round((image_resolution * crop_aspect) / patch_size)) * patch_size)
    else:
        raise ValueError(f"Unsupported preprocess mode '{mode}'")

    sx_crop_to_infer = float(infer_w) / float(crop_w)
    sy_crop_to_infer = float(infer_h) / float(crop_h)

    return {
        "orig_width": int(orig_width),
        "orig_height": int(orig_height),
        "orientation": "landscape" if orig_width > orig_height else ("portrait" if orig_height > orig_width else "square"),
        "crop_box_xywh": [int(left), int(top), int(crop_w), int(crop_h)],
        "was_cropped": bool(crop_w != orig_width or crop_h != orig_height),
        "infer_width": int(infer_w),
        "infer_height": int(infer_h),
        "sx_crop_to_infer": sx_crop_to_infer,
        "sy_crop_to_infer": sy_crop_to_infer,
        "sx_infer_to_crop": float(crop_w) / float(infer_w),
        "sy_infer_to_crop": float(crop_h) / float(infer_h),
        "mode": mode,
        "image_resolution": int(image_resolution),
        "patch_size": int(patch_size),
    }


def transform_intrinsics(
    K_src: np.ndarray,
    crop_left: float = 0.0,
    crop_top: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """
    Transforms a 3x3 pinhole intrinsics matrix under optional cropping and anisotropic resizing:
        u_dst = (u_src - crop_left) * scale_x
        v_dst = (v_src - crop_top)  * scale_y
    """
    K = np.asarray(K_src, dtype=np.float64).copy()
    K[0, 0] *= float(scale_x)
    K[1, 1] *= float(scale_y)
    K[0, 2] = (K[0, 2] - float(crop_left)) * float(scale_x)
    K[1, 2] = (K[1, 2] - float(crop_top)) * float(scale_y)
    return K


def infer_to_native_intrinsics(K_infer: np.ndarray, preproc: Dict[str, Any]) -> np.ndarray:
    """Maps inference-grid intrinsics back into original pre-crop native pixel coordinates."""
    left, top, _, _ = preproc["crop_box_xywh"]
    sx_inv = float(preproc["sx_infer_to_crop"])
    sy_inv = float(preproc["sy_infer_to_crop"])
    K = np.asarray(K_infer, dtype=np.float64).copy()
    K[0, 0] *= sx_inv
    K[1, 1] *= sy_inv
    K[0, 2] = K[0, 2] * sx_inv + float(left)
    K[1, 2] = K[1, 2] * sy_inv + float(top)
    return K


def native_to_infer_intrinsics(K_native: np.ndarray, preproc: Dict[str, Any]) -> np.ndarray:
    """Maps native full-resolution pinhole intrinsics into VGGT inference pixel grid."""
    left, top, _, _ = preproc["crop_box_xywh"]
    return transform_intrinsics(
        K_native,
        crop_left=left,
        crop_top=top,
        scale_x=preproc["sx_crop_to_infer"],
        scale_y=preproc["sy_crop_to_infer"],
    )


def apply_sim3_to_points(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies p' = scale * (R @ p) + t row-wise to (N, 3) points."""
    pts = np.asarray(points, dtype=np.float64)
    return float(scale) * (pts @ np.asarray(R, dtype=np.float64).T) + np.asarray(t, dtype=np.float64)


def apply_sim3_to_extrinsic(ext_w2c: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Transforms an OpenCV [R_c | t_c] world-to-camera extrinsic matrix under world similarity:
        p_w' = scale * R @ p_w + t
    So that projection u = K (R_c' p_w' + t_c') remains pixel-identical while depth scales by `scale`:
        c_w' = scale * R @ (-R_c.T @ t_c) + t
        R_c' = R_c @ R.T
        t_c' = -R_c' @ c_w'
    """
    ext = np.asarray(ext_w2c, dtype=np.float64)
    R_c = ext[:3, :3]
    t_c = ext[:3, 3]
    c_w = -R_c.T @ t_c
    R_w = np.asarray(R, dtype=np.float64)
    t_w = np.asarray(t, dtype=np.float64)
    c_new = float(scale) * (R_w @ c_w) + t_w
    R_new = R_c @ R_w.T
    t_new = -R_new @ c_new
    return np.hstack([R_new, t_new.reshape(3, 1)])


def camera_center_from_extrinsic(ext_w2c: np.ndarray) -> np.ndarray:
    """Returns 3D world coordinates of the camera optical center: -R^T @ t."""
    R = np.asarray(ext_w2c[:3, :3], dtype=np.float64)
    t = np.asarray(ext_w2c[:3, 3], dtype=np.float64)
    return -R.T @ t


def extrinsic_from_center_and_rotation(R_w2c: np.ndarray, center_w: np.ndarray) -> np.ndarray:
    """Constructs 3x4 [R | -R @ c] world-to-camera matrix."""
    R = np.asarray(R_w2c, dtype=np.float64)
    c = np.asarray(center_w, dtype=np.float64)
    t = -R @ c
    return np.hstack([R, t.reshape(3, 1)])


def backproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
) -> np.ndarray:
    """Backprojects (u, v, depth) in pixel coordinates into world 3D points."""
    u_a = np.asarray(u, dtype=np.float64)
    v_a = np.asarray(v, dtype=np.float64)
    d_a = np.asarray(depth, dtype=np.float64)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    p_cam = np.stack([(u_a - cx) / fx * d_a, (v_a - cy) / fy * d_a, d_a], axis=-1)
    R = np.asarray(ext_w2c[:3, :3], dtype=np.float64)
    t = np.asarray(ext_w2c[:3, 3], dtype=np.float64)
    return (p_cam - t) @ R


def project_world_points(
    pts_w: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projects (M, 3) world points into (u, v, z_cam)."""
    pts = np.asarray(pts_w, dtype=np.float64)
    R = np.asarray(ext_w2c[:3, :3], dtype=np.float64)
    t = np.asarray(ext_w2c[:3, 3], dtype=np.float64)
    p_cam = (pts @ R.T) + t
    z = p_cam[:, 2]
    z_safe = np.where(np.abs(z) > 1e-9, z, 1e-9)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    u = fx * (p_cam[:, 0] / z_safe) + cx
    v = fy * (p_cam[:, 1] / z_safe) + cy
    return u, v, z


def world_to_gltf_canonical(
    pts_w: np.ndarray,
    canonical_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts OpenCV Y-down world coordinates into Three.js/glTF centered Y-up coordinates."""
    T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
    pts = np.asarray(pts_w, dtype=np.float64) * T_canon
    center = pts.mean(axis=0) if canonical_center is None else np.asarray(canonical_center, dtype=np.float64)
    return pts - center, center


def gltf_canonical_to_world(pts_gltf: np.ndarray, canonical_center: np.ndarray) -> np.ndarray:
    """Inverse of `world_to_gltf_canonical` for round-trip verification."""
    T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
    return (np.asarray(pts_gltf, dtype=np.float64) + np.asarray(canonical_center, dtype=np.float64)) * T_canon


@dataclass
class CanonicalCameraFrame:
    """Single canonical representation for a curated frame across all V2 pipeline stages."""

    selected_index: int
    orig_frame_index: int
    timestamp_sec: float
    frame_path: str
    frame_sha256: str
    orig_width: int
    orig_height: int
    infer_width: int
    infer_height: int
    crop_box_xywh: List[int]
    sx_crop_to_infer: float
    sy_crop_to_infer: float
    intrinsics_infer: List[List[float]]
    intrinsics_native: List[List[float]]
    extrinsic_w2c: List[List[float]]
    distortion_coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    camera_model: str = "PINHOLE"
    window_ids: List[int] = field(default_factory=list)
    is_bridge_frame: bool = False
    selection_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalCameraRig:
    """Complete scene camera & calibration record shared across all reconstruction branches."""

    run_id: str
    scene_id: str
    video_sha256: str
    calibration_version: str = "vggt_pinhole_v2"
    pose_version: str = "raw_window_sim3"
    world_frame_state: str = "vggt_opencv_y_down"
    metric_scale_classification: str = "estimated"  # "measured" | "calibrated" | "estimated"
    metric_scale_note: str = "Derived from VGGT-Omega relative scale without physical GCP/laser measurement."
    conventions: Dict[str, Any] = field(default_factory=lambda: dict(COORDINATE_CONVENTIONS))
    floor_leveling_rotation: List[List[float]] = field(default_factory=lambda: np.eye(3, dtype=np.float64).tolist())
    gltf_canonical_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    frames: List[CanonicalCameraFrame] = field(default_factory=list)

    def intrinsics_infer_array(self) -> np.ndarray:
        return np.array([f.intrinsics_infer for f in self.frames], dtype=np.float64)

    def intrinsics_native_array(self) -> np.ndarray:
        return np.array([f.intrinsics_native for f in self.frames], dtype=np.float64)

    def extrinsics_array(self) -> np.ndarray:
        return np.array([f.extrinsic_w2c for f in self.frames], dtype=np.float64)

    def update_extrinsics(self, new_extrinsics: np.ndarray, pose_version: str, world_frame_state: Optional[str] = None):
        ext = np.asarray(new_extrinsics, dtype=np.float64)
        if len(ext) != len(self.frames):
            raise ValueError(f"Expected {len(self.frames)} extrinsics, got {len(ext)}")
        for i, fr in enumerate(self.frames):
            fr.extrinsic_w2c = ext[i].tolist()
        self.pose_version = pose_version
        if world_frame_state is not None:
            self.world_frame_state = world_frame_state

    def compute_rig_sha256(self, include_run_id: bool = False) -> str:
        d = self.to_dict()
        if not include_run_id:
            d.pop("run_id", None)
        payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scene_id": self.scene_id,
            "video_sha256": self.video_sha256,
            "calibration_version": self.calibration_version,
            "pose_version": self.pose_version,
            "world_frame_state": self.world_frame_state,
            "metric_scale_classification": self.metric_scale_classification,
            "metric_scale_note": self.metric_scale_note,
            "conventions": self.conventions,
            "floor_leveling_rotation": self.floor_leveling_rotation,
            "gltf_canonical_center": self.gltf_canonical_center,
            "num_frames": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
        }

    def save_json(self, out_path) -> str:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return self.compute_rig_sha256()
