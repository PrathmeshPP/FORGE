"""
F.O.R.G.E. — Multi-View Geometry-Consistency Filtering Module.

Provides isolated, configurable multi-view photogrammetric consistency checks across
predicted metric depth maps and camera poses. Rejects floating artifacts, sky bleed,
and single-view hallucinated geometry by enforcing cross-view depth agreement and
two-way epipolar reprojection constraints.

CRITICAL ISOLATION NOTICE:
This module is an opt-in experimental filter. It does NOT replace or modify the core
reconstruction pipeline or existing baseline model outputs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np


@dataclass
class MultiViewConsistencyConfig:
    """Configuration parameters for multi-view geometry-consistency filtering."""
    depth_relative_thresh: float = 0.04       # Maximum allowed relative depth discrepancy |z_proj - d_samp| / z_proj
    reproj_error_thresh_pix: float = 2.5      # Maximum allowed two-way epipolar reprojection error in pixels
    min_consistent_views: int = 1             # Minimum number of other overlapping views that must confirm the point
    min_confidence: float = 0.15              # Minimum predicted depth confidence floor
    max_depth_m: Optional[float] = None       # Optional depth ceiling to suppress horizon/sky bleed
    downsample_voxel_m: Optional[float] = None  # Optional post-filtering spatial voxel decimation
    neighbor_window_size: Optional[int] = None # Optional constraint on temporal view neighborhood (None = all views)


@dataclass
class MultiViewConsistencyResult:
    """Summary of multi-view consistency filtering execution and telemetry."""
    total_input_points: int
    total_retained_points: int
    pruned_points_count: int
    pruned_points_pct: float
    processing_time_s: float
    per_frame_retained_pct: List[float]
    config: Dict[str, Any]
    summary_message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bilinear_sample_depth(
    depth_map: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bilinearly samples a 2D depth map at sub-pixel (u, v) coordinates.

    Args:
        depth_map: 2D array (H, W) of metric depth values.
        u_coords: 1D array of horizontal coordinates (columns).
        v_coords: 1D array of vertical coordinates (rows).

    Returns:
        Tuple of:
          - sampled_depth: 1D array of interpolated depth values.
          - valid_mask: 1D boolean array indicating coordinates strictly inside image bounds.
    """
    height, width = depth_map.shape[:2]
    u = np.asarray(u_coords, dtype=np.float64)
    v = np.asarray(v_coords, dtype=np.float64)

    valid_mask = (u >= 0.0) & (u <= width - 1.0) & (v >= 0.0) & (v <= height - 1.0)

    u_safe = np.clip(u, 0.0, width - 1.0)
    v_safe = np.clip(v, 0.0, height - 1.0)

    u0 = np.floor(u_safe).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v0 = np.floor(v_safe).astype(np.int64)
    v1 = np.minimum(v0 + 1, height - 1)

    wu = u_safe - u0
    wv = v_safe - v0

    d00 = depth_map[v0, u0]
    d01 = depth_map[v0, u1]
    d10 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]

    # Check finite validity of sampling neighborhood
    finite_neigh = np.isfinite(d00) & np.isfinite(d01) & np.isfinite(d10) & np.isfinite(d11)
    valid_mask &= finite_neigh

    top = (1.0 - wu) * d00 + wu * d01
    bottom = (1.0 - wu) * d10 + wu * d11
    sampled = (1.0 - wv) * top + wv * bottom

    valid_mask &= (sampled > 0.0) & np.isfinite(sampled)
    return sampled, valid_mask


def filter_point_cloud_multiview(
    depth_maps: np.ndarray,
    camera_poses_extrinsics: np.ndarray,
    camera_intrinsics: np.ndarray,
    color_frames: Optional[List[np.ndarray]] = None,
    confidences: Optional[np.ndarray] = None,
    config: Optional[MultiViewConsistencyConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, MultiViewConsistencyResult]:
    """
    Applies multi-view geometric consistency filtering to multi-frame depth maps.

    Enforces:
    1. Cross-view depth consistency: |z_proj - z_sampled| / z_proj <= depth_relative_thresh
    2. Two-way reprojection consistency: ||(u_back, v_back) - (u_orig, v_orig)|| <= reproj_error_thresh_pix
    3. Multi-view consensus: At least min_consistent_views agree on the surface hypothesis.

    Args:
        depth_maps: (N, H, W, 1) or (N, H, W) metric depth array.
        camera_poses_extrinsics: (N, 3, 4) OpenCV [R | T] world-to-camera matrix array.
        camera_intrinsics: (N, 3, 3) camera calibration matrix array.
        color_frames: Optional list of N (H, W, 3) uint8 or float RGB frames.
        confidences: Optional (N, H, W) confidence scores.
        config: MultiViewConsistencyConfig filtering parameters.

    Returns:
        Tuple of:
          - filtered_points: (M, 3) float64 array of validated 3D world points.
          - filtered_colors: (M, 3) float64 array of RGB colors [0.0, 1.0].
          - result: MultiViewConsistencyResult metadata and telemetry.
    """
    start_time = time.perf_counter()
    cfg = config or MultiViewConsistencyConfig()

    d_array = np.asarray(depth_maps, dtype=np.float64)
    if d_array.ndim == 4 and d_array.shape[-1] == 1:
        d_array = d_array[..., 0]

    num_frames, height, width = d_array.shape
    ext = np.asarray(camera_poses_extrinsics, dtype=np.float64)
    intr = np.asarray(camera_intrinsics, dtype=np.float64)

    # 1. Pixel coordinate grid
    v_grid, u_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    u_flat = u_grid.flatten()
    v_flat = v_grid.flatten()
    num_pixels = height * width

    # Per-frame camera parameters
    rotations = ext[:, :3, :3]       # (N, 3, 3)
    translations = ext[:, :3, 3]     # (N, 3)
    inv_rotations = np.transpose(rotations, (0, 2, 1)) # (N, 3, 3)

    per_frame_retained_pct = []
    accumulated_points = []
    accumulated_colors = []
    total_input_count = 0

    for i in range(num_frames):
        depth_i = d_array[i]
        d_i_flat = depth_i.flatten()

        valid_i = np.isfinite(d_i_flat) & (d_i_flat > 0.05)
        if cfg.max_depth_m is not None:
            valid_i &= (d_i_flat <= cfg.max_depth_m)

        if confidences is not None:
            conf_i_flat = np.asarray(confidences[i]).flatten()
            valid_i &= (conf_i_flat >= cfg.min_confidence)

        frame_input_count = int(np.count_nonzero(valid_i))
        total_input_count += frame_input_count

        if frame_input_count == 0:
            per_frame_retained_pct.append(0.0)
            continue

        # Active pixel indices for frame i
        u_active = u_flat[valid_i]
        v_active = v_flat[valid_i]
        z_active = d_i_flat[valid_i]

        fx_i = intr[i, 0, 0]
        fy_i = intr[i, 1, 1]
        cx_i = intr[i, 0, 2]
        cy_i = intr[i, 1, 2]

        # Backproject active pixels to Camera frame i
        x_cam_i = (u_active - cx_i) / fx_i * z_active
        y_cam_i = (v_active - cy_i) / fy_i * z_active
        pts_cam_i = np.column_stack([x_cam_i, y_cam_i, z_active]) # (M_i, 3)

        # Transform to World coordinates: P_world = R_i^T * (P_cam_i - T_i)
        pts_world_i = np.dot(pts_cam_i - translations[i], rotations[i]) # (M_i, 3)

        # Consistency vote accumulator across other views
        consistent_votes = np.zeros(len(pts_world_i), dtype=np.int32)

        # Determine target views to check
        target_views = [j for j in range(num_frames) if j != i]
        if cfg.neighbor_window_size is not None:
            w = cfg.neighbor_window_size
            target_views = [j for j in target_views if abs(j - i) <= w]

        for j in target_views:
            # Project pts_world_i into Camera frame j: P_cam_j = R_j * P_world + T_j
            pts_cam_j = np.dot(pts_world_i, rotations[j].T) + translations[j]

            z_proj_j = pts_cam_j[:, 2]
            positive_z = z_proj_j > 0.05

            fx_j = intr[j, 0, 0]
            fy_j = intr[j, 1, 1]
            cx_j = intr[j, 0, 2]
            cy_j = intr[j, 1, 2]

            # Projected 2D pixel coordinates in View j
            u_proj_j = np.zeros_like(z_proj_j)
            v_proj_j = np.zeros_like(z_proj_j)
            u_proj_j[positive_z] = fx_j * (pts_cam_j[positive_z, 0] / z_proj_j[positive_z]) + cx_j
            v_proj_j[positive_z] = fy_j * (pts_cam_j[positive_z, 1] / z_proj_j[positive_z]) + cy_j

            # Sample depth at projected pixel coordinates in View j
            sampled_depth_j, in_bounds_j = bilinear_sample_depth(
                d_array[j], u_proj_j, v_proj_j
            )

            # Check 1: Relative depth agreement
            candidate_mask = positive_z & in_bounds_j
            depth_err_rel = np.full_like(z_proj_j, np.inf)
            depth_err_rel[candidate_mask] = (
                np.abs(z_proj_j[candidate_mask] - sampled_depth_j[candidate_mask]) / z_proj_j[candidate_mask]
            )
            depth_consistent = depth_err_rel <= cfg.depth_relative_thresh

            # Check 2: Two-way epipolar reprojection error back to View i
            # Reconstruct 3D point in View j using sampled depth
            reproj_candidate_mask = candidate_mask & depth_consistent
            reproj_err_pix = np.full_like(z_proj_j, np.inf)

            if np.any(reproj_candidate_mask):
                u_c = u_proj_j[reproj_candidate_mask]
                v_c = v_proj_j[reproj_candidate_mask]
                z_c = sampled_depth_j[reproj_candidate_mask]

                # 3D in Cam j
                x_cj = (u_c - cx_j) / fx_j * z_c
                y_cj = (v_c - cy_j) / fy_j * z_c
                pts_cj = np.column_stack([x_cj, y_cj, z_c])

                # Transform Cam j -> World
                pts_wj = np.dot(pts_cj - translations[j], rotations[j])

                # Transform World -> Cam i
                pts_ci_back = np.dot(pts_wj, rotations[i].T) + translations[i]
                z_back = pts_ci_back[:, 2]
                pos_back = z_back > 0.05

                u_back = np.zeros_like(z_back)
                v_back = np.zeros_like(z_back)
                u_back[pos_back] = fx_i * (pts_ci_back[pos_back, 0] / z_back[pos_back]) + cx_i
                v_back[pos_back] = fy_i * (pts_ci_back[pos_back, 1] / z_back[pos_back]) + cy_i

                # Reprojection pixel distance from original (u_active, v_active)
                u_orig_c = u_active[reproj_candidate_mask]
                v_orig_c = v_active[reproj_candidate_mask]

                dist_pix = np.sqrt((u_back - u_orig_c) ** 2 + (v_back - v_orig_c) ** 2)
                dist_pix[~pos_back] = np.inf

                reproj_err_pix[reproj_candidate_mask] = dist_pix

            view_j_consistent = (depth_consistent) & (reproj_err_pix <= cfg.reproj_error_thresh_pix)
            consistent_votes += view_j_consistent.astype(np.int32)

        # Consensus thresholding
        kept_mask = consistent_votes >= cfg.min_consistent_views
        kept_count = int(np.count_nonzero(kept_mask))
        retained_pct = 100.0 * kept_count / max(frame_input_count, 1)
        per_frame_retained_pct.append(round(retained_pct, 2))

        if kept_count > 0:
            accumulated_points.append(pts_world_i[kept_mask])

            # Sample colors if provided
            if color_frames is not None and i < len(color_frames):
                frame_rgb = color_frames[i]
                if frame_rgb.dtype == np.uint8:
                    frame_rgb = frame_rgb.astype(np.float64) / 255.0
                u_kept = u_active[kept_mask]
                v_kept = v_active[kept_mask]
                colors_i = frame_rgb[v_kept, u_kept]
                accumulated_colors.append(colors_i)
            else:
                # Default neutral color
                accumulated_colors.append(np.full((kept_count, 3), 0.7, dtype=np.float64))

    if len(accumulated_points) > 0:
        final_points = np.vstack(accumulated_points)
        final_colors = np.vstack(accumulated_colors)
    else:
        final_points = np.empty((0, 3), dtype=np.float64)
        final_colors = np.empty((0, 3), dtype=np.float64)

    # Optional spatial voxel decimation
    if cfg.downsample_voxel_m is not None and len(final_points) > 0:
        voxel_size = cfg.downsample_voxel_m
        voxel_indices = np.floor(final_points / voxel_size).astype(np.int64)
        # Fast unique voxel downsampling
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        final_points = final_points[unique_idx]
        final_colors = final_colors[unique_idx]

    duration = time.perf_counter() - start_time
    total_retained = len(final_points)
    pruned_count = total_input_count - total_retained
    pruned_pct = round(100.0 * pruned_count / max(total_input_count, 1), 2)

    msg = (
        f"Multi-View Consistency Filter completed in {duration:.2f}s. "
        f"Retained: {total_retained:,}/{total_input_count:,} points "
        f"({100.0 - pruned_pct:.1f}% kept, {pruned_pct:.1f}% pruned). "
        f"Min consensus: {cfg.min_consistent_views} views, "
        f"Depth tol: {cfg.depth_relative_thresh:.1%}, Reproj tol: {cfg.reproj_error_thresh_pix}px."
    )

    result = MultiViewConsistencyResult(
        total_input_points=total_input_count,
        total_retained_points=total_retained,
        pruned_points_count=pruned_count,
        pruned_points_pct=pruned_pct,
        processing_time_s=round(duration, 3),
        per_frame_retained_pct=per_frame_retained_pct,
        config=asdict(cfg),
        summary_message=msg
    )

    return final_points, final_colors, result
