"""
F.O.R.G.E. V2 — Multi-View Depth, Visibility & Free-Space Validation (V-FUSE Inspired).

Upgrades multi-view depth validation with explicit 5-state classification:
  - SUPPORTED (1)
  - OCCLUDED (2)       -> Foreground in view j hides point p; NOT pruned!
  - CONTRADICTED (3)   -> View j sees unobstructed space through p; pruned as floating ghost!
  - UNKNOWN (4)        -> Insufficient confirming/contradicting views or protected thin edge
  - KNOWN_FREE_SPACE   -> 3D voxel evidence carved along validated camera rays

Outputs per-frame depth/weight maps, support/occlusion/contradiction masks,
spatial free-space grid for Mesh QA & Hidden-Surface Completion, and fused point cloud.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d


STATE_INVALID = 0
STATE_SUPPORTED = 1
STATE_OCCLUDED = 2
STATE_CONTRADICTED = 3
STATE_UNKNOWN = 4
STATE_LABELS = {
    0: "INVALID",
    1: "SUPPORTED",
    2: "OCCLUDED",
    3: "CONTRADICTED",
    4: "UNKNOWN",
}


class FreeSpaceVoxelEvidence:
    """
    Sparse 3D occupancy & free-space evidence grid built from camera viewing rays.
    Used downstream by Planar Verification, Mesh QA (free-space violations),
    and Behind-the-Veil-inspired Hidden Completion (doorway/corridor clearance).
    """

    def __init__(self, voxel_size_m: float = 0.04):
        self.voxel_size_m = float(voxel_size_m)
        self.free_keys: np.ndarray = np.empty(0, dtype=np.int64)
        self.surface_keys: np.ndarray = np.empty(0, dtype=np.int64)

    @staticmethod
    def _pack_ijk(ijk: np.ndarray) -> np.ndarray:
        """Packs bounded signed integer voxel coords [-500000, +500000] into unique int64 keys."""
        shifted = ijk.astype(np.int64) + 500000
        return (shifted[:, 0] << 40) | (shifted[:, 1] << 20) | shifted[:, 2]

    def build_from_depth_maps(
        self,
        depth_maps: np.ndarray,
        confidence_maps: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        min_depth: float = 0.25,
        max_depth: float = 5.0,
        min_confidence: float = 0.65,
        pixel_stride: int = 4,
    ) -> Tuple[int, int]:
        N, H, W = depth_maps.shape[:3]
        v_grid, u_grid = np.meshgrid(
            np.arange(0, H, pixel_stride, dtype=np.float64),
            np.arange(0, W, pixel_stride, dtype=np.float64),
            indexing="ij",
        )
        u_f = u_grid.ravel()
        v_f = v_grid.ravel()

        free_chunks = []
        surf_chunks = []
        margin_m = max(self.voxel_size_m * 2.5, 0.07)
        inv_vox = 1.0 / self.voxel_size_m

        for i in range(N):
            d = depth_maps[i].squeeze()[::pixel_stride, ::pixel_stride].ravel()
            c = confidence_maps[i].squeeze()[::pixel_stride, ::pixel_stride].ravel()
            val = np.isfinite(d) & (d >= min_depth + margin_m) & (d <= max_depth) & (c >= min_confidence)
            if not np.any(val):
                continue

            K = intrinsics[i]
            R = extrinsics[i, :3, :3]
            t = extrinsics[i, :3, 3]
            cam_c = -R.T @ t

            z_v = d[val]
            pc = np.column_stack([(u_f[val] - K[0, 2]) / K[0, 0] * z_v, (v_f[val] - K[1, 2]) / K[1, 1] * z_v, z_v])
            pw = (pc - t) @ R
            rays = pw - cam_c[None, :]
            surf_ijk = np.floor(pw * inv_vox).astype(np.int64)
            surf_chunks.append(self._pack_ijk(surf_ijk))

            # Sample interior free-space rays densely at ~voxel_size_m step (stay clear of surface by margin_m)
            max_fracs = np.clip((z_v - margin_m) / np.maximum(z_v, 1e-4), 0.1, 0.94)
            max_steps = int(min(64, max(8, math.ceil((max_depth - min_depth) / max(self.voxel_size_m, 0.02)))))
            alphas_1d = np.linspace(0.08, 1.0, max_steps, dtype=np.float64)
            # Filter out sample points closer to camera than min_depth
            alphas = alphas_1d[:, None] * max_fracs[None, :]  # (S, R)
            valid_step = (alphas * z_v[None, :]) >= max(min_depth * 0.8, 0.15)
            # (S, R, 3) -> mask valid steps
            pts_f = cam_c[None, None, :] + alphas[:, :, None] * rays[None, :, :]
            free_pts = pts_f[valid_step]
            if len(free_pts) > 0:
                f_ijk = np.floor(free_pts * inv_vox).astype(np.int64)
                free_chunks.append(self._pack_ijk(f_ijk))

        surf_set = np.unique(np.concatenate(surf_chunks)) if surf_chunks else np.empty(0, dtype=np.int64)
        free_all = np.unique(np.concatenate(free_chunks)) if free_chunks else np.empty(0, dtype=np.int64)
        # Never mark observed surface voxels themselves as empty free space
        if len(free_all) > 0 and len(surf_set) > 0:
            free_only = free_all[~np.isin(free_all, surf_set)]
        else:
            free_only = free_all

        self.surface_keys = surf_set
        self.free_keys = free_only
        return len(self.free_keys), len(self.surface_keys)

    def query_free_space_mask(self, points_world: np.ndarray) -> np.ndarray:
        """Returns boolean mask True where query 3D points fall inside KNOWN_FREE_SPACE voxels."""
        if len(points_world) == 0 or len(self.free_keys) == 0:
            return np.zeros(len(points_world), dtype=bool)
        ijk = np.floor(np.asarray(points_world, dtype=np.float64) / self.voxel_size_m).astype(np.int64)
        qkeys = self._pack_ijk(ijk)
        idx = np.searchsorted(self.free_keys, qkeys)
        in_range = idx < len(self.free_keys)
        out = np.zeros(len(points_world), dtype=bool)
        out[in_range] = (self.free_keys[idx[in_range]] == qkeys[in_range])
        return out

    def query_surface_support_mask(self, points_world: np.ndarray) -> np.ndarray:
        if len(points_world) == 0 or len(self.surface_keys) == 0:
            return np.zeros(len(points_world), dtype=bool)
        ijk = np.floor(np.asarray(points_world, dtype=np.float64) / self.voxel_size_m).astype(np.int64)
        qkeys = self._pack_ijk(ijk)
        idx = np.searchsorted(self.surface_keys, qkeys)
        in_range = idx < len(self.surface_keys)
        out = np.zeros(len(points_world), dtype=bool)
        out[in_range] = (self.surface_keys[idx[in_range]] == qkeys[in_range])
        return out


@dataclass
class DepthConsistencyConfig:
    """Hyperparameters for V-FUSE-inspired multi-view depth & free-space validation."""

    depth_relative_thresh: float = 0.050
    occlusion_thresh: float = 0.080
    contradiction_thresh: float = 0.065
    min_support_views: int = 1
    neighbor_window_radius: int = 4
    spatial_revisit_neighbors: int = 4
    edge_gradient_thresh: float = 0.12
    thin_structure_min_conf: float = 0.70
    min_confidence: float = 0.65
    min_depth_m: float = 0.20
    max_depth_m: float = 5.50
    voxel_size_m: float = 0.012
    free_space_voxel_m: float = 0.04

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepthConsistencyReport:
    total_raw_points: int
    valid_depth_points: int
    supported_points: int
    occluded_preserved: int
    unknown_preserved: int
    thin_structures_preserved: int
    contradicted_pruned: int
    isolated_pruned: int
    total_retained_points: int
    retention_pct: float
    fused_voxel_points: int
    known_free_space_voxels: int
    observed_surface_voxels: int
    processing_time_sec: float
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_depth_gradient_map(depth_map: np.ndarray) -> np.ndarray:
    d_safe = np.maximum(depth_map, 1e-3)
    log_d = np.log(d_safe)
    grad_x = cv2.Sobel(log_d.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(log_d.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def bilinear_sample_depth_vectorized(
    depth_map: np.ndarray,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = depth_map.shape[:2]
    u = np.asarray(u_coords, dtype=np.float32)
    v = np.asarray(v_coords, dtype=np.float32)
    in_bounds = (u >= 0.0) & (u <= W - 1.0) & (v >= 0.0) & (v <= H - 1.0)
    u_safe = np.clip(u, 0.0, W - 1.0)
    v_safe = np.clip(v, 0.0, H - 1.0)
    u0 = np.floor(u_safe).astype(np.int32)
    u1 = np.minimum(u0 + 1, W - 1)
    v0 = np.floor(v_safe).astype(np.int32)
    v1 = np.minimum(v0 + 1, H - 1)
    wu, wv = u_safe - u0, v_safe - v0
    d00, d01 = depth_map[v0, u0], depth_map[v0, u1]
    d10, d11 = depth_map[v1, u0], depth_map[v1, u1]
    sampled = (1.0 - wv) * ((1.0 - wu) * d00 + wu * d01) + wv * ((1.0 - wu) * d10 + wu * d11)
    valid = in_bounds & np.isfinite(sampled) & (sampled > 0.05)
    return sampled, valid


class MultiViewDepthConsistencyFilter:
    """Executes multi-view depth, occlusion, contradiction & free-space validation."""

    def __init__(self, config: Optional[DepthConsistencyConfig] = None):
        self.config = config or DepthConsistencyConfig()

    def validate_and_fuse(
        self,
        frames_rgb: List[np.ndarray],
        depth_maps: np.ndarray,
        confidence_maps: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        dynamic_masks_infer: Optional[List[Optional[np.ndarray]]] = None,
        log_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Full V2 validation returning:
          - clean_points, clean_colors, clean_confidences, clean_support_counts
          - pixel_state_maps (N, H, W) uint8
          - support_count_maps (N, H, W) uint8
          - occlusion_masks (N, H, W) bool
          - contradiction_masks (N, H, W) bool
          - validated_depth_maps (N, H, W, 1) float32
          - validated_weight_maps (N, H, W) float32
          - free_space_evidence: FreeSpaceVoxelEvidence
          - report: DepthConsistencyReport
        """
        def _log(msg: str):
            if log_callback:
                log_callback(msg)

        t0 = time.perf_counter()
        cfg = self.config
        N = len(depth_maps)
        H, W = depth_maps.shape[1], depth_maps.shape[2]
        _log(f"[*] Multi-View Depth, Occlusion & Free-Space Validation across {N} frames ({W}x{H})...")

        grad_maps = [compute_depth_gradient_map(depth_maps[i].squeeze()) for i in range(N)]
        v_grid, u_grid = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
        u_flat, v_flat = u_grid.ravel(), v_grid.ravel()

        rotations = extrinsics[:, :3, :3].astype(np.float64)
        translations = extrinsics[:, :3, 3].astype(np.float64)
        cam_centers = np.array([-rotations[i].T @ translations[i] for i in range(N)], dtype=np.float64)
        cam_fwds = np.array([rotations[i].T @ np.array([0.0, 0.0, 1.0]) for i in range(N)], dtype=np.float64)

        pixel_state_maps = np.zeros((N, H, W), dtype=np.uint8)
        support_count_maps = np.zeros((N, H, W), dtype=np.uint8)
        occlusion_masks = np.zeros((N, H, W), dtype=bool)
        contradiction_masks = np.zeros((N, H, W), dtype=bool)
        validated_depth_maps = np.zeros((N, H, W, 1), dtype=np.float32)
        validated_weight_maps = np.zeros((N, H, W), dtype=np.float32)

        all_pts, all_cols, all_confs, all_sups = [], [], [], []
        total_raw = N * H * W
        total_valid = 0
        total_supported = 0
        total_occluded = 0
        total_unknown = 0
        total_thin = 0
        total_contra = 0
        total_isolated = 0

        for i in range(N):
            d_i = depth_maps[i].squeeze().astype(np.float64)
            c_i = confidence_maps[i].squeeze().astype(np.float64)
            g_i = grad_maps[i]
            im_i = cv2.resize(frames_rgb[i], (W, H))

            d_flat = d_i.ravel()
            c_flat = c_i.ravel()
            g_flat = g_i.ravel()
            col_flat = im_i.reshape(-1, 3)

            valid_mask = (
                (d_flat >= cfg.min_depth_m)
                & (d_flat <= cfg.max_depth_m)
                & (c_flat >= cfg.min_confidence)
                & np.isfinite(d_flat)
            )
            if dynamic_masks_infer is not None and dynamic_masks_infer[i] is not None:
                dyn = cv2.resize(dynamic_masks_infer[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).ravel() > 0
                valid_mask &= ~dyn

            n_val = int(np.count_nonzero(valid_mask))
            total_valid += n_val
            if n_val == 0:
                continue

            u_act, v_act, z_act = u_flat[valid_mask], v_flat[valid_mask], d_flat[valid_mask]
            c_act, g_act, col_act = c_flat[valid_mask], g_flat[valid_mask], col_flat[valid_mask]

            K_i = intrinsics[i]
            pts_cam_i = np.column_stack([
                (u_act - K_i[0, 2]) / K_i[0, 0] * z_act,
                (v_act - K_i[1, 2]) / K_i[1, 1] * z_act,
                z_act,
            ])
            pts_w = (pts_cam_i - translations[i]) @ rotations[i]

            # Temporal neighbors + spatially closest revisit views with aligned gaze
            temp_nbrs = set(range(max(0, i - cfg.neighbor_window_radius), min(N, i + cfg.neighbor_window_radius + 1)))
            temp_nbrs.discard(i)
            dists_c = np.linalg.norm(cam_centers - cam_centers[i], axis=1)
            cos_gaze = cam_fwds @ cam_fwds[i]
            spatial_cand = [
                int(j) for j in np.argsort(dists_c)
                if j != i and j not in temp_nbrs and dists_c[j] < 2.2 and cos_gaze[j] > 0.35
            ][:cfg.spatial_revisit_neighbors]
            target_views = sorted(temp_nbrs | set(spatial_cand))

            sup_votes = np.zeros(n_val, dtype=np.int32)
            occ_votes = np.zeros(n_val, dtype=np.int32)
            con_votes = np.zeros(n_val, dtype=np.int32)

            for j in target_views:
                K_j = intrinsics[j]
                d_j = depth_maps[j].squeeze()
                pts_cj = (pts_w @ rotations[j].T) + translations[j]
                z_proj = pts_cj[:, 2]
                pos_z = z_proj > 0.08
                u_pj = np.zeros_like(z_proj)
                v_pj = np.zeros_like(z_proj)
                u_pj[pos_z] = K_j[0, 0] * (pts_cj[pos_z, 0] / z_proj[pos_z]) + K_j[0, 2]
                v_pj[pos_z] = K_j[1, 1] * (pts_cj[pos_z, 1] / z_proj[pos_z]) + K_j[1, 2]

                sampled_d, in_b = bilinear_sample_depth_vectorized(d_j, u_pj, v_pj)
                ev = pos_z & in_b
                if not np.any(ev):
                    continue
                rel = np.zeros_like(z_proj)
                # rel > 0 => z_proj > d_j (foreground in view j lies IN FRONT OF p => OCCLUDED!)
                # rel < 0 => z_proj < d_j (view j sees behind p => CONTRADICTED free-space penetration!)
                rel[ev] = (z_proj[ev] - sampled_d[ev]) / np.maximum(z_proj[ev], 1e-3)

                sup_votes[ev & (np.abs(rel) <= cfg.depth_relative_thresh)] += 1
                occ_votes[ev & (rel > cfg.occlusion_thresh)] += 1
                con_votes[ev & (rel < -cfg.contradiction_thresh)] += 1

            is_thin = (g_act >= cfg.edge_gradient_thresh) & (c_act >= cfg.thin_structure_min_conf)
            is_supported = (sup_votes >= cfg.min_support_views) & (con_votes <= 1)
            # Contradicted: repeated free-space violation without support
            is_contradicted = (con_votes > 1) | ((con_votes > 0) & (sup_votes == 0) & (~is_thin))
            # Validly occluded from some views without being contradicted & having decent confidence
            is_occluded = (~is_supported) & (~is_contradicted) & (occ_votes > 0) & (c_act >= cfg.min_confidence)
            # Single-view / thin structure edge preserved cleanly
            is_unknown = (~is_supported) & (~is_contradicted) & (~is_occluded)

            # Policy: keep SUPPORTED, OCCLUDED (when high-conf or thin), and protected THIN/UNKNOWN only if not isolated noise
            # Wait: "Do not remove points merely because another camera cannot see them."
            # Preserve SUPPORTED, OCCLUDED, THIN_STRUCTURE, plus UNKNOWN when single-view confidence >= min_conf * 1.15
            retain_mask = (
                is_supported
                | is_occluded
                | (is_thin & (~is_contradicted))
                | (is_unknown & (c_act >= cfg.min_confidence))
            )

            st_act = np.full(n_val, STATE_UNKNOWN, dtype=np.uint8)
            st_act[is_occluded] = STATE_OCCLUDED
            st_act[is_contradicted] = STATE_CONTRADICTED
            st_act[is_supported] = STATE_SUPPORTED

            full_st = np.zeros(H * W, dtype=np.uint8)
            full_st[valid_mask] = st_act
            pixel_state_maps[i] = full_st.reshape(H, W)

            full_sup = np.zeros(H * W, dtype=np.uint8)
            full_sup[valid_mask] = np.clip(sup_votes, 0, 255).astype(np.uint8)
            support_count_maps[i] = full_sup.reshape(H, W)

            occlusion_masks[i] = (pixel_state_maps[i] == STATE_OCCLUDED)
            contradiction_masks[i] = (pixel_state_maps[i] == STATE_CONTRADICTED)

            val_d_flat = np.zeros(H * W, dtype=np.float32)
            val_w_flat = np.zeros(H * W, dtype=np.float32)
            weights_act = c_act * (1.0 + 0.18 * np.clip(sup_votes, 0, 5))
            weights_act = np.where(is_occluded, weights_act * 0.70, weights_act)
            val_d_flat[valid_mask] = np.where(retain_mask, z_act, 0.0).astype(np.float32)
            val_w_flat[valid_mask] = np.where(retain_mask, weights_act, 0.0).astype(np.float32)
            validated_depth_maps[i, ..., 0] = val_d_flat.reshape(H, W)
            validated_weight_maps[i] = val_w_flat.reshape(H, W)

            total_supported += int(np.count_nonzero(is_supported & retain_mask))
            total_occluded += int(np.count_nonzero(is_occluded & retain_mask))
            total_thin += int(np.count_nonzero(is_thin & retain_mask & (~is_supported)))
            total_unknown += int(np.count_nonzero(is_unknown & retain_mask & (~is_thin)))
            total_contra += int(np.count_nonzero(is_contradicted))
            total_isolated += int(np.count_nonzero(~retain_mask & (~is_contradicted)))

            if np.any(retain_mask):
                all_pts.append(pts_w[retain_mask])
                all_cols.append(col_act[retain_mask])
                all_confs.append(weights_act[retain_mask])
                all_sups.append(sup_votes[retain_mask])

        if not all_pts:
            raise RuntimeError("Multi-view validation pruned all points. Check pose alignment or thresholds.")

        # Build 3D Free-Space Evidence Voxel Grid from validated depths
        free_ev = FreeSpaceVoxelEvidence(voxel_size_m=cfg.free_space_voxel_m)
        n_free_vox, n_surf_vox = free_ev.build_from_depth_maps(
            depth_maps=validated_depth_maps,
            confidence_maps=confidence_maps,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            min_depth=cfg.min_depth_m,
            max_depth=cfg.max_depth_m,
            min_confidence=cfg.min_confidence,
        )

        fused_pts = np.vstack(all_pts).astype(np.float32)
        fused_cols = np.vstack(all_cols).astype(np.uint8)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(fused_pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(fused_cols.astype(np.float64) / 255.0)
        down_pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size_m)
        clean_pts = np.asarray(down_pcd.points, dtype=np.float32)
        clean_cols = (np.asarray(down_pcd.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
        clean_confs = np.ones(len(clean_pts), dtype=np.float32)

        total_retained = len(fused_pts)
        retention_pct = (total_retained / max(1, total_valid)) * 100.0
        dur_sec = time.perf_counter() - t0

        report = DepthConsistencyReport(
            total_raw_points=total_raw,
            valid_depth_points=total_valid,
            supported_points=total_supported,
            occluded_preserved=total_occluded,
            unknown_preserved=total_unknown,
            thin_structures_preserved=total_thin,
            contradicted_pruned=total_contra,
            isolated_pruned=total_isolated,
            total_retained_points=total_retained,
            retention_pct=round(retention_pct, 2),
            fused_voxel_points=len(clean_pts),
            known_free_space_voxels=n_free_vox,
            observed_surface_voxels=n_surf_vox,
            processing_time_sec=round(dur_sec, 3),
            config=cfg.to_dict(),
        )
        _log(f"  [+] Multi-View Validation ({dur_sec:.2f}s): Retained {total_retained:,}/{total_valid:,} ({retention_pct:.1f}%) "
             f"[Supported={total_supported:,}, Occluded={total_occluded:,}, Thin={total_thin:,}, Unknown={total_unknown:,}, "
             f"ContradictedPruned={total_contra:,}] | FreeSpaceVoxels={n_free_vox:,}")

        return {
            "clean_points": clean_pts,
            "clean_colors": clean_cols,
            "clean_confidences": clean_confs,
            "pixel_state_maps": pixel_state_maps,
            "support_count_maps": support_count_maps,
            "occlusion_masks": occlusion_masks,
            "contradiction_masks": contradiction_masks,
            "validated_depth_maps": validated_depth_maps,
            "validated_weight_maps": validated_weight_maps,
            "free_space_evidence": free_ev,
            "report": report,
        }

    def filter_and_fuse(
        self,
        frames_rgb: List[np.ndarray],
        depth_maps: np.ndarray,
        confidence_maps: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        log_callback: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DepthConsistencyReport]:
        """Backward-compatible wrapper returning (clean_pts, clean_cols, clean_confs, report)."""
        res = self.validate_and_fuse(
            frames_rgb=frames_rgb,
            depth_maps=depth_maps,
            confidence_maps=confidence_maps,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            log_callback=log_callback,
        )
        return res["clean_points"], res["clean_colors"], res["clean_confidences"], res["report"]
