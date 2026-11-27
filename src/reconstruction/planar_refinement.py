"""
F.O.R.G.E. V2 — PlanarGS-Inspired Planar Proposal, Cross-View Verification,
Boundary-First Merge Gating, and Conservative Geometry Refinement.

This module adapts the planar-prior verification principles of PlanarGS to F.O.R.G.E.'s
point-cloud / Poisson / TSDF architecture without training a 3D Gaussian Splatting model:
1. Optional local YOLOv8-seg object/person proposal extraction (8 GB VRAM-scheduled).
2. Normal-aware + 3D Euclidean connected-component planar patch detection (walls, floors,
   ceilings, distinct tabletops/desks) preserving furniture separation and legs.
3. Cross-view depth & free-space verification of every proposed plane.
4. Boundary-first merge gating: never merges adjacent desks across a gap/free-space barrier,
   never merges a tabletop with an adjacent wall, and never merges perpendicular walls.
5. Interior-tapered conservative correction that preserves edges, legs, doorways, and genuine
   floor slopes while never modifying camera poses.
6. Visibility-aware projection of canonical corrected points into synchronized TSDF depth maps.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree

from src.reconstruction.depth_consistency_filter import (
    FreeSpaceVoxelEvidence,
    bilinear_sample_depth_vectorized,
)


DEFAULT_YOLO_SEG_PATHS = [
    r"C:\Users\25beevdt025\Desktop\SIH\yolov8n-seg.pt",
    r"C:\Users\25beevdt025\Desktop\FORGE\models\yolov8n-seg.pt",
]

# Indoor COCO categories preserved as static scene furniture rather than pruned as dynamic carriers.
INDOOR_FURNITURE_CLASSES = (
    "bench",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "keyboard",
    "book",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
)
# Articulated / non-planar furniture classes whose geometry must NOT be pulled into wall/floor planes.
PROTECTED_NON_PLANAR_NAMES = {
    "chair",
    "couch",
    "bench",
    "tv",
    "laptop",
    "keyboard",
    "book",
    "backpack",
    "handbag",
    "suitcase",
    "potted plant",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
}
PLANAR_FURNITURE_NAMES = {"bed", "dining table"}


@dataclass
class PlanarRefinementConfig:
    enabled: bool = True
    voxel_size: float = 0.012
    cluster_eps_m: float = 0.036          # Strict connectivity gap (3.6 cm) prevents merging separate desks!
    min_patch_points: int = 90
    plane_dist_thresh_m: float = 0.025
    max_patch_mad_m: float = 0.022
    min_crossview_support_ratio: float = 0.55
    merge_max_angle_deg: float = 5.5
    merge_max_plane_offset_m: float = 0.022
    merge_max_gap_m: float = 0.035        # Only merge adjacent coplanar fragments physically < 3.5 cm apart
    boundary_taper_m: float = 0.055       # Keep 1.5cm edge strictly untouched; taper smoothly up to 7cm interior
    boundary_hard_margin_m: float = 0.015
    max_correction_m: float = 0.042
    max_blend_strength: float = 0.65
    enable_yolo_proposals: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlanarRegionRecord:
    region_id: int
    semantic_label: str  # "floor" | "ceiling" | "wall" | "tabletop_or_bed" | "slanted_plane"
    point_indices: np.ndarray
    normal: np.ndarray
    d_coeff: float
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    mad_before_m: float
    mad_after_m: float
    crossview_support_ratio: float
    supporting_views_count: int
    free_space_conflict_ratio: float
    accepted: bool
    reject_reason: Optional[str] = None
    furniture_instance_id: int = 0

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "region_id": int(self.region_id),
            "semantic_label": self.semantic_label,
            "num_points": int(len(self.point_indices)),
            "normal": [round(float(x), 4) for x in self.normal],
            "d_coeff": round(float(self.d_coeff), 5),
            "centroid": [round(float(x), 4) for x in self.centroid],
            "bbox_min": [round(float(x), 4) for x in self.bbox_min],
            "bbox_max": [round(float(x), 4) for x in self.bbox_max],
            "mad_before_m": round(float(self.mad_before_m), 5),
            "mad_after_m": round(float(self.mad_after_m), 5),
            "crossview_support_ratio": round(float(self.crossview_support_ratio), 3),
            "supporting_views_count": int(self.supporting_views_count),
            "free_space_conflict_ratio": round(float(self.free_space_conflict_ratio), 4),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
            "furniture_instance_id": int(self.furniture_instance_id),
        }


def run_yolo_indoor_segmentation_scheduled(
    frames_rgb: List[np.ndarray],
    infer_hw: Tuple[int, int],
    model_path: Optional[str] = None,
    device: str = "auto",
    log_callback: Optional[Callable[[str], None]] = None,
    return_furniture_masks: bool = False,
) -> Any:
    """
    Runs local YOLOv8n-seg sequentially with 8 GB VRAM cleanup to extract:
      - dynamic `person` masks (dilated slightly for safe background/tracker exclusion)
      - static indoor furniture instance/class masks (chair, dining table, bed, couch, tv, laptop, ...)
        so furniture objects are preserved and never merged into adjacent objects or walls.
    Guaranteed to unload YOLO weights & empty CUDA cache before returning.
    Returns `(dyn_masks, info)` by default, or `(dyn_masks, furn_cls_masks, furn_inst_masks, info)`
    when `return_furniture_masks=True`.
    """
    H_inf, W_inf = int(infer_hw[0]), int(infer_hw[1])
    resolved = model_path
    if resolved is None:
        for cand in DEFAULT_YOLO_SEG_PATHS:
            if Path(cand).is_file():
                resolved = cand
                break

    def _empty_return(reason: str):
        empty_dyn: List[Optional[np.ndarray]] = [None] * len(frames_rgb)
        info_err = {"available": False, "reason": reason}
        if return_furniture_masks:
            empty_cls = [np.full((H_inf, W_inf), -1, dtype=np.int16) for _ in frames_rgb]
            empty_inst = [np.zeros((H_inf, W_inf), dtype=np.int32) for _ in frames_rgb]
            return empty_dyn, empty_cls, empty_inst, info_err
        return empty_dyn, info_err

    if resolved is None or not Path(resolved).is_file():
        if log_callback:
            log_callback("  [*] YOLOv8n-seg weights not found locally; skipping dynamic person mask step.")
        return _empty_return("weights_not_found")

    try:
        from ultralytics import YOLO
    except ImportError:
        return _empty_return("ultralytics_not_installed")

    t0 = time.perf_counter()
    dev = ("cuda:0" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model = YOLO(resolved)
    name_map = {int(k): str(v) for k, v in model.names.items()}
    person_ids = {k for k, v in name_map.items() if v == "person"}
    furniture_ids = {k for k, v in name_map.items() if v in INDOOR_FURNITURE_CLASSES}
    furn_counts: Dict[str, int] = {}
    dyn_masks: List[Optional[np.ndarray]] = []
    furn_cls_masks: List[np.ndarray] = []
    furn_inst_masks: List[np.ndarray] = []
    frames_with_people = 0
    frames_with_furniture = 0
    global_inst_counter = 1

    try:
        for img_rgb in frames_rgb:
            # Ultralytics accepts BGR uint8 numpy cleanly
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            res = model.predict(source=bgr, conf=0.35, iou=0.45, device=dev, verbose=False)[0]
            mask_out = np.zeros((H_inf, W_inf), dtype=np.uint8)
            cls_map = np.full((H_inf, W_inf), -1, dtype=np.int16)
            inst_map = np.zeros((H_inf, W_inf), dtype=np.int32)
            has_person = False
            has_furn = False

            if res.boxes is not None and len(res.boxes) > 0:
                H_orig, W_orig = img_rgb.shape[:2]
                # Process larger detections first so smaller foreground items (laptop/chair) overwrite background bed/table
                confs = [float(res.boxes.conf[k].item()) for k in range(len(res.boxes))]
                det_order = sorted(range(len(res.boxes)), key=lambda k: confs[k])
                for k in det_order:
                    cid = int(res.boxes.cls[k].item())
                    cname = name_map.get(cid, str(cid))
                    m_res = None
                    if res.masks is not None and k < len(res.masks.data):
                        m_raw = res.masks.data[k].cpu().numpy()
                        m_res = cv2.resize((m_raw > 0.45).astype(np.uint8), (W_inf, H_inf), interpolation=cv2.INTER_NEAREST)
                    elif res.boxes is not None:
                        x1, y1, x2, y2 = res.boxes.xyxy[k].cpu().numpy()
                        sx = W_inf / max(float(W_orig), 1.0)
                        sy = H_inf / max(float(H_orig), 1.0)
                        ix1 = max(0, min(W_inf - 1, int(round(x1 * sx))))
                        ix2 = max(ix1 + 1, min(W_inf, int(round(x2 * sx))))
                        iy1 = max(0, min(H_inf - 1, int(round(y1 * sy))))
                        iy2 = max(iy1 + 1, min(H_inf, int(round(y2 * sy))))
                        m_res = np.zeros((H_inf, W_inf), dtype=np.uint8)
                        m_res[iy1:iy2, ix1:ix2] = 1

                    if m_res is None:
                        continue

                    if cid in person_ids:
                        mask_out = np.maximum(mask_out, m_res)
                        has_person = True
                    elif cid in furniture_ids:
                        furn_counts[cname] = furn_counts.get(cname, 0) + 1
                        # Never overwrite dynamic person carrier pixels as static furniture
                        valid_pix = (m_res > 0) & (mask_out == 0)
                        if np.any(valid_pix):
                            cls_map[valid_pix] = np.int16(cid)
                            inst_map[valid_pix] = np.int32(global_inst_counter)
                            global_inst_counter += 1
                            has_furn = True

            if has_person:
                mask_out = cv2.dilate(mask_out, np.ones((5, 5), dtype=np.uint8), iterations=1)
                frames_with_people += 1
                dyn_masks.append(mask_out)
                # Ensure dilated person border isn't flagged as furniture
                cls_map[mask_out > 0] = -1
                inst_map[mask_out > 0] = 0
            else:
                dyn_masks.append(None)

            if has_furn:
                frames_with_furniture += 1
            furn_cls_masks.append(cls_map)
            furn_inst_masks.append(inst_map)
    finally:
        del model
        gc.collect()
        if "cuda" in dev and torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = {
        "available": True,
        "weights_path": str(Path(resolved).resolve()),
        "device_used": dev,
        "frames_processed": len(frames_rgb),
        "frames_with_dynamic_person": frames_with_people,
        "frames_with_furniture_masks": frames_with_furniture,
        "total_furniture_instance_detections": global_inst_counter - 1,
        "indoor_furniture_detections": furn_counts,
        "class_id_to_name": {str(k): name_map[k] for k in sorted(furniture_ids | person_ids) if k in name_map},
        "protected_non_planar_class_ids": sorted(k for k, v in name_map.items() if v in PROTECTED_NON_PLANAR_NAMES),
        "planar_furniture_class_ids": sorted(k for k, v in name_map.items() if v in PLANAR_FURNITURE_NAMES),
        "duration_sec": round(time.perf_counter() - t0, 2),
    }
    if log_callback:
        log_callback(
            f"  [+] YOLO Indoor Semantic Audit ({info['duration_sec']}s): "
            f"person masks in {frames_with_people}/{len(frames_rgb)} frames, "
            f"furniture masks in {frames_with_furniture}/{len(frames_rgb)} frames, detections={furn_counts}"
        )
    if return_furniture_masks:
        return dyn_masks, furn_cls_masks, furn_inst_masks, info
    return dyn_masks, info


def associate_yolo_semantics_to_3d_points(
    points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    validated_depth_maps: np.ndarray,
    furn_cls_masks: List[np.ndarray],
    furn_inst_masks: Optional[List[np.ndarray]] = None,
    dynamic_masks: Optional[List[Optional[np.ndarray]]] = None,
    class_id_to_name: Optional[Dict[Any, str]] = None,
    max_reproj_depth_rel_diff: float = 0.075,
    min_depth: float = 0.18,
    max_depth: float = 5.50,
    instance_cluster_eps_m: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Projects canonical/cleaned 3D points into YOLO furniture segmentation masks across
    depth-visible cameras (ignoring any pixels flagged as moving `person`) and consolidates
    per-point semantic class IDs and spatially separated 3D furniture instance IDs.
    """
    pts = np.asarray(points, dtype=np.float64)
    N_pts = len(pts)
    point_sem_cls = np.full(N_pts, -1, dtype=np.int32)
    point_furn_inst = np.zeros(N_pts, dtype=np.int32)

    if N_pts == 0 or not furn_cls_masks:
        return point_sem_cls, point_furn_inst, {
            "labeled_points": 0,
            "labeled_fraction_pct": 0.0,
            "unique_3d_furniture_instances": 0,
            "class_point_counts": {},
        }

    d_arr = validated_depth_maps[..., 0] if validated_depth_maps.ndim == 4 else validated_depth_maps
    N_cams, H, W = d_arr.shape
    M_masks = min(N_cams, len(furn_cls_masks))

    # Track votes per point efficiently via sparse dictionary lists only on foreground hits
    vote_lists: List[List[int]] = [[] for _ in range(N_pts)]
    frame_inst_votes: List[List[int]] = [[] for _ in range(N_pts)]
    person_obs_hits = np.zeros(N_pts, dtype=np.int32)
    valid_obs_hits = np.zeros(N_pts, dtype=np.int32)

    for ci in range(M_masks):
        cls_m = furn_cls_masks[ci]
        if cls_m is None:
            continue
        inst_m = furn_inst_masks[ci] if (furn_inst_masks is not None and ci < len(furn_inst_masks)) else None
        dyn_m = dynamic_masks[ci] if (dynamic_masks is not None and ci < len(dynamic_masks)) else None

        R = extrinsics[ci, :3, :3].astype(np.float64)
        t = extrinsics[ci, :3, 3].astype(np.float64)
        K = intrinsics[ci].astype(np.float64)

        pc = (pts @ R.T) + t
        zc = pc[:, 2]
        in_z = (zc >= min_depth) & (zc <= max_depth)
        if not np.any(in_z):
            continue

        idx_z = np.where(in_z)[0]
        u = np.round(K[0, 0] * (pc[idx_z, 0] / zc[idx_z]) + K[0, 2]).astype(np.int32)
        v = np.round(K[1, 1] * (pc[idx_z, 1] / zc[idx_z]) + K[1, 2]).astype(np.int32)
        in_b = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_b):
            continue

        idx_v = idx_z[in_b]
        u_v, v_v = u[in_b], v[in_b]
        sd = d_arr[ci, v_v, u_v].astype(np.float64)
        vis_ok = (sd >= min_depth) & (np.abs(zc[idx_v] - sd) / np.maximum(zc[idx_v], 1e-3) <= max_reproj_depth_rel_diff)
        if not np.any(vis_ok):
            continue

        idx_vis = idx_v[vis_ok]
        u_vis, v_vis = u_v[vis_ok], v_v[vis_ok]
        valid_obs_hits[idx_vis] += 1

        if dyn_m is not None:
            p_hit = dyn_m[v_vis, u_vis] > 0
            person_obs_hits[idx_vis[p_hit]] += 1
            keep_static = ~p_hit
            idx_vis = idx_vis[keep_static]
            u_vis, v_vis = u_vis[keep_static], v_vis[keep_static]

        cids = cls_m[v_vis, u_vis]
        fg = cids >= 0
        if np.any(fg):
            fg_idx = idx_vis[fg]
            fg_cids = cids[fg]
            fg_insts = inst_m[v_vis[fg], u_vis[fg]] if inst_m is not None else np.zeros(len(fg_idx), dtype=np.int32)
            for p_i, c_i, ins_i in zip(fg_idx.tolist(), fg_cids.tolist(), fg_insts.tolist()):
                vote_lists[p_i].append(c_i)
                if ins_i > 0:
                    frame_inst_votes[p_i].append(ins_i)

    # Majority class assignment requiring majority of static foreground hits (and not majority dynamic person)
    modal_frame_inst = np.zeros(N_pts, dtype=np.int32)
    for i in range(N_pts):
        if valid_obs_hits[i] > 0 and person_obs_hits[i] > valid_obs_hits[i] * 0.5:
            continue
        votes = vote_lists[i]
        if not votes:
            continue
        # At least 1 hit with majority over total static views or >=2 consistent hits
        if len(votes) >= 2 or (valid_obs_hits[i] <= 2):
            vals, cnts = np.unique(votes, return_counts=True)
            best_c = int(vals[np.argmax(cnts)])
            if int(cnts.max()) >= max(1, len(votes) // 2):
                point_sem_cls[i] = best_c
                if frame_inst_votes[i]:
                    iv, ic = np.unique(frame_inst_votes[i], return_counts=True)
                    modal_frame_inst[i] = int(iv[np.argmax(ic)])

    # Assign 3D separated furniture instance IDs via spatial DBSCAN within each semantic class
    next_3d_inst = 1
    unique_cls = [int(c) for c in np.unique(point_sem_cls) if c >= 0]
    for cid in unique_cls:
        c_idx = np.where(point_sem_cls == cid)[0]
        if len(c_idx) < 15:
            point_sem_cls[c_idx] = -1
            continue
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(pts[c_idx])
        lbls = np.asarray(sub_pcd.cluster_dbscan(eps=instance_cluster_eps_m, min_points=10, print_progress=False))
        for lid in np.unique(lbls):
            if lid < 0:
                continue
            comp = c_idx[lbls == lid]
            if len(comp) < 15:
                continue
            # Also split inside same spatial cluster if modal 2D instances from identical frame strongly separate objects
            point_furn_inst[comp] = next_3d_inst
            next_3d_inst += 1

    # Clear unclustered noise points (< 15 pts) from semantic class so spurious 2D mask fringes don't corrupt
    point_sem_cls[(point_sem_cls >= 0) & (point_furn_inst == 0)] = -1

    name_lut: Dict[int, str] = {}
    if class_id_to_name:
        name_lut = {int(k): str(v) for k, v in class_id_to_name.items()}
    class_point_counts: Dict[str, int] = {}
    for cid in np.unique(point_sem_cls):
        if cid < 0:
            continue
        cname = name_lut.get(int(cid), f"class_{int(cid)}")
        class_point_counts[cname] = int(np.count_nonzero(point_sem_cls == cid))

    stats = {
        "labeled_points": int(np.count_nonzero(point_sem_cls >= 0)),
        "labeled_fraction_pct": round(float(np.count_nonzero(point_sem_cls >= 0)) / max(1, N_pts) * 100.0, 2),
        "unique_3d_furniture_instances": int(next_3d_inst - 1),
        "class_point_counts": class_point_counts,
    }
    return point_sem_cls, point_furn_inst, stats


def _fit_trimmed_plane(pts: np.ndarray, iters: int = 3) -> Tuple[np.ndarray, float]:
    """Iteratively trims outer 15% residuals via Huber/L1 reweighting and fits SVD plane n.p + d = 0."""
    X = np.asarray(pts, dtype=np.float64)
    if len(X) < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0
    mask = np.ones(len(X), dtype=bool)
    n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    d = 0.0
    for _ in range(iters):
        Xin = X[mask]
        if len(Xin) < 3:
            break
        mu = Xin.mean(axis=0)
        cov = np.cov((Xin - mu).T)
        if cov.ndim == 0:
            break
        eigvals, eigvecs = np.linalg.eigh(cov)
        n = eigvecs[:, 0]
        n /= max(np.linalg.norm(n), 1e-9)
        d = -float(np.dot(n, mu))
        dists = np.abs(X @ n + d)
        thresh = max(0.012, float(np.percentile(dists, 82)))
        mask = dists <= thresh
    return n, d


def _region_semantic_tag(
    n: np.ndarray,
    centroid: np.ndarray,
    true_floor_y: float,
    ceiling_y: float,
    y_bottom_extent: float,
) -> str:
    """Labels planar regions using leveled OpenCV world coordinates (Y down, larger Y = lower)."""
    abs_ny = abs(float(n[1]))
    if abs_ny >= 0.88:
        # True floor lies near lowest scene elevation (few or no structures extending >=12cm below it)
        if abs(float(centroid[1]) - true_floor_y) <= 0.09 and (y_bottom_extent - float(centroid[1])) < 0.12:
            return "floor"
        if centroid[1] <= ceiling_y + 0.18:
            return "ceiling"
        return "tabletop_or_bed"
    if abs_ny <= 0.35:
        return "wall"
    return "slanted_plane"


def _boundary_first_can_merge(
    reg_a: PlanarRegionRecord,
    reg_b: PlanarRegionRecord,
    points: np.ndarray,
    free_space_ev: Optional[FreeSpaceVoxelEvidence],
    cfg: PlanarRefinementConfig,
) -> Tuple[bool, str]:
    """
    Boundary-first merge verifier: prevents merging separate tabletops across gaps/aisles,
    prevents merging distinct YOLO furniture instances, and never merges perpendicular surfaces or tabletops into walls.
    """
    if reg_a.semantic_label != reg_b.semantic_label:
        return False, "different_semantic_class"

    if (
        reg_a.furniture_instance_id > 0
        and reg_b.furniture_instance_id > 0
        and reg_a.furniture_instance_id != reg_b.furniture_instance_id
    ):
        return False, "different_yolo_furniture_instance"

    # Check normal alignment (up to sign flip)
    dot_n = float(np.dot(reg_a.normal, reg_b.normal))
    if dot_n < 0:
        n_b = -reg_b.normal
        d_b = -reg_b.d_coeff
        dot_n = -dot_n
    else:
        n_b = reg_b.normal
        d_b = reg_b.d_coeff

    angle_deg = math.degrees(math.acos(np.clip(dot_n, -1.0, 1.0)))
    if angle_deg > cfg.merge_max_angle_deg:
        return False, f"normal_angle ({angle_deg:.1f}deg)"

    off_a = abs(float(np.dot(reg_a.normal, reg_b.centroid) + reg_a.d_coeff))
    off_b = abs(float(np.dot(n_b, reg_a.centroid) + d_b))
    if max(off_a, off_b) > cfg.merge_max_plane_offset_m:
        return False, f"plane_offset ({max(off_a, off_b)*100:.1f}cm)"

    # Bounding-box quick reject before exact KDTree boundary distance
    gap_vec = np.maximum(0.0, np.maximum(reg_a.bbox_min - reg_b.bbox_max, reg_b.bbox_min - reg_a.bbox_max))
    if float(np.linalg.norm(gap_vec)) > cfg.merge_max_gap_m * 1.5:
        return False, "disjoint_bboxes"

    pa = points[reg_a.point_indices]
    pb = points[reg_b.point_indices]
    tree_b = cKDTree(pb)
    dists, nn_idx = tree_b.query(pa, k=1)
    min_gap = float(np.min(dists))
    if min_gap > cfg.merge_max_gap_m:
        return False, f"physical_gap ({min_gap*100:.1f}cm > {cfg.merge_max_gap_m*100:.1f}cm)"

    # Free-space check right across nearest boundary bridge
    if free_space_ev is not None:
        i_min = int(np.argmin(dists))
        p0 = pa[i_min]
        p1 = pb[nn_idx[i_min]]
        bridges = np.array([0.25 * p0 + 0.75 * p1, 0.50 * p0 + 0.50 * p1, 0.75 * p0 + 0.25 * p1])
        if np.any(free_space_ev.query_free_space_mask(bridges)):
            return False, "free_space_barrier_between_regions"

    return True, "coplanar_continuous_adjacent"


def refine_scene_geometry_planar_gs(
    points: np.ndarray,
    colors: np.ndarray,
    cam_centers: List[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    depth_maps: np.ndarray,
    confidence_maps: np.ndarray,
    free_space_evidence: Optional[FreeSpaceVoxelEvidence] = None,
    config: Optional[PlanarRefinementConfig] = None,
    log_callback: Optional[Callable[[str], None]] = None,
    point_semantic_classes: Optional[np.ndarray] = None,
    point_furniture_instance_ids: Optional[np.ndarray] = None,
    class_id_to_name: Optional[Dict[Any, str]] = None,
) -> Dict[str, Any]:
    """
    Executes PlanarGS-inspired planar proposal, cross-view verification, boundary-first merge gating,
    YOLO furniture object isolation/protection, and conservative interior-tapered refinement.
    Never modifies camera poses.
    """
    t0 = time.perf_counter()
    cfg = config or PlanarRefinementConfig()
    pts = np.asarray(points, dtype=np.float64).copy()
    cols = np.asarray(colors, dtype=np.uint8).copy()

    if (not cfg.enabled) or len(pts) < 250:
        return {
            "refined_points": pts.astype(np.float32),
            "refined_colors": cols,
            "enabled": False,
            "regions": [],
            "thin_structure_points": np.zeros((0, 3), dtype=np.float32),
            "metrics": {
                "num_proposed_patches": 0,
                "num_accepted_planes": 0,
                "points_corrected": 0,
                "furniture_protected_points": 0,
                "duration_sec": 0.0,
            },
        }

    # Resolve YOLO semantic protection masks on 3D points (protect chair/couch/tv/etc. from plane swallowing)
    sem_cls_arr = np.asarray(point_semantic_classes, dtype=np.int32) if (
        point_semantic_classes is not None and len(point_semantic_classes) == len(pts)
    ) else np.full(len(pts), -1, dtype=np.int32)
    furn_inst_arr = np.asarray(point_furniture_instance_ids, dtype=np.int32) if (
        point_furniture_instance_ids is not None and len(point_furniture_instance_ids) == len(pts)
    ) else np.zeros(len(pts), dtype=np.int32)

    name_lut: Dict[int, str] = {}
    if class_id_to_name:
        name_lut = {int(k): str(v) for k, v in class_id_to_name.items()}
    else:
        # Standard COCO IDs fallback when class_id_to_name map isn't supplied
        name_lut = {
            13: "bench", 24: "backpack", 26: "handbag", 28: "suitcase",
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            60: "dining table", 62: "tv", 63: "laptop", 66: "keyboard",
            68: "microwave", 69: "oven", 71: "sink", 72: "refrigerator", 73: "book",
        }
    protected_cids = {cid for cid, cname in name_lut.items() if cname in PROTECTED_NON_PLANAR_NAMES}
    planar_furn_cids = {cid for cid, cname in name_lut.items() if cname in PLANAR_FURNITURE_NAMES}
    protected_nonplanar_mask = np.isin(sem_cls_arr, list(protected_cids)) if protected_cids else np.zeros(len(pts), dtype=bool)
    planar_furn_mask = np.isin(sem_cls_arr, list(planar_furn_cids)) if planar_furn_cids else np.zeros(len(pts), dtype=bool)
    furniture_protected_count = int(np.count_nonzero(protected_nonplanar_mask))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=cfg.voxel_size * 4.0, max_nn=30)
    )
    mean_cam = np.mean(cam_centers, axis=0) if cam_centers else pts.mean(axis=0)
    pcd.orient_normals_towards_camera_location(mean_cam)
    normals = np.asarray(pcd.normals, dtype=np.float64)

    # Infer lowest horizontal surface & ceiling bounds relative to camera centers (OpenCV Y-down)
    mean_cam_y = float(mean_cam[1])
    y_bottom_extent = float(np.percentile(pts[:, 1], 98))
    horiz_y_vals = pts[np.abs(normals[:, 1]) >= 0.88, 1]
    true_floor_y = float(np.percentile(horiz_y_vals, 96)) if len(horiz_y_vals) >= 40 else y_bottom_extent
    pts_above_cam = pts[pts[:, 1] < mean_cam_y - 0.18, 1]
    ceiling_y = float(np.percentile(pts_above_cam, 15)) if len(pts_above_cam) > 50 else float(mean_cam_y - 0.45)

    # 1. Partition points into normal-compatible bins, keeping non-planar furniture EXCLUDED from wall/floor bins
    eligible_for_planes = ~protected_nonplanar_mask
    horiz_mask = eligible_for_planes & (np.abs(normals[:, 1]) >= 0.88)
    vert_mask = eligible_for_planes & (~planar_furn_mask) & (np.abs(normals[:, 1]) <= 0.35)

    bins: List[np.ndarray] = []
    if np.count_nonzero(horiz_mask) >= cfg.min_patch_points:
        bins.append(np.where(horiz_mask)[0])
    # Quantize vertical wall normals into 8 centered orientation buckets (0 and 90 deg sit dead center!)
    if np.count_nonzero(vert_mask) >= cfg.min_patch_points:
        v_idx = np.where(vert_mask)[0]
        vn = normals[v_idx]
        ang = (np.arctan2(vn[:, 2], vn[:, 0]) + math.pi / 16.0) % math.pi
        bucket_ids = np.floor(ang / (math.pi / 8.0)).astype(int) % 8
        for b in range(8):
            sub = v_idx[bucket_ids == b]
            if len(sub) >= cfg.min_patch_points:
                bins.append(sub)

    # 2. Strict Euclidean connected-component clustering within each normal bucket
    proposed_regions: List[PlanarRegionRecord] = []
    d_arr = depth_maps[..., 0] if depth_maps.ndim == 4 else depth_maps
    N_cams = len(extrinsics)
    R_all = extrinsics[:, :3, :3].astype(np.float64)
    t_all = extrinsics[:, :3, 3].astype(np.float64)

    for b_indices in bins:
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(pts[b_indices])
        labels = np.asarray(sub_pcd.cluster_dbscan(eps=cfg.cluster_eps_m, min_points=5, print_progress=False))
        unique_lbls = [l for l in np.unique(labels) if l >= 0]
        for l_id in unique_lbls:
            comp_local = np.where(labels == l_id)[0]
            if len(comp_local) < cfg.min_patch_points:
                continue
            comp_global = b_indices[comp_local]
            p_sub = pts[comp_global]

            n_fit, d_fit = _fit_trimmed_plane(p_sub)
            # Align normal direction with average local point normal
            mean_pt_norm = normals[comp_global].mean(axis=0)
            if float(np.dot(n_fit, mean_pt_norm)) < 0:
                n_fit = -n_fit
                d_fit = -d_fit

            residuals = np.abs(p_sub @ n_fit + d_fit)
            mad_before = float(np.median(residuals))
            inlier_2cm = float(np.mean(residuals <= cfg.plane_dist_thresh_m))

            # Reject curved/cluttered non-planar components (chairs, rounded monitors, cables)
            if mad_before > cfg.max_patch_mad_m or inlier_2cm < 0.72:
                continue

            # Dominant YOLO furniture instance ID (keep tabletop/bed objects cleanly separated!)
            inst_vals = furn_inst_arr[comp_global]
            fg_inst = inst_vals[inst_vals > 0]
            dom_furn_inst = 0
            if len(fg_inst) >= max(10, int(0.25 * len(comp_global))):
                iv, ic = np.unique(fg_inst, return_counts=True)
                dom_furn_inst = int(iv[np.argmax(ic)])

            # Cross-view support verification on up to 200 sampled points across cameras
            step_s = max(1, len(p_sub) // 160)
            p_samp = p_sub[::step_s]
            n_samp = len(p_samp)
            cam_hits = np.zeros(n_samp, dtype=np.int32)
            supporting_views = 0
            for ci in range(N_cams):
                pc = (p_samp @ R_all[ci].T) + t_all[ci]
                zc = pc[:, 2]
                pos = zc > 0.2
                if np.count_nonzero(pos) < 5:
                    continue
                K = intrinsics[ci]
                uc = K[0, 0] * (pc[:, 0] / np.maximum(zc, 1e-4)) + K[0, 2]
                vc = K[1, 1] * (pc[:, 1] / np.maximum(zc, 1e-4)) + K[1, 2]
                sd, in_b = bilinear_sample_depth_vectorized(d_arr[ci], uc, vc)
                agree = pos & in_b & (np.abs(zc - sd) / np.maximum(zc, 1e-3) <= 0.065)
                if np.count_nonzero(agree) >= 5:
                    supporting_views += 1
                    cam_hits[agree] += 1

            cv_ratio = float(np.mean(cam_hits >= 1)) if N_cams > 0 else 1.0
            fse_conflict = float(np.mean(free_space_evidence.query_free_space_mask(p_samp))) if free_space_evidence is not None else 0.0

            centroid = p_sub.mean(axis=0)
            sem = _region_semantic_tag(n_fit, centroid, true_floor_y, ceiling_y, y_bottom_extent)
            # If YOLO explicitly recognizes bed / dining table surface on >=25% of patch, never misclassify as floor/wall
            if np.count_nonzero(planar_furn_mask[comp_global]) >= max(10, int(0.25 * len(comp_global))):
                sem = "tabletop_or_bed"

            accepted = bool(cv_ratio >= cfg.min_crossview_support_ratio and fse_conflict <= 0.12)
            reject_r = None
            if not accepted:
                reject_r = f"cv={cv_ratio:.2f}, fse_conflict={fse_conflict:.2f}"

            proposed_regions.append(
                PlanarRegionRecord(
                    region_id=len(proposed_regions),
                    semantic_label=sem,
                    point_indices=comp_global,
                    normal=n_fit,
                    d_coeff=d_fit,
                    centroid=centroid,
                    bbox_min=p_sub.min(axis=0),
                    bbox_max=p_sub.max(axis=0),
                    mad_before_m=mad_before,
                    mad_after_m=mad_before,
                    crossview_support_ratio=cv_ratio,
                    supporting_views_count=supporting_views,
                    free_space_conflict_ratio=fse_conflict,
                    accepted=accepted,
                    reject_reason=reject_r,
                    furniture_instance_id=dom_furn_inst,
                )
            )

    # 3. Boundary-First Merge Check across accepted coplanar adjacent regions
    accepted_regs = [r for r in proposed_regions if r.accepted]
    rejected_merge_count = 0
    merged_pairs_count = 0

    parent = {r.region_id: r.region_id for r in accepted_regs}

    def _find(rid: int) -> int:
        while parent[rid] != rid:
            parent[rid] = parent[parent[rid]]
            rid = parent[rid]
        return rid

    for i_a in range(len(accepted_regs)):
        for i_b in range(i_a + 1, len(accepted_regs)):
            ra, rb = accepted_regs[i_a], accepted_regs[i_b]
            if _find(ra.region_id) == _find(rb.region_id):
                continue
            ok, reason = _boundary_first_can_merge(ra, rb, pts, free_space_evidence, cfg)
            if ok:
                parent[_find(rb.region_id)] = _find(ra.region_id)
                merged_pairs_count += 1
            elif reason in ("physical_gap", "free_space_barrier_between_regions", "different_yolo_furniture_instance") or "gap" in reason or "barrier" in reason:
                rejected_merge_count += 1

    # Consolidate merged clusters
    groups: Dict[int, List[PlanarRegionRecord]] = {}
    for r in accepted_regs:
        groups.setdefault(_find(r.region_id), []).append(r)

    final_regions: List[PlanarRegionRecord] = []
    planar_member_mask = np.zeros(len(pts), dtype=bool)
    for root_id, members in groups.items():
        all_idx = np.unique(np.concatenate([m.point_indices for m in members]))
        planar_member_mask[all_idx] = True
        p_grp = pts[all_idx]
        n_f, d_f = _fit_trimmed_plane(p_grp)
        if float(np.dot(n_f, members[0].normal)) < 0:
            n_f, d_f = -n_f, -d_f
        res = np.abs(p_grp @ n_f + d_f)
        dom_inst = max((m.furniture_instance_id for m in members), default=0)
        final_regions.append(
            PlanarRegionRecord(
                region_id=len(final_regions),
                semantic_label=members[0].semantic_label,
                point_indices=all_idx,
                normal=n_f,
                d_coeff=d_f,
                centroid=p_grp.mean(axis=0),
                bbox_min=p_grp.min(axis=0),
                bbox_max=p_grp.max(axis=0),
                mad_before_m=float(np.median(res)),
                mad_after_m=float(np.median(res)),
                crossview_support_ratio=float(np.mean([m.crossview_support_ratio for m in members])),
                supporting_views_count=max(m.supporting_views_count for m in members),
                free_space_conflict_ratio=max(m.free_space_conflict_ratio for m in members),
                accepted=True,
                furniture_instance_id=dom_inst,
            )
        )

    # 4. Conservative Interior-Tapered Plane Correction (boundary edges & furniture untouched!)
    non_member_idx = np.where(~planar_member_mask)[0]
    non_member_tree = cKDTree(pts[non_member_idx]) if len(non_member_idx) > 0 else None

    pts_corr = pts.copy()
    corrected_count = 0
    shifts_mm: List[float] = []
    boundary_untouched = 0

    for reg in final_regions:
        idx = reg.point_indices
        p_reg = pts[idx]
        signed_d = (p_reg @ reg.normal + reg.d_coeff)

        # Boundary detection: BOTH proximity to non-planar objects AND outer perimeter of this plane!
        dist_nonmem = non_member_tree.query(p_reg, k=1)[0] if non_member_tree is not None else np.full(len(p_reg), 99.0)
        outlier_local = np.where(np.abs(signed_d) > cfg.plane_dist_thresh_m)[0]
        if len(outlier_local) > 0:
            dist_out = cKDTree(p_reg[outlier_local]).query(p_reg, k=1)[0]
            dist_bnd = np.minimum(dist_nonmem, dist_out)
        else:
            dist_bnd = dist_nonmem

        # Taper blend: 0 within hard_margin (1.5cm), smoothly ramps over boundary_taper_m (5.5cm)
        t_raw = np.clip((dist_bnd - cfg.boundary_hard_margin_m) / max(cfg.boundary_taper_m, 1e-4), 0.0, 1.0)
        smooth_t = (t_raw * t_raw * (3.0 - 2.0 * t_raw)) * cfg.max_blend_strength

        in_range = (np.abs(signed_d) <= cfg.max_correction_m) & (~protected_nonplanar_mask[idx])
        delta = np.where(in_range & (smooth_t > 0.01), smooth_t * signed_d, 0.0)
        p_cand = p_reg - delta[:, None] * reg.normal[None, :]

        # Free-space veto: never push corrected points into KNOWN_FREE_SPACE
        if free_space_evidence is not None and np.any(delta != 0.0):
            fse_bad = free_space_evidence.query_free_space_mask(p_cand)
            delta = np.where(fse_bad, 0.0, delta)
            p_cand = p_reg - delta[:, None] * reg.normal[None, :]

        pts_corr[idx] = p_cand
        moved = np.abs(delta) > 1e-5
        corrected_count += int(np.count_nonzero(moved))
        boundary_untouched += int(np.count_nonzero(~moved))
        if np.any(moved):
            shifts_mm.extend((np.abs(delta[moved]) * 1000.0).tolist())

        # Updated MAD after conservative correction
        res_after = np.abs(p_cand @ reg.normal + reg.d_coeff)
        reg.mad_after_m = float(np.median(res_after))

    dur = round(time.perf_counter() - t0, 3)
    tabletop_mads_before = [r.mad_before_m for r in final_regions if r.semantic_label == "tabletop_or_bed"]
    tabletop_mads_after = [r.mad_after_m for r in final_regions if r.semantic_label == "tabletop_or_bed"]
    wall_mads_before = [r.mad_before_m for r in final_regions if r.semantic_label == "wall"]
    wall_mads_after = [r.mad_after_m for r in final_regions if r.semantic_label == "wall"]

    # Collect protected non-planar furniture + unmerged edge non-member structures for thin-structure QA
    thin_struct_mask = protected_nonplanar_mask | (~planar_member_mask & (furn_inst_arr > 0))
    thin_structure_points = pts_corr[thin_struct_mask].astype(np.float32)

    metrics = {
        "num_proposed_patches": len(proposed_regions),
        "num_accepted_planes_before_merge": len(accepted_regs),
        "num_accepted_planes_final": len(final_regions),
        "coplanar_merges_performed": merged_pairs_count,
        "boundary_gap_merges_blocked": rejected_merge_count,
        "points_corrected": corrected_count,
        "points_boundary_preserved": boundary_untouched,
        "furniture_protected_points": furniture_protected_count,
        "thin_structure_points_tracked": int(len(thin_structure_points)),
        "mean_correction_mm": round(float(np.mean(shifts_mm)) if shifts_mm else 0.0, 3),
        "max_correction_mm": round(float(np.max(shifts_mm)) if shifts_mm else 0.0, 3),
        "tabletop_mad_before_mm": round(float(np.median(tabletop_mads_before) * 1000.0), 3) if tabletop_mads_before else None,
        "tabletop_mad_after_mm": round(float(np.median(tabletop_mads_after) * 1000.0), 3) if tabletop_mads_after else None,
        "wall_mad_before_mm": round(float(np.median(wall_mads_before) * 1000.0), 3) if wall_mads_before else None,
        "wall_mad_after_mm": round(float(np.median(wall_mads_after) * 1000.0), 3) if wall_mads_after else None,
        "duration_sec": dur,
    }
    if log_callback:
        log_callback(
            f"  [+] PlanarGS Refinement ({dur:.2f}s): {len(final_regions)} verified planes "
            f"({merged_pairs_count} adjacent merges, {rejected_merge_count} gap-blocked) | "
            f"Corrected {corrected_count:,} interior pts (mean={metrics['mean_correction_mm']:.2f}mm, "
            f"furniture_protected={furniture_protected_count:,}, tabletop MAD: {metrics['tabletop_mad_before_mm']}->{metrics['tabletop_mad_after_mm']}mm)"
        )

    return {
        "refined_points": pts_corr.astype(np.float32),
        "refined_colors": cols,
        "enabled": True,
        "regions": final_regions,
        "thin_structure_points": thin_structure_points,
        "metrics": metrics,
    }


def project_canonical_points_to_depth_maps(
    canonical_points: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    validated_depth_maps: np.ndarray,
    validated_weight_maps: np.ndarray,
    max_depth_delta_m: float = 0.06,
    min_depth: float = 0.20,
    max_depth: float = 5.50,
    splat_radius_px: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Synchronizes TSDF input depths directly with the canonical (possibly planar-refined) point cloud
    via visibility-aware z-buffered camera projection and edge-safe sub-voxel splat propagation.

    Priority rules:
      1. Exact pixel nearest-depth projection (`du=0, dv=0`) claims its pixel whenever valid within
         `max_depth_delta_m`.
      2. If `splat_radius_px >= 1`, un-hit valid neighboring pixels within cardinal then diagonal
         offsets receive the canonical depth ONLY when BOTH the projected depth and raw source-to-
         neighbor depth stay within `max_depth_delta_m` (never crossing foreground/background edges).
      3. Contradicted / invalid pixels (`~valid_cur`) are never hallucinated.
    """
    pts = np.asarray(canonical_points, dtype=np.float64)
    d_base = validated_depth_maps[..., 0] if validated_depth_maps.ndim == 4 else validated_depth_maps
    N, H, W = d_base.shape
    out_depths = np.zeros((N, H, W, 1), dtype=np.float32)
    out_weights = validated_weight_maps.astype(np.float32).copy()

    updated_exact = 0
    updated_splat = 0
    total_valid_pixels = 0
    per_cam_proj_counts = []

    splat_offsets: List[Tuple[int, int]] = []
    if splat_radius_px >= 1:
        splat_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i in range(N):
        R = extrinsics[i, :3, :3].astype(np.float64)
        t = extrinsics[i, :3, 3].astype(np.float64)
        K = intrinsics[i].astype(np.float64)
        d_raw = d_base[i].astype(np.float64)
        d_cur = d_raw.copy()
        valid_cur = (d_cur >= min_depth) & (d_cur <= max_depth)
        total_valid_pixels += int(np.count_nonzero(valid_cur))

        if len(pts) > 0:
            pc = (pts @ R.T) + t
            zc = pc[:, 2]
            in_front = (zc >= min_depth) & (zc <= max_depth)
            if np.any(in_front):
                uf = np.round(K[0, 0] * (pc[in_front, 0] / zc[in_front]) + K[0, 2]).astype(np.int32)
                vf = np.round(K[1, 1] * (pc[in_front, 1] / zc[in_front]) + K[1, 2]).astype(np.int32)
                zb = zc[in_front]
                in_b = (uf >= 0) & (uf < W) & (vf >= 0) & (vf < H)
                uf, vf, zb = uf[in_b], vf[in_b], zb[in_b]

                # Sort nearest-first so foreground points claim z-buffer pixel first
                order = np.argsort(zb)
                uf_s, vf_s, zb_s = uf[order], vf[order], zb[order]
                lin_idx = vf_s * W + uf_s
                uniq_lin, first_pos = np.unique(lin_idx, return_index=True)

                vy = uniq_lin // W
                vx = uniq_lin % W
                z_nearest = zb_s[first_pos]

                # Pass 1: exact z-buffered hits within same depth layer
                cur_at_pix = d_raw[vy, vx]
                can_update = valid_cur[vy, vx] & (np.abs(z_nearest - cur_at_pix) <= max_depth_delta_m)
                vy_e = vy[can_update]
                vx_e = vx[can_update]
                zn_e = z_nearest[can_update]
                d_cur[vy_e, vx_e] = zn_e
                updated_exact += int(len(vy_e))
                per_cam_proj_counts.append(len(vy_e))

                # Pass 2: edge-safe cardinal -> diagonal propagation for un-hit pixels
                if splat_offsets and len(vy_e) > 0:
                    claimed = np.zeros((H, W), dtype=bool)
                    claimed[vy_e, vx_e] = True
                    d_src_e = d_raw[vy_e, vx_e]
                    # Process nearest canonical depth first when multiple neighbors map to same un-hit pixel
                    for du, dv in splat_offsets:
                        vx_n = vx_e + du
                        vy_n = vy_e + dv
                        ib = (vx_n >= 0) & (vx_n < W) & (vy_n >= 0) & (vy_n < H)
                        if not np.any(ib):
                            continue
                        ux, uy = vx_n[ib], vy_n[ib]
                        z_s, ds_s = zn_e[ib], d_src_e[ib]
                        d_nbr = d_raw[uy, ux]
                        allow = (
                            (~claimed[uy, ux])
                            & valid_cur[uy, ux]
                            & (np.abs(z_s - d_nbr) <= max_depth_delta_m)
                            & (np.abs(ds_s - d_nbr) <= max_depth_delta_m)
                        )
                        if np.any(allow):
                            ux_a, uy_a, zs_a = ux[allow], uy[allow], z_s[allow]
                            # Sort by ascending |z_s - d_raw| if colliding inside same offset
                            d_cur[uy_a, ux_a] = zs_a
                            claimed[uy_a, ux_a] = True
                            updated_splat += int(len(np.unique(uy_a * W + ux_a)))

        out_depths[i, ..., 0] = np.where(valid_cur, d_cur, 0.0).astype(np.float32)

    updated_total = updated_exact + updated_splat
    stats = {
        "total_valid_pixels": total_valid_pixels,
        "updated_exact_pixels": updated_exact,
        "updated_splat_pixels": updated_splat,
        "updated_from_canonical_points": updated_total,
        "update_fraction_pct": round(updated_total / max(1, total_valid_pixels) * 100.0, 2),
        "exact_update_fraction_pct": round(updated_exact / max(1, total_valid_pixels) * 100.0, 2),
        "splat_radius_px": int(splat_radius_px),
    }
    return out_depths, out_weights, stats
