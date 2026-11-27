"""
F.O.R.G.E. V2 — Global Scalable TSDF Reconstruction, Conditional Localized Poisson Repair,
and Physical Evidence-Based Mesh Quality Assurance.

Architecture Rules Enforced:
1. One mandatory global Open3D ScalableTSDFVolume integrating all accepted poses & corrected
   depth maps (the legacy 6-submap TSDF branch and automatic submap switching are removed).
2. Original observed-only global TSDF mesh is ALWAYS preserved (`mesh_global_tsdf_observed.ply`),
   even when QA fails.
3. Conditional Poisson is NEVER run routinely across the entire scene. Instead, boundary holes
   are classified against voxel size, local support points/normals, azimuthal perimeter enclosure,
   occlusion, and KNOWN_FREE_SPACE evidence:
     - Case A (Small cracks/holes): conservative local triangle closing only; NO Poisson.
     - Case B (Qualifying larger interior gap with dense consistent support & no free space):
       localized Screened Poisson on ONLY the local support ring, patch cropped strictly to gap
       interior, saved separately, stitched onto a separate copy, and accepted ONLY if full Mesh
       QA passes with zero degradation.
     - Case C (Open doorways / perimeter rim / unsupported regions / known free space): NEVER filled.
4. Comprehensive Mesh QA: double-shell %, normal deviation, wall/floor flatness MAD, holes,
   free-space violations, thin-structure preservation, rendered-depth agreement, and topology.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import open3d as o3d
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from src.reconstruction.conditional_gap_repair import (
    assemble_stitched_patch_into_mesh,
    build_boundary_pinned_planar_patch,
    build_polygon_clipped_stitched_poisson_patch,
    render_before_after_repair_comparison,
    render_gap_diagnostic_views,
    verify_patch_local_qa,
)
from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence


def _get_process_rss_mb() -> float:
    """Returns current process RSS memory in MB using stdlib/ctypes or psutil if present."""
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2), 2)
    except Exception:
        return 0.0


@dataclass
class CanonicalSceneGeometry:
    """Single canonical corrected 3D representation linking points, normals, semantics & depths."""

    points: np.ndarray                      # (M, 3) float32
    colors: np.ndarray                      # (M, 3) uint8
    normals: np.ndarray                     # (M, 3) float32 ray-oriented
    confidences: np.ndarray                 # (M,) float32
    synchronized_depth_maps: np.ndarray     # (N, H, W, 1) float32
    validated_weight_maps: np.ndarray       # (N, H, W) float32
    extrinsics: np.ndarray                  # (N, 3, 4) float64
    intrinsics: np.ndarray                  # (N, 3, 3) float64
    cam_centers: List[np.ndarray]
    voxel_size: float
    planar_refinement_applied: bool
    free_space_evidence: Optional[FreeSpaceVoxelEvidence] = None
    point_semantic_classes: Optional[np.ndarray] = None        # (M,) int32 COCO class id (-1=none)
    point_furniture_instance_ids: Optional[np.ndarray] = None  # (M,) int32 instance id (-1=none)
    thin_structure_points: Optional[np.ndarray] = None         # (K, 3) float32 thin-structure reference


@dataclass
class MeshBranchQAConfig:
    voxel_size: float = 0.012
    poisson_depth: int = 8                  # Localized patch Poisson depth (NEVER full-room depth=10 bloat)
    density_trim_quantile: float = 0.02
    min_cluster_faces: int = 25
    tsdf_voxel_length: float = 0.012
    tsdf_trunc: float = 0.048               # 4x voxel truncation preserves grazing & slightly noisy surfaces
    min_depth: float = 0.20
    max_depth: float = 5.50
    min_confidence: float = 0.65
    enable_conditional_repair: bool = True
    # Continuous Case A / B1 (Planar CDT) / B2 (Curved Poisson) / C (Preserve) thresholds
    small_hole_max_voxel_diam: float = 5.0
    case_a_max_loop_verts: int = 24
    case_b_min_area_m2: float = 0.0065      # Cleanly keeps tiny pinholes (<=small_thresh & <0.0065m2) in Case A while B1 handles medium/large gaps with zero dead zone
    large_hole_max_diam_m: float = 0.90
    max_case_b_patches: int = 40            # Cap localized Case B (B1/B2) patches per scene to biggest qualifying gaps
    max_case_a_patches: int = 250
    case_b_min_support_pts: int = 60
    case_b_min_enclosed_sectors: int = 6    # Out of 8 azimuthal sectors (blocks outer boundary edges!)
    case_b_max_normal_std_deg: float = 30.0
    b1_planar_mad_max_m: float = 0.018
    b1_max_normal_std_deg: float = 20.0
    b2_band_halfwidth_vox: float = 2.6
    case_c_max_freespace_ratio: float = 0.16
    patch_qa_max_dbl_shell_delta: float = 0.008
    # QA Pass/Fail thresholds (never forced!)
    qa_min_triangles: int = 1000
    qa_min_obs_support_ratio: float = 0.78
    qa_min_completeness_ratio: float = 0.62
    qa_max_freespace_violation_ratio: float = 0.14
    qa_max_bridging_severity: float = 30.0
    qa_max_rendered_depth_rel_err: float = 0.12
    qa_max_non_manifold_edge_ratio: float = 0.03
    qa_max_double_shell_ratio: float = 0.25

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MeshCandidateQAReport:
    candidate_name: str
    method: str
    vertex_count: int
    triangle_count: int
    is_watertight: bool
    open_boundary_edges: int
    non_manifold_edges: int
    non_manifold_vertices: int
    connected_components: int
    fragmentation_ratio: float
    hole_count: int
    hole_area_m2: float
    obs_support_ratio: float
    obs_dist_median_m: float
    unsupported_surface_ratio: float
    surface_completeness_ratio: float
    free_space_violation_ratio: float
    bridging_severity_score: float
    duplicate_wall_ratio: float
    double_shell_ratio: float
    local_normal_median_deg: float
    noisy_normal_nbr_fraction: float
    wall_flatness_mad_mm: Optional[float]
    floor_flatness_mad_mm: Optional[float]
    thin_structure_preservation_ratio: float
    rendered_depth_median_rel_err: float
    rendered_depth_inlier_ratio: float
    composite_quality_score: float
    passes_qa: bool
    failure_reasons: List[str] = field(default_factory=list)
    build_time_sec: float = 0.0
    qa_time_sec: float = 0.0
    saved_ply_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_canonical_scene_geometry(
    points: np.ndarray,
    colors: np.ndarray,
    cam_centers: List[np.ndarray],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    synchronized_depth_maps: np.ndarray,
    validated_weight_maps: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    voxel_size: float = 0.012,
    planar_refinement_applied: bool = False,
    free_space_evidence: Optional[FreeSpaceVoxelEvidence] = None,
    point_semantic_classes: Optional[np.ndarray] = None,
    point_furniture_instance_ids: Optional[np.ndarray] = None,
    thin_structure_points: Optional[np.ndarray] = None,
) -> CanonicalSceneGeometry:
    """Constructs the single canonical geometry object with ray-oriented surface normals."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.02, voxel_size * 4.0), max_nn=30)
    )
    mean_cam = np.mean(cam_centers, axis=0) if cam_centers else np.mean(points, axis=0)
    pcd.orient_normals_towards_camera_location(mean_cam)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    return CanonicalSceneGeometry(
        points=np.asarray(points, dtype=np.float32),
        colors=np.asarray(colors, dtype=np.uint8),
        normals=normals,
        confidences=np.asarray(confidences, dtype=np.float32) if confidences is not None else np.ones(len(points), dtype=np.float32),
        synchronized_depth_maps=np.asarray(synchronized_depth_maps, dtype=np.float32),
        validated_weight_maps=np.asarray(validated_weight_maps, dtype=np.float32),
        extrinsics=np.asarray(extrinsics, dtype=np.float64),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        cam_centers=cam_centers,
        voxel_size=float(voxel_size),
        planar_refinement_applied=bool(planar_refinement_applied),
        free_space_evidence=free_space_evidence,
        point_semantic_classes=point_semantic_classes,
        point_furniture_instance_ids=point_furniture_instance_ids,
        thin_structure_points=thin_structure_points,
    )


def _clean_mesh_topology_with_stats(
    mesh: o3d.geometry.TriangleMesh,
    min_cluster_faces: int,
    support_points: Optional[np.ndarray] = None,
    support_radius_m: float = 0.028,
    min_supported_cluster_faces: int = 4,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, int]]:
    """
    Cleans degenerate/non-manifold elements and trims disconnected noise while preserving
    real fragmented clusters (>= `min_supported_cluster_faces`) backed by canonical points.
    """
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    stat = {"preserved_small_clusters": 0, "preserved_small_faces": 0, "removed_unsupported_faces": 0}
    if len(mesh.triangles) > 0 and min_cluster_faces > 0:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        cl_np = np.asarray(clusters)
        counts_np = np.asarray(counts)
        keep_ids: Set[int] = set(np.where(counts_np >= min_cluster_faces)[0].tolist())

        if support_points is not None and len(support_points) > 0:
            small_cids = np.where((counts_np >= min_supported_cluster_faces) & (counts_np < min_cluster_faces))[0]
            if len(small_cids) > 0:
                verts_np = np.asarray(mesh.vertices, dtype=np.float64)
                tris_np = np.asarray(mesh.triangles, dtype=np.int32)
                tri_cents = verts_np[tris_np].mean(axis=1)
                small_face_mask = np.isin(cl_np, small_cids)
                pt_tree = cKDTree(np.asarray(support_points, dtype=np.float64))
                dists, _ = pt_tree.query(tri_cents[small_face_mask], k=1)
                supported_cids = np.unique(cl_np[small_face_mask][dists <= support_radius_m])
                if len(supported_cids) > 0:
                    stat["preserved_small_clusters"] = int(len(supported_cids))
                    stat["preserved_small_faces"] = int(np.sum(counts_np[supported_cids]))
                    keep_ids.update(supported_cids.tolist())

        if len(keep_ids) > 0:
            mask = np.isin(cl_np, list(keep_ids))
            stat["removed_unsupported_faces"] = int(np.count_nonzero(~mask))
            mesh.remove_triangles_by_mask(~mask)
            mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh, stat


def _clean_mesh_topology(
    mesh: o3d.geometry.TriangleMesh,
    min_cluster_faces: int,
    support_points: Optional[np.ndarray] = None,
    support_radius_m: float = 0.028,
) -> o3d.geometry.TriangleMesh:
    cleaned, _ = _clean_mesh_topology_with_stats(
        mesh, min_cluster_faces=min_cluster_faces, support_points=support_points, support_radius_m=support_radius_m
    )
    return cleaned


def build_global_tsdf_branch(
    geom: CanonicalSceneGeometry,
    frames_bgr: List[np.ndarray],
    cfg: MeshBranchQAConfig,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[o3d.geometry.TriangleMesh, Dict[str, Any]]:
    """
    Mandatory single global Open3D ScalableTSDFVolume integration across all accepted views.
    Uses evidence-preserving cluster filtering and reports canonical geometry retention.
    Never switches to or falls back to submap TSDF chunking.
    """
    t0 = time.perf_counter()
    ram_before = _get_process_rss_mb()
    d_maps = geom.synchronized_depth_maps
    N, H_inf, W_inf = d_maps.shape[:3]

    sdf_trunc = max(0.025, float(cfg.tsdf_trunc))
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=float(cfg.tsdf_voxel_length),
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    if log_callback:
        log_callback(
            f"  [*] Running Mandatory Global ScalableTSDFVolume (N={N} frames, "
            f"voxel={cfg.tsdf_voxel_length*1000:.1f}mm, sdf_trunc={sdf_trunc*1000:.1f}mm)..."
        )

    integrated_frames = 0
    for i in range(N):
        im_rgb = cv2.cvtColor(cv2.resize(frames_bgr[i], (W_inf, H_inf)), cv2.COLOR_BGR2RGB)
        d = d_maps[i].squeeze()
        w = geom.validated_weight_maps[i].squeeze()
        valid = (d >= cfg.min_depth) & (d <= cfg.max_depth) & (w > 0.0)
        if not np.any(valid):
            continue
        d_clean = np.where(valid, d, 0.0).astype(np.float32)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(im_rgb),
            o3d.geometry.Image(d_clean),
            depth_scale=1.0,
            depth_trunc=float(cfg.max_depth),
            convert_rgb_to_intensity=False,
        )
        K = geom.intrinsics[i]
        pin = o3d.camera.PinholeCameraIntrinsic(
            W_inf, H_inf, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        ext4 = np.eye(4, dtype=np.float64)
        ext4[:3, :] = geom.extrinsics[i]
        vol.integrate(rgbd, pin, ext4)
        integrated_frames += 1

    mesh = vol.extract_triangle_mesh()
    ram_peak_est = max(ram_before, _get_process_rss_mb())
    mesh, trim_stats = _clean_mesh_topology_with_stats(
        mesh,
        cfg.min_cluster_faces,
        support_points=geom.points,
        support_radius_m=max(0.026, cfg.tsdf_voxel_length * 2.2),
    )
    dur = round(time.perf_counter() - t0, 3)

    # Measure exact canonical point retention on the extracted observed TSDF surface
    pts_obs = geom.points.astype(np.float64)
    canon_tot = int(len(pts_obs))
    within_2vox_pct = 0.0
    within_3vox_pct = 0.0
    lost_gt3vox = canon_tot
    thin_lost_pct = 0.0
    if len(mesh.triangles) > 0 and canon_tot > 0:
        n_samp = min(250000, max(50000, len(mesh.triangles) * 2))
        samp = mesh.sample_points_uniformly(number_of_points=n_samp)
        surf_pts = np.vstack([np.asarray(mesh.vertices, dtype=np.float64), np.asarray(samp.points, dtype=np.float64)])
        surf_tree = cKDTree(surf_pts)
        d_pts, _ = surf_tree.query(pts_obs, k=1)
        t2 = max(cfg.tsdf_voxel_length * 2.0, 0.024)
        t3 = max(cfg.tsdf_voxel_length * 3.0, 0.036)
        within_2vox_pct = round(float(np.mean(d_pts <= t2) * 100.0), 2)
        within_3vox_pct = round(float(np.mean(d_pts <= t3) * 100.0), 2)
        lost_gt3vox = int(np.count_nonzero(d_pts > t3))
        if geom.thin_structure_points is not None and len(geom.thin_structure_points) > 0:
            d_thin, _ = surf_tree.query(geom.thin_structure_points.astype(np.float64), k=1)
            thin_lost_pct = round(float(np.mean(d_thin > t3) * 100.0), 2)

    stats = {
        "method": "global_scalable_tsdf",
        "submap_tsdf_enabled": False,
        "num_submaps": 1,
        "frames_total": N,
        "frames_integrated": integrated_frames,
        "voxel_length_m": float(cfg.tsdf_voxel_length),
        "sdf_trunc_m": sdf_trunc,
        "ram_before_mb": ram_before,
        "ram_after_mb": _get_process_rss_mb(),
        "ram_peak_est_mb": ram_peak_est,
        "duration_sec": dur,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.triangles)),
        "canonical_points_total": canon_tot,
        "canonical_points_within_2vox_pct": within_2vox_pct,
        "canonical_points_within_3vox_pct": within_3vox_pct,
        "canonical_points_lost_gt3vox": lost_gt3vox,
        "canonical_points_lost_pct": round(lost_gt3vox / max(1, canon_tot) * 100.0, 2),
        "thin_points_lost_pct": thin_lost_pct,
        **trim_stats,
    }
    return mesh, stats


def analyze_tsdf_boundary_holes(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float,
) -> List[Dict[str, Any]]:
    """
    Extracts closed ordered CCW open-boundary edge loops from a triangle mesh, untangling
    branching vertices (>2 boundary edges) cleanly, and computes 3D centroid, diameter,
    projected polygon area, perimeter, loop planarity MAD, and local tangent basis.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    if len(verts) == 0 or len(tris) == 0:
        return []

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    vnorms = np.asarray(mesh.vertex_normals, dtype=np.float64)

    e0 = np.sort(tris[:, [0, 1]], axis=1)
    e1 = np.sort(tris[:, [1, 2]], axis=1)
    e2 = np.sort(tris[:, [2, 0]], axis=1)
    all_e = np.vstack([e0, e1, e2])
    uniq_e, counts = np.unique(all_e, axis=0, return_counts=True)
    bound_e = uniq_e[counts == 1]
    if len(bound_e) == 0:
        return []

    adj: Dict[int, List[int]] = {}
    for u, v in bound_e:
        u_i, v_i = int(u), int(v)
        adj.setdefault(u_i, []).append(v_i)
        adj.setdefault(v_i, []).append(u_i)

    visited_edges = set()
    loops: List[List[int]] = []

    for start_v in adj:
        for nb_start in adj[start_v]:
            ek = (min(start_v, nb_start), max(start_v, nb_start))
            if ek in visited_edges:
                continue
            loop = [start_v]
            prev = start_v
            curr = nb_start
            visited_edges.add(ek)
            closed = False
            for _ in range(min(4000, len(bound_e) + 2)):
                loop.append(curr)
                nbs = [x for x in adj.get(curr, []) if x != prev]
                if not nbs:
                    break
                # Pick best unvisited next vertex: prefer returning to start_v after >=3 steps, or smoothest coplanar continuation
                unvis = [cand for cand in nbs if (min(curr, cand), max(curr, cand)) not in visited_edges]
                if not unvis:
                    if start_v in nbs and len(loop) >= 3:
                        closed = True
                    break
                if len(unvis) == 1:
                    nxt = unvis[0]
                elif start_v in unvis and len(loop) >= 3:
                    nxt = start_v
                else:
                    v_in = verts[curr] - verts[prev]
                    v_in /= max(np.linalg.norm(v_in), 1e-8)
                    vn_c = vnorms[curr]
                    best_score = -1e9
                    nxt = unvis[0]
                    for cand in unvis:
                        v_out = verts[cand] - verts[curr]
                        v_out /= max(np.linalg.norm(v_out), 1e-8)
                        n_align = abs(float(np.dot(vnorms[cand], vn_c)))
                        score = float(np.dot(v_in, v_out)) + 0.5 * n_align
                        if score > best_score:
                            best_score = score
                            nxt = cand

                if nxt == start_v:
                    visited_edges.add((min(curr, nxt), max(curr, nxt)))
                    closed = True
                    break
                visited_edges.add((min(curr, nxt), max(curr, nxt)))
                prev, curr = curr, nxt
            if closed and len(loop) >= 3:
                loops.append(loop)

    records: List[Dict[str, Any]] = []
    for h_id, l_verts in enumerate(loops):
        pts_l = verts[l_verts]
        cent = pts_l.mean(axis=0)
        d_cent = np.linalg.norm(pts_l - cent, axis=1)
        diam = float(d_cent.max() * 2.0) if len(pts_l) > 0 else 0.0

        centered = pts_l - cent
        cov = (centered.T @ centered) / max(1, len(pts_l))
        _, eigvecs = np.linalg.eigh(cov)
        n_plane = eigvecs[:, 0].copy()
        n_mean_v = vnorms[l_verts].mean(axis=0)
        if np.linalg.norm(n_mean_v) > 1e-6 and float(np.dot(n_plane, n_mean_v)) < 0:
            n_plane = -n_plane

        # Guarantee right-handed orthonormal basis (u_ax x v_ax == n_plane)
        u_ax = eigvecs[:, 2].copy()
        v_ax = np.cross(n_plane, u_ax)
        v_ax /= max(np.linalg.norm(v_ax), 1e-8)

        px = centered @ u_ax
        py = centered @ v_ax
        signed_area = 0.5 * float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
        if signed_area < 0.0:
            l_verts = list(reversed(l_verts))
            pts_l = verts[l_verts]
            centered = pts_l - cent
            px = centered @ u_ax
            py = centered @ v_ax
            signed_area = abs(signed_area)

        area = abs(signed_area)
        perim = float(np.sum(np.linalg.norm(np.roll(pts_l, -1, axis=0) - pts_l, axis=1)))
        eq_diam = 2.0 * math.sqrt(max(area, 1e-9) / math.pi)
        eff_diam = max(diam, eq_diam)

        loop_plane_mad = float(np.median(np.abs(centered @ n_plane)))
        rim_vn = vnorms[l_verts]
        rim_dots = np.clip(np.abs(rim_vn @ n_plane), 0.0, 1.0)
        rim_vnorm_std_deg = float(np.std(np.degrees(np.arccos(rim_dots))))

        records.append({
            "hole_id": h_id,
            "vertex_indices": l_verts,
            "num_vertices": len(l_verts),
            "centroid": cent,
            "normal": n_plane,
            "u_axis": u_ax,
            "v_axis": v_ax,
            "span_diameter_m": round(diam, 4),
            "eq_diameter_m": round(eq_diam, 4),
            "diameter_m": round(eff_diam, 4),
            "area_m2": round(area, 5),
            "perimeter_m": round(perim, 4),
            "diameter_in_voxels": round(eff_diam / max(voxel_size, 1e-4), 2),
            "loop_plane_mad_m": round(loop_plane_mad, 5),
            "rim_vnorm_std_deg": round(rim_vnorm_std_deg, 2),
            "extent_u_m": round(float(px.max() - px.min()), 4) if len(px) else 0.0,
            "extent_v_m": round(float(py.max() - py.min()), 4) if len(py) else 0.0,
        })

    records.sort(key=lambda r: r["area_m2"], reverse=True)
    return records


def run_conditional_localized_repair(
    mesh_observed: o3d.geometry.TriangleMesh,
    geom: CanonicalSceneGeometry,
    meshes_dir: Path,
    cfg: MeshBranchQAConfig,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    F.O.R.G.E. V2.2 Continuous Gap Classification & Per-Patch QA-Gated Localized Repair:
      - NEVER overwrites `mesh_observed`.
      - Eliminates Case A/B dead zone with explicit continuous rules:
          * Category A (`CATEGORY_A_TINY_PINHOLE`): small pinhole (`diam <= small_thresh_m` & `area < min_b`
            & simple vertex count) -> boundary-pinned triangulation sharing original TSDF loop vertices.
          * Category B1 (`CATEGORY_B1_PLANAR_CONSTRAINED`): qualifying planar/near-planar supported enclosed
            gap -> boundary-pinned Constrained 2D Polygon Triangulation + interior grid/harmonic lift.
          * Category B2 (`CATEGORY_B2_CURVED_POISSON`): qualifying curved supported enclosed gap -> strict
            co-normal localized Screened Poisson, cropped strictly to 2D boundary polygon + tight band,
            single largest component, welded to TSDF loop vertices.
          * Category C (`CATEGORY_C_OPEN_OR_UNSUPPORTED`): open doorway / wall rim / >max diam / unsupported /
            known free space -> preserved untouched.
      - EVERY candidate patch (A/B1/B2) undergoes independent Per-Patch Local QA BEFORE committing!
    """
    t0 = time.perf_counter()
    s_vox = float(cfg.tsdf_voxel_length)
    small_thresh_m = max(cfg.small_hole_max_voxel_diam * s_vox, 0.055)

    holes = analyze_tsdf_boundary_holes(mesh_observed, s_vox)
    if not holes:
        return {
            "poisson_triggered": False,
            "localized_poisson_executed": False,
            "b1_planar_triggered": False,
            "case_b_repair_triggered": False,
            "repair_attempted": False,
            "small_holes_repaired": 0,
            "b1_patches_accepted": 0,
            "b2_poisson_patches_accepted": 0,
            "qualifying_gaps_count": 0,
            "rejected_gaps_count": 0,
            "rejected_per_patch_qa_count": 0,
            "total_holes_detected": 0,
            "repaired_mesh": None,
            "case_a_decisions": [],
            "case_b_patches": [],
            "rejected_patches": [],
            "case_c_preserved_gaps": [],
            "gap_classifications": [],
            "accepted_face_start_idx": int(len(mesh_observed.triangles)),
            "biggest_accepted_centroid": None,
            "note": "Global TSDF has 0 closed open boundary loops; repair cleanly bypassed.",
            "duration_sec": round(time.perf_counter() - t0, 3),
        }

    pts_obs = geom.points.astype(np.float64)
    nrm_obs = geom.normals.astype(np.float64)
    col_obs = geom.colors.astype(np.float64)
    obs_tree = cKDTree(pts_obs) if len(pts_obs) > 0 else None

    verts_work = np.asarray(mesh_observed.vertices, dtype=np.float64).copy()
    tris_work = np.asarray(mesh_observed.triangles, dtype=np.int32).copy()
    cols_work = (
        np.asarray(mesh_observed.vertex_colors, dtype=np.float64).copy()
        if mesh_observed.has_vertex_colors()
        else np.full_like(verts_work, 0.75)
    )
    if not mesh_observed.has_vertex_normals():
        mesh_observed.compute_vertex_normals()
    vnorms_obs = np.asarray(mesh_observed.vertex_normals, dtype=np.float64)
    mesh_tree = cKDTree(verts_work) if len(verts_work) > 0 else None

    accepted_face_start_idx = int(len(tris_work))
    case_a_list: List[Dict[str, Any]] = []
    case_b_patches: List[Dict[str, Any]] = []
    rejected_patches: List[Dict[str, Any]] = []
    case_c_list: List[Dict[str, Any]] = []
    gap_classifications: List[Dict[str, Any]] = []

    b1_attempted = 0
    b2_attempted = 0
    b1_accepted = 0
    b2_accepted = 0
    a_attempted = 0
    biggest_accepted_centroid: Optional[np.ndarray] = None
    biggest_accepted_area = -1.0

    for h in holes:
        hid = int(h["hole_id"])
        diam = float(h["diameter_m"])
        eq_diam = float(h.get("eq_diameter_m", diam))
        area = float(h["area_m2"])
        d_vox = float(h.get("diameter_in_voxels", diam / max(s_vox, 1e-4)))
        perim = float(h.get("perimeter_m", 0.0))
        cent = np.asarray(h["centroid"], dtype=np.float64)
        n_loop = np.asarray(h["normal"], dtype=np.float64)
        u_ax = np.asarray(h["u_axis"], dtype=np.float64)
        v_ax = np.asarray(h["v_axis"], dtype=np.float64)
        l_idx = list(h["vertex_indices"])
        pts_l = verts_work[l_idx]
        poly_2d = np.column_stack([(pts_l - cent) @ u_ax, (pts_l - cent) @ v_ax])
        loop_mad = float(h.get("loop_plane_mad_m", 0.0))

        # 1. Sample interior polygon points cleanly for FreeSpace check
        probe_pts_list = [cent]
        for rad_s in (0.35, 0.65):
            ring_p = cent[None, :] + (pts_l - cent) * rad_s
            if len(ring_p) > 12:
                ring_p = ring_p[::max(1, len(ring_p) // 12)]
            probe_pts_list.append(ring_p)
        probe_pts = np.vstack([p.reshape(-1, 3) for p in probe_pts_list])
        if geom.free_space_evidence is not None and len(probe_pts) > 0:
            fse_ratio = float(np.mean(geom.free_space_evidence.query_free_space_mask(probe_pts)))
        else:
            fse_ratio = 0.0

        # 2. Azimuthal perimeter enclosure on TSDF mesh vertices STRICTLY OUTSIDE the gap polygon
        enclosed_sectors = 0
        if mesh_tree is not None and len(pts_l) >= 3:
            r_enc = max(0.14, min(0.55, diam * 0.65 + 0.08))
            m_idxs = mesh_tree.query_ball_point(cent, r=r_enc)
            if m_idxs:
                rel_m = verts_work[m_idxs] - cent[None, :]
                ortho_m = np.abs(rel_m @ n_loop)
                mu, mv = rel_m @ u_ax, rel_m @ v_ax
                muv = np.column_stack([mu, mv])
                # Expand slightly (+max(0.02, s_vox*1.2)) so outer perimeter vertices on the rim itself count as inside/on!
                on_or_in_poly = MplPath(poly_2d).contains_points(muv, radius=max(0.02, s_vox * 1.2))
                outside_poly = (
                    (ortho_m <= max(0.065, s_vox * 4.5))
                    & (~on_or_in_poly)
                )
                if np.count_nonzero(outside_poly) >= 8:
                    ang_s = np.floor(((np.arctan2(mv[outside_poly], mu[outside_poly]) + math.pi) / (2.0 * math.pi)) * 8.0).astype(int) % 8
                    enclosed_sectors = int(len(np.unique(ang_s)))

        # 3. Tight orthogonal & co-normal support evidence (NO wide opposite-wall sweep!)
        sup_rad = max(0.09, min(0.36, diam * 0.50 + 0.07))
        rim_sup_indices = np.zeros(0, dtype=np.int64)
        interior_sup_indices = np.zeros(0, dtype=np.int64)
        ang_std_deg = float(h.get("rim_vnorm_std_deg", 0.0))

        if obs_tree is not None:
            raw_idxs = obs_tree.query_ball_point(cent, r=sup_rad)
            if raw_idxs:
                r_arr = np.asarray(raw_idxs, dtype=np.int64)
                rel_s = pts_obs[r_arr] - cent[None, :]
                ortho_s = np.abs(rel_s @ n_loop)
                dots_s = np.abs(nrm_obs[r_arr] @ n_loop)
                slab_tol = max(0.045, s_vox * 3.5)
                coplanar_mask = (ortho_s <= slab_tol) & (dots_s >= 0.65)
                if np.any(coplanar_mask):
                    rim_sup_indices = r_arr[coplanar_mask]
                    rel_cop = rel_s[coplanar_mask]
                    s_2d = np.column_stack([rel_cop @ u_ax, rel_cop @ v_ax])
                    in_poly = MplPath(poly_2d).contains_points(s_2d)
                    interior_sup_indices = rim_sup_indices[in_poly]
                    n_cop = nrm_obs[rim_sup_indices]
                    mean_n = n_cop.mean(axis=0)
                    if np.linalg.norm(mean_n) > 1e-6:
                        mean_n /= np.linalg.norm(mean_n)
                        ang_std_deg = float(np.std(np.degrees(np.arccos(np.clip(np.abs(n_cop @ mean_n), 0.0, 1.0)))))

        support_count = int(len(rim_sup_indices))
        interior_support_count = int(len(interior_sup_indices))

        def _record_gap(category: str, method: str, status: str, reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            rec = {
                "hole_id": hid,
                "category": category,
                "selected_method": method,
                "decision_status": status,
                "decision_reason": reason,
                "centroid": [round(float(x), 4) for x in cent],
                "num_vertices": len(l_idx),
                "diameter_m": round(diam, 4),
                "eq_diameter_m": round(eq_diam, 4),
                "area_m2": round(area, 5),
                "perimeter_m": round(perim, 4),
                "diameter_in_voxels": round(d_vox, 2),
                "loop_plane_mad_m": round(loop_mad, 5),
                "normal_std_deg": round(ang_std_deg, 2),
                "support_count": support_count,
                "interior_support_count": interior_support_count,
                "enclosed_sectors": enclosed_sectors,
                "free_space_ratio": round(fse_ratio, 4),
            }
            if extra:
                rec.update(extra)
            gap_classifications.append(rec)
            return rec

        # 4. Continuous Gap Classification (ZERO Dead Zone!)
        # 4a. Case C exclusions (Open doors, perimeter rims, oversized openings, free-space openings)
        if fse_ratio > cfg.case_c_max_freespace_ratio:
            rec = _record_gap("CATEGORY_C_FREESPACE_OPENING", "NONE_PRESERVE", "PRESERVED",
                              f"KNOWN_FREE_SPACE rays traverse interior ({fse_ratio:.2f} > {cfg.case_c_max_freespace_ratio:.2f}).")
            case_c_list.append(rec)
            continue

        if diam > cfg.large_hole_max_diam_m:
            rec = _record_gap("CATEGORY_C_OVERSIZED_OPENING", "NONE_PRESERVE", "PRESERVED",
                              f"Diameter {diam:.2f}m exceeds safe ceiling {cfg.large_hole_max_diam_m:.2f}m.")
            case_c_list.append(rec)
            continue

        if enclosed_sectors < cfg.case_b_min_enclosed_sectors:
            rec = _record_gap("CATEGORY_C_OPEN_BOUNDARY_RIM", "NONE_PRESERVE", "PRESERVED",
                              f"Enclosed in {enclosed_sectors}/8 azimuthal sectors (< {cfg.case_b_min_enclosed_sectors}); outer room/object rim.")
            case_c_list.append(rec)
            continue

        # 4b. Case A: Tiny pinhole (< small_thresh_m and compact vertex loop and < case_b_min_area_m2)
        is_tiny_case_a = (diam <= small_thresh_m) and (len(l_idx) <= cfg.case_a_max_loop_verts) and (area < cfg.case_b_min_area_m2)
        if is_tiny_case_a:
            if a_attempted >= cfg.max_case_a_patches:
                rec = _record_gap("CATEGORY_A_TINY_PINHOLE", "NONE_CAPPED", "PRESERVED", "Exceeded max_case_a_patches budget.")
                case_a_list.append(rec)
                continue
            a_attempted += 1
            p_pts, p_tris, p_gids = build_boundary_pinned_planar_patch(
                h, verts_work, pts_obs, interior_sup_indices, s_vox,
                max_height_delta_m=cfg.b1_planar_mad_max_m, allow_interior_subdivision=False,
                existing_tris=tris_work,
            )
            ok_qa, qa_reason, qa_met, committed = verify_patch_local_qa(
                h, verts_work, tris_work, p_pts, p_tris, p_gids,
                geom.free_space_evidence, geom.synchronized_depth_maps,
                geom.extrinsics, geom.intrinsics, pts_obs, col_obs, obs_tree,
                voxel_size=s_vox, min_depth=cfg.min_depth, max_depth=cfg.max_depth,
                max_freespace_ratio=cfg.case_c_max_freespace_ratio,
                max_normal_deg=32.0, max_dbl_shell_delta=cfg.patch_qa_max_dbl_shell_delta,
            )
            if ok_qa:
                verts_work, tris_work, _, added_t, _ = assemble_stitched_patch_into_mesh(
                    verts_work, tris_work, cols_work, p_pts, p_tris, p_gids, pts_obs, col_obs, obs_tree
                )
                cols_work = (
                    np.vstack([cols_work, np.tile(cols_work[l_idx].mean(axis=0), (max(0, len(verts_work) - len(cols_work)), 1))])
                    if len(verts_work) > len(cols_work) else cols_work
                )
                rec = _record_gap("CATEGORY_A_TINY_PINHOLE", "A_PINHOLE_TRIANGULATION", "ACCEPTED",
                                  "Boundary-pinned tiny pinhole fill passed per-patch QA.", {"per_patch_qa": qa_met})
                case_a_list.append(rec)
            else:
                rec = _record_gap("CATEGORY_A_TINY_PINHOLE", "A_PINHOLE_TRIANGULATION", f"REJECTED_PER_PATCH_QA",
                                  qa_reason, {"per_patch_qa": qa_met})
                case_a_list.append(rec)
                rejected_patches.append(rec)
            continue

        # 4c. Case B1 (Planar/Near-Planar) vs Case B2 (Curved Poisson) for ALL remaining enclosed holes
        min_sup_req = min(cfg.case_b_min_support_pts, max(12, int(round(area * 1800.0 + len(l_idx) * 1.2))))
        if support_count < min_sup_req and interior_support_count < 8:
            rec = _record_gap("CATEGORY_C_INSUFFICIENT_SUPPORT", "NONE_PRESERVE", "PRESERVED",
                              f"Support count {support_count} < required {min_sup_req} (interior={interior_support_count}).")
            case_c_list.append(rec)
            continue

        is_planar_b1 = (loop_mad <= cfg.b1_planar_mad_max_m) and (ang_std_deg <= cfg.b1_max_normal_std_deg)
        if not is_planar_b1 and ang_std_deg > cfg.case_b_max_normal_std_deg:
            rec = _record_gap("CATEGORY_C_HIGH_CURVATURE_UNSTABLE", "NONE_PRESERVE", "PRESERVED",
                              f"Curvature/normal spread {ang_std_deg:.1f}deg > {cfg.case_b_max_normal_std_deg:.1f}deg.")
            case_c_list.append(rec)
            continue

        if (b1_accepted + b2_accepted) >= cfg.max_case_b_patches:
            rec = _record_gap(
                "CATEGORY_B1_PLANAR_CONSTRAINED" if is_planar_b1 else "CATEGORY_B2_CURVED_POISSON",
                "NONE_CAPPED", "PRESERVED",
                f"Reached max_case_b_patches limit ({cfg.max_case_b_patches}).",
            )
            case_c_list.append(rec)
            continue

        if is_planar_b1:
            b1_attempted += 1
            cat_name = "CATEGORY_B1_PLANAR_CONSTRAINED"
            method_name = "B1_CONSTRAINED_PLANAR_TRIANGULATION"
            p_pts, p_tris, p_gids = build_boundary_pinned_planar_patch(
                h, verts_work, pts_obs, interior_sup_indices, s_vox,
                max_height_delta_m=cfg.b1_planar_mad_max_m, allow_interior_subdivision=True,
                existing_tris=tris_work,
            )
            max_norm_qa = 26.0
        else:
            b2_attempted += 1
            cat_name = "CATEGORY_B2_CURVED_POISSON"
            method_name = "B2_LOCALIZED_POISSON"
            p_pts, p_tris, p_gids = build_polygon_clipped_stitched_poisson_patch(
                h, verts_work, vnorms_obs, pts_obs, nrm_obs, rim_sup_indices,
                voxel_size=s_vox,
                poisson_depth=int(cfg.poisson_depth),
                density_trim_quantile=float(cfg.density_trim_quantile),
                band_halfwidth_m=float(cfg.b2_band_halfwidth_vox) * s_vox,
                existing_tris=tris_work,
            )
            max_norm_qa = 34.0

        ok_qa, qa_reason, qa_met, _ = verify_patch_local_qa(
            h, verts_work, tris_work, p_pts, p_tris, p_gids,
            geom.free_space_evidence, geom.synchronized_depth_maps,
            geom.extrinsics, geom.intrinsics, pts_obs, col_obs, obs_tree,
            voxel_size=s_vox, min_depth=cfg.min_depth, max_depth=cfg.max_depth,
            max_freespace_ratio=cfg.case_c_max_freespace_ratio,
            max_normal_deg=max_norm_qa, max_dbl_shell_delta=cfg.patch_qa_max_dbl_shell_delta,
        )

        if not ok_qa:
            rec = _record_gap(cat_name, method_name, "REJECTED_PER_PATCH_QA", qa_reason, {"per_patch_qa": qa_met})
            rejected_patches.append(rec)
            if log_callback:
                log_callback(f"    [-] Rejected {method_name} hole #{hid} (D={diam:.2f}m, A={area:.4f}m2): {qa_reason}")
            continue

        # Commit stitched patch into working mesh & export individual verified patch PLY
        verts_work, tris_work, cols_work, added_global_tris, _ = assemble_stitched_patch_into_mesh(
            verts_work, tris_work, cols_work, p_pts, p_tris, p_gids, pts_obs, col_obs, obs_tree
        )
        # Update vertex normals conservatively only where needed
        if len(verts_work) > len(vnorms_obs):
            extra_n = np.tile(n_loop, (len(verts_work) - len(vnorms_obs), 1))
            vnorms_obs = np.vstack([vnorms_obs, extra_n])

        patch_mesh = o3d.geometry.TriangleMesh()
        patch_mesh.vertices = o3d.utility.Vector3dVector(p_pts)
        patch_mesh.triangles = o3d.utility.Vector3iVector(p_tris)
        if obs_tree is not None and len(p_pts) > 0:
            _, c_idx_p = obs_tree.query(p_pts, k=1)
            patch_mesh.vertex_colors = o3d.utility.Vector3dVector(col_obs[c_idx_p] / 255.0)
        patch_path = meshes_dir / f"poisson_repair_patch_{hid}.ply"
        o3d.io.write_triangle_mesh(str(patch_path), patch_mesh, write_vertex_colors=True)

        if is_planar_b1:
            b1_accepted += 1
        else:
            b2_accepted += 1
        if area > biggest_accepted_area:
            biggest_accepted_area = area
            biggest_accepted_centroid = cent.copy()

        patch_rec = _record_gap(
            cat_name, method_name, "ACCEPTED", "Passed all per-patch stitching & physical QA checks.",
            {
                "classification": f"CASE_{'B1_PLANAR_CONSTRAINED' if is_planar_b1 else 'B2_LOCALIZED_POISSON'}_PATCH",
                "repair_method": method_name,
                "patch_triangles": int(len(added_global_tris)),
                "saved_patch_ply": str(patch_path.resolve()),
                "per_patch_qa": qa_met,
            },
        )
        case_b_patches.append(patch_rec)
        if log_callback:
            log_callback(
                f"    [+] Accepted {method_name} hole #{hid}: D={diam:.2f}m, A={area:.4f}m2, "
                f"open_delta={qa_met.get('open_edges_delta', 0)}, dbl_delta={qa_met.get('local_dbl_delta', 0):.3f} -> {len(added_global_tris)} tris"
            )

    repaired_mesh: Optional[o3d.geometry.TriangleMesh] = None
    if len(tris_work) > accepted_face_start_idx:
        repaired_mesh = o3d.geometry.TriangleMesh()
        repaired_mesh.vertices = o3d.utility.Vector3dVector(verts_work)
        repaired_mesh.vertex_colors = o3d.utility.Vector3dVector(cols_work)
        repaired_mesh.triangles = o3d.utility.Vector3iVector(tris_work)
        repaired_mesh.compute_vertex_normals()
        repaired_mesh.compute_triangle_normals()

    note = (
        f"Gap Repair: {b1_accepted}/{b1_attempted} B1 Planar, {b2_accepted}/{b2_attempted} B2 Poisson, "
        f"{sum(1 for a in case_a_list if a.get('decision_status') == 'ACCEPTED')} Case A accepted; "
        f"{len(case_c_list)} Case C preserved; {len(rejected_patches)} patches rejected by per-patch QA."
    )
    if log_callback:
        log_callback(f"  [*] Conditional Repair: {note}")

    return {
        "poisson_triggered": bool(b2_attempted > 0),
        "localized_poisson_executed": bool(b2_attempted > 0),
        "b1_planar_triggered": bool(b1_attempted > 0),
        "case_b_repair_triggered": bool((b1_attempted + b2_attempted) > 0),
        "repair_attempted": bool((a_attempted + b1_attempted + b2_attempted) > 0),
        "small_holes_repaired": sum(1 for a in case_a_list if a.get("decision_status") == "ACCEPTED"),
        "b1_patches_accepted": b1_accepted,
        "b2_poisson_patches_accepted": b2_accepted,
        "qualifying_gaps_count": len(case_b_patches),
        "rejected_gaps_count": len(case_c_list),
        "rejected_per_patch_qa_count": len(rejected_patches),
        "total_holes_detected": len(holes),
        "repaired_mesh": repaired_mesh,
        "case_a_decisions": case_a_list,
        "case_b_patches": case_b_patches,
        "rejected_patches": rejected_patches,
        "case_c_preserved_gaps": case_c_list,
        "gap_classifications": gap_classifications,
        "accepted_face_start_idx": accepted_face_start_idx,
        "biggest_accepted_centroid": biggest_accepted_centroid,
        "note": note,
        "duration_sec": round(time.perf_counter() - t0, 3),
    }


def _compute_mesh_geometric_details(
    mesh: o3d.geometry.TriangleMesh,
    rc_scene: Optional[o3d.t.geometry.RaycastingScene],
    geom: CanonicalSceneGeometry,
    cfg: MeshBranchQAConfig,
) -> Dict[str, Any]:
    """Computes double-shell ratio, local normal deviation, wall/floor MAD flatness, holes & thin preservation."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    fn = np.asarray(mesh.triangle_normals, dtype=np.float64)
    fc = verts[tris].mean(axis=1)

    # 1. Local neighbor normal deviation across <3*voxel neighbors
    step_n = max(1, len(fc) // 8000)
    tree_fc = cKDTree(fc)
    nn_d, nn_idx = tree_fc.query(fc[::step_n], k=min(10, len(fc)))
    if nn_d.ndim == 2 and nn_d.shape[1] > 1:
        dots = np.abs(np.sum(fn[::step_n, None, :] * fn[nn_idx], axis=-1))[:, 1:]
        angs = np.degrees(np.arccos(np.clip(dots, 0.0, 1.0)))
        close_m = nn_d[:, 1:] < max(0.03, cfg.tsdf_voxel_length * 2.5)
        loc_norm_med = float(np.median(angs[close_m])) if np.any(close_m) else 0.0
        noisy_nbr_frac = float(np.mean(angs[close_m] > 25.0)) if np.any(close_m) else 0.0
    else:
        loc_norm_med, noisy_nbr_frac = 0.0, 0.0

    # 2. Horizontal wall double-shell ratio via RaycastingScene multi-hit intersections (1.5-12cm shells)
    double_shell_ratio = 0.0
    if rc_scene is not None and len(geom.cam_centers) > 0:
        cams_np = np.asarray(geom.cam_centers, dtype=np.float64)
        mean_c = cams_np.mean(axis=0)
        ys = np.linspace(float(cams_np[:, 1].min()) - 0.10, float(cams_np[:, 1].max()) + 0.15, 4)
        angs = np.linspace(0.0, 2.0 * math.pi, 90, endpoint=False)
        origins, dirs = [], []
        for yv in ys:
            for av in angs:
                origins.append([mean_c[0], yv, mean_c[2]])
                dirs.append([math.cos(av), 0.0, math.sin(av)])
        rays = o3d.core.Tensor(
            np.hstack([np.array(origins, dtype=np.float32), np.array(dirs, dtype=np.float32)]),
            dtype=o3d.core.Dtype.Float32,
        )
        hits = rc_scene.list_intersections(rays)
        splits = hits["ray_splits"].numpy()
        t_hits = hits["t_hit"].numpy()
        dbl_cnt = 0
        for ri in range(len(origins)):
            s, e = int(splits[ri]), int(splits[ri + 1])
            if e - s >= 2:
                ts = np.sort(t_hits[s:e])
                gaps = np.diff(ts)
                if np.any((gaps > 0.015) & (gaps < 0.12)):
                    dbl_cnt += 1
        double_shell_ratio = float(dbl_cnt / max(1, len(origins)))

    # 3. Wall & floor flatness MAD directly on fitted dominant mesh surfaces
    wall_mad_mm, floor_mad_mm = None, None
    samp_v = verts[::max(1, len(verts) // 12000)]
    if len(samp_v) >= 200:
        pcd_m = o3d.geometry.PointCloud()
        pcd_m.points = o3d.utility.Vector3dVector(samp_v)
        o3d.utility.random.seed(42)
        rest = pcd_m
        wall_mads, floor_mads = [], []
        for _ in range(5):
            if len(rest.points) < 120:
                break
            eq, inl = rest.segment_plane(0.025, 3, 600)
            if len(inl) < 80:
                break
            pts_in = np.asarray(rest.points)[inl]
            n_pl = np.array(eq[:3], dtype=np.float64)
            n_pl /= max(np.linalg.norm(n_pl), 1e-8)
            mad = float(np.median(np.abs(pts_in @ n_pl + eq[3])) * 1000.0)
            if abs(n_pl[1]) >= 0.88:
                # Floor lies near lowest elevation in OpenCV Y-down coords
                if float(np.median(pts_in[:, 1])) > float(np.median(samp_v[:, 1])):
                    floor_mads.append(mad)
            elif abs(n_pl[1]) <= 0.35:
                wall_mads.append(mad)
            rest = rest.select_by_index(inl, invert=True)
        if wall_mads:
            wall_mad_mm = round(float(np.median(wall_mads)), 2)
        if floor_mads:
            floor_mad_mm = round(float(np.median(floor_mads)), 2)

    # 4. Holes count & area
    holes = analyze_tsdf_boundary_holes(mesh, cfg.tsdf_voxel_length)
    hole_cnt = len(holes)
    hole_area = round(float(sum(h["area_m2"] for h in holes)), 5)

    # 5. Thin structure / furniture boundary preservation
    thin_pres = 1.0
    if geom.thin_structure_points is not None and len(geom.thin_structure_points) >= 10:
        d_thin, _ = cKDTree(verts).query(geom.thin_structure_points.astype(np.float64), k=1)
        thin_pres = float(np.mean(d_thin <= max(cfg.tsdf_voxel_length * 3.0, 0.036)))

    return {
        "local_normal_median_deg": round(loc_norm_med, 2),
        "noisy_normal_nbr_fraction": round(noisy_nbr_frac, 4),
        "double_shell_ratio": round(double_shell_ratio, 4),
        "wall_flatness_mad_mm": wall_mad_mm,
        "floor_flatness_mad_mm": floor_mad_mm,
        "hole_count": hole_cnt,
        "hole_area_m2": hole_area,
        "thin_structure_preservation_ratio": round(thin_pres, 4),
    }


def evaluate_mesh_candidate_qa(
    candidate_name: str,
    method: str,
    mesh: o3d.geometry.TriangleMesh,
    geom: CanonicalSceneGeometry,
    cfg: MeshBranchQAConfig,
    build_time_sec: float = 0.0,
    eval_cam_indices: Optional[List[int]] = None,
) -> MeshCandidateQAReport:
    """Evaluates geometry strictly against observation support, free-space, double shells, raycast depth & topology."""
    t0 = time.perf_counter()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int32)
    n_v, n_t = len(verts), len(tris)

    if n_v < 20 or n_t < 20:
        return MeshCandidateQAReport(
            candidate_name=candidate_name, method=method,
            vertex_count=n_v, triangle_count=n_t, is_watertight=False,
            open_boundary_edges=0, non_manifold_edges=0, non_manifold_vertices=0,
            connected_components=0, fragmentation_ratio=1.0,
            hole_count=0, hole_area_m2=0.0,
            obs_support_ratio=0.0, obs_dist_median_m=99.0, unsupported_surface_ratio=1.0,
            surface_completeness_ratio=0.0, free_space_violation_ratio=0.0,
            bridging_severity_score=100.0, duplicate_wall_ratio=0.0,
            double_shell_ratio=0.0, local_normal_median_deg=0.0, noisy_normal_nbr_fraction=0.0,
            wall_flatness_mad_mm=None, floor_flatness_mad_mm=None,
            thin_structure_preservation_ratio=0.0,
            rendered_depth_median_rel_err=1.0, rendered_depth_inlier_ratio=0.0,
            composite_quality_score=0.0, passes_qa=False,
            failure_reasons=["empty_or_degenerate_mesh"],
            build_time_sec=build_time_sec, qa_time_sec=round(time.perf_counter() - t0, 3),
        )

    is_water = bool(mesh.is_watertight())
    non_man_e = len(mesh.get_non_manifold_edges())
    non_man_v = len(mesh.get_non_manifold_vertices())

    e0 = np.sort(tris[:, [0, 1]], axis=1)
    e1 = np.sort(tris[:, [1, 2]], axis=1)
    e2 = np.sort(tris[:, [2, 0]], axis=1)
    all_e = np.vstack([e0, e1, e2])
    _, e_counts = np.unique(all_e, axis=0, return_counts=True)
    open_bound_e = int(np.count_nonzero(e_counts == 1))

    clusters, c_counts, _ = mesh.cluster_connected_triangles()
    c_counts_np = np.asarray(c_counts)
    n_comp = len(c_counts_np)
    frag_ratio = float(1.0 - (c_counts_np.max() / max(1, n_t))) if n_comp > 0 else 1.0

    pts_obs = geom.points.astype(np.float64)
    tree_obs = cKDTree(pts_obs)
    tree_verts = cKDTree(verts)

    samp_v = verts[::max(1, n_v // 8000)]
    d_v2obs, _ = tree_obs.query(samp_v, k=1)
    obs_thresh = cfg.voxel_size * 3.5
    unsup_thresh = cfg.voxel_size * 4.5
    obs_support = float(np.mean(d_v2obs <= obs_thresh))
    obs_med_d = float(np.median(d_v2obs))
    unsup_ratio = float(np.mean(d_v2obs > unsup_thresh))

    samp_obs = pts_obs[::max(1, len(pts_obs) // 8000)]
    d_obs2v, _ = tree_verts.query(samp_obs, k=1)
    completeness = float(np.mean(d_obs2v <= obs_thresh))

    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    max_tri_edge = np.maximum.reduce([
        np.linalg.norm(p1 - p0, axis=1),
        np.linalg.norm(p2 - p1, axis=1),
        np.linalg.norm(p0 - p2, axis=1),
    ])
    med_edge = float(np.median(max_tri_edge))
    centroids = (p0 + p1 + p2) / 3.0
    step_c = max(1, n_t // 4000)
    sc = centroids[::step_c]
    se = max_tri_edge[::step_c]
    d_c2obs, _ = tree_obs.query(sc, k=1)
    bridging_hits = (se > max(med_edge * 3.0, cfg.voxel_size * 4.0)) & (d_c2obs > obs_thresh)
    bridging_sev = float(np.clip(np.mean(bridging_hits) * 150.0, 0.0, 100.0))

    if geom.free_space_evidence is not None:
        fs_viol = float(np.mean(geom.free_space_evidence.query_free_space_mask(sc)))
    else:
        fs_viol = 0.0

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    tnorms = np.asarray(mesh.triangle_normals, dtype=np.float64)[::step_c]
    vert_mask = np.abs(tnorms[:, 1]) <= 0.35
    dup_wall_ratio = 0.0
    if np.count_nonzero(vert_mask) >= 40:
        vc = sc[vert_mask]
        vn = tnorms[vert_mask]
        vtree = cKDTree(vc)
        _, nn_idxs = vtree.query(vc, k=min(8, len(vc)))
        dup_hits = 0
        for idx_k in range(len(vc)):
            n0 = vn[idx_k]
            nbrs = nn_idxs[idx_k][1:]
            deltas = vc[nbrs] - vc[idx_k]
            norm_sep = np.abs(deltas @ n0)
            par_align = np.abs(vn[nbrs] @ n0)
            if np.any((norm_sep >= 0.03) & (norm_sep <= 0.12) & (par_align >= 0.90)):
                dup_hits += 1
        dup_wall_ratio = float(dup_hits / len(vc))

    # Build BVH RaycastingScene once for both rendered depth QA & multi-hit double-shell detection
    N_cams = len(geom.extrinsics)
    d_maps = geom.synchronized_depth_maps[..., 0] if geom.synchronized_depth_maps.ndim == 4 else geom.synchronized_depth_maps
    _, H_d, W_d = d_maps.shape
    cam_idxs = eval_cam_indices if eval_cam_indices is not None else np.linspace(0, N_cams - 1, min(10, N_cams), dtype=int).tolist()

    rc_scene = None
    rc_errs = []
    if n_t > 0:
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        rc_scene = o3d.t.geometry.RaycastingScene()
        rc_scene.add_triangles(tmesh)

    if rc_scene is not None and N_cams > 0:
        ys = np.linspace(int(H_d * 0.12), int(H_d * 0.88), 24, dtype=int)
        xs = np.linspace(int(W_d * 0.12), int(W_d * 0.88), 24, dtype=int)
        ug, vg = np.meshgrid(xs, ys)
        uf, vf = ug.ravel().astype(np.float64), vg.ravel().astype(np.float64)

        for ci in cam_idxs:
            obs_d = d_maps[ci, vg, ug].ravel()
            val = (obs_d >= cfg.min_depth) & (obs_d <= cfg.max_depth)
            if np.count_nonzero(val) < 15:
                continue
            K = geom.intrinsics[ci]
            R = geom.extrinsics[ci, :3, :3]
            t = geom.extrinsics[ci, :3, 3]
            cam_c = (-R.T @ t).astype(np.float32)

            dirs_c = np.column_stack([
                (uf[val] - K[0, 2]) / K[0, 0],
                (vf[val] - K[1, 2]) / K[1, 1],
                np.ones(np.count_nonzero(val)),
            ])
            dirs_w = (dirs_c @ R).astype(np.float32)
            rays_tensor = o3d.core.Tensor(
                np.hstack([np.tile(cam_c, (len(dirs_w), 1)), dirs_w]), dtype=o3d.core.Dtype.Float32
            )
            ans = rc_scene.cast_rays(rays_tensor)
            t_hit = ans["t_hit"].numpy()
            hit_ok = np.isfinite(t_hit) & (t_hit > 0.05) & (t_hit < 20.0)
            if np.any(hit_ok):
                z_render = t_hit[hit_ok]
                rel_e = np.abs(z_render - obs_d[val][hit_ok]) / np.maximum(obs_d[val][hit_ok], 1e-3)
                rc_errs.extend(rel_e.tolist())

    rc_med_err = float(np.median(rc_errs)) if rc_errs else 1.0
    rc_inlier = float(np.mean(np.asarray(rc_errs) <= 0.08)) if rc_errs else 0.0

    geom_details = _compute_mesh_geometric_details(mesh, rc_scene, geom, cfg)

    failures = []
    if n_t < cfg.qa_min_triangles:
        failures.append(f"insufficient_triangles ({n_t} < {cfg.qa_min_triangles})")
    if obs_support < cfg.qa_min_obs_support_ratio:
        failures.append(f"low_observation_support ({obs_support:.3f} < {cfg.qa_min_obs_support_ratio})")
    if completeness < cfg.qa_min_completeness_ratio:
        failures.append(f"low_completeness ({completeness:.3f} < {cfg.qa_min_completeness_ratio})")
    if fs_viol > cfg.qa_max_freespace_violation_ratio:
        failures.append(f"excessive_freespace_violations ({fs_viol:.3f} > {cfg.qa_max_freespace_violation_ratio})")
    if bridging_sev > cfg.qa_max_bridging_severity:
        failures.append(f"excessive_bridging ({bridging_sev:.1f} > {cfg.qa_max_bridging_severity})")
    if rc_med_err > cfg.qa_max_rendered_depth_rel_err:
        failures.append(f"rendered_depth_mismatch ({rc_med_err:.3f} > {cfg.qa_max_rendered_depth_rel_err})")
    if geom_details["double_shell_ratio"] > cfg.qa_max_double_shell_ratio:
        failures.append(f"excessive_double_shells ({geom_details['double_shell_ratio']:.3f} > {cfg.qa_max_double_shell_ratio})")
    if (non_man_e / max(1, n_t)) > cfg.qa_max_non_manifold_edge_ratio:
        failures.append(f"non_manifold_topology ({non_man_e} edges)")

    comp_score = (
        0.26 * completeness
        + 0.22 * obs_support
        + 0.20 * max(0.0, 1.0 - fs_viol * 4.0)
        + 0.16 * max(0.0, 1.0 - rc_med_err * 4.0)
        + 0.08 * max(0.0, 1.0 - geom_details["double_shell_ratio"] * 2.0)
        + 0.08 * max(0.0, 1.0 - bridging_sev / 100.0)
    )

    return MeshCandidateQAReport(
        candidate_name=candidate_name,
        method=method,
        vertex_count=n_v,
        triangle_count=n_t,
        is_watertight=is_water,
        open_boundary_edges=open_bound_e,
        non_manifold_edges=non_man_e,
        non_manifold_vertices=non_man_v,
        connected_components=n_comp,
        fragmentation_ratio=round(frag_ratio, 4),
        hole_count=geom_details["hole_count"],
        hole_area_m2=geom_details["hole_area_m2"],
        obs_support_ratio=round(obs_support, 4),
        obs_dist_median_m=round(obs_med_d, 5),
        unsupported_surface_ratio=round(unsup_ratio, 4),
        surface_completeness_ratio=round(completeness, 4),
        free_space_violation_ratio=round(fs_viol, 4),
        bridging_severity_score=round(bridging_sev, 2),
        duplicate_wall_ratio=round(dup_wall_ratio, 4),
        double_shell_ratio=geom_details["double_shell_ratio"],
        local_normal_median_deg=geom_details["local_normal_median_deg"],
        noisy_normal_nbr_fraction=geom_details["noisy_normal_nbr_fraction"],
        wall_flatness_mad_mm=geom_details["wall_flatness_mad_mm"],
        floor_flatness_mad_mm=geom_details["floor_flatness_mad_mm"],
        thin_structure_preservation_ratio=geom_details["thin_structure_preservation_ratio"],
        rendered_depth_median_rel_err=round(rc_med_err, 4),
        rendered_depth_inlier_ratio=round(rc_inlier, 4),
        composite_quality_score=round(float(comp_score), 4),
        passes_qa=bool(len(failures) == 0),
        failure_reasons=failures,
        build_time_sec=build_time_sec,
        qa_time_sec=round(time.perf_counter() - t0, 3),
    )


def run_global_tsdf_and_conditional_repair(
    geom: CanonicalSceneGeometry,
    frames_bgr: List[np.ndarray],
    meshes_dir: Path,
    cfg: Optional[MeshBranchQAConfig] = None,
    eval_cam_indices: Optional[List[int]] = None,
    renders_dir: Optional[Path] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs the single mandatory Global Scalable TSDF branch, preserves unmodified observed geometry,
    classifies ALL boundary gaps continuously (A/B1/B2/C), applies per-patch QA-gated stitching
    for Case A/B1/B2 patches, runs Global Mesh QA + no-degrade checks, and generates gap
    diagnostic + before/after comparison renders. Never forces a failing candidate to PASS.
    """
    def _log(msg: str):
        if log_callback:
            log_callback(msg)

    c = cfg or MeshBranchQAConfig(voxel_size=geom.voxel_size, tsdf_voxel_length=geom.voxel_size)
    meshes_dir.mkdir(parents=True, exist_ok=True)
    out_renders_dir = renders_dir if renders_dir is not None else (meshes_dir.parent / "renders")
    out_renders_dir.mkdir(parents=True, exist_ok=True)

    # 1. Primary Observed-Only Global Scalable TSDF (always preserved!)
    mesh_obs, tsdf_stats = build_global_tsdf_branch(geom, frames_bgr, c, log_callback=_log)
    obs_ply = meshes_dir / "mesh_global_tsdf_observed.ply"
    o3d.io.write_triangle_mesh(str(obs_ply), mesh_obs, write_vertex_colors=True)

    qa_obs = evaluate_mesh_candidate_qa(
        "global_tsdf_observed", "global_scalable_tsdf", mesh_obs, geom, c, tsdf_stats["duration_sec"], eval_cam_indices
    )
    qa_obs.saved_ply_path = str(obs_ply.resolve())
    _log(
        f"    Global TSDF Observed QA: pass={qa_obs.passes_qa} | V={qa_obs.vertex_count:,} F={qa_obs.triangle_count:,} | "
        f"dbl_shell={qa_obs.double_shell_ratio:.3f} norm_med={qa_obs.local_normal_median_deg:.1f}deg "
        f"holes={qa_obs.hole_count} ({qa_obs.hole_area_m2:.3f}m2) sup={qa_obs.obs_support_ratio:.3f} "
        f"comp={qa_obs.surface_completeness_ratio:.3f} fse_viol={qa_obs.free_space_violation_ratio:.3f} "
        f"lost3v={tsdf_stats.get('canonical_points_lost_pct', 0):.1f}% | score={qa_obs.composite_quality_score:.4f} "
        f"fails={qa_obs.failure_reasons}"
    )

    # 2. Continuous Gap Classification & Per-Patch QA-Gated Localized Repair
    repair_info: Dict[str, Any] = {
        "poisson_triggered": False,
        "localized_poisson_executed": False,
        "b1_planar_triggered": False,
        "case_b_repair_triggered": False,
        "repair_attempted": False,
        "small_holes_repaired": 0,
        "b1_patches_accepted": 0,
        "b2_poisson_patches_accepted": 0,
        "qualifying_gaps_count": 0,
        "rejected_gaps_count": 0,
        "rejected_per_patch_qa_count": 0,
        "total_holes_detected": qa_obs.hole_count,
        "observed_triangle_count": int(len(mesh_obs.triangles)),
        "accepted_patch_triangle_count": 0,
        "accepted_face_start_idx": int(len(mesh_obs.triangles)),
        "case_a_decisions": [],
        "case_b_patches": [],
        "rejected_patches": [],
        "case_c_preserved_gaps": [],
        "note": "Conditional repair disabled.",
        "repair_acceptance_decision": "BYPASSED",
        "visual_artifacts": {},
    }
    qa_rep: Optional[MeshCandidateQAReport] = None
    mesh_rep: Optional[o3d.geometry.TriangleMesh] = None
    gap_classifications: List[Dict[str, Any]] = []
    biggest_cent: Optional[np.ndarray] = None

    if c.enable_conditional_repair and len(mesh_obs.triangles) >= 20:
        repair_res = run_conditional_localized_repair(mesh_obs, geom, meshes_dir, c, log_callback=_log)
        mesh_rep = repair_res.pop("repaired_mesh", None)
        gap_classifications = repair_res.pop("gap_classifications", [])
        biggest_cent = repair_res.pop("biggest_accepted_centroid", None)
        repair_info.update(repair_res)

        if mesh_rep is not None and len(mesh_rep.triangles) >= 20:
            rep_ply = meshes_dir / "mesh_global_tsdf_repaired.ply"
            o3d.io.write_triangle_mesh(str(rep_ply), mesh_rep, write_vertex_colors=True)
            qa_rep = evaluate_mesh_candidate_qa(
                "global_tsdf_repaired", "global_tsdf_with_conditional_repair", mesh_rep, geom, c, repair_info["duration_sec"], eval_cam_indices
            )
            qa_rep.saved_ply_path = str(rep_ply.resolve())

            # Strict Global non-degradation QA check before accepting repaired copy!
            no_degrade = (
                qa_rep.free_space_violation_ratio <= qa_obs.free_space_violation_ratio + 0.005
                and qa_rep.unsupported_surface_ratio <= qa_obs.unsupported_surface_ratio + 0.015
                and qa_rep.double_shell_ratio <= qa_obs.double_shell_ratio + 0.015
                and qa_rep.rendered_depth_median_rel_err <= qa_obs.rendered_depth_median_rel_err + 0.010
                and qa_rep.non_manifold_edges <= qa_obs.non_manifold_edges + 5
            )
            if qa_rep.passes_qa and no_degrade:
                repair_info["repair_acceptance_decision"] = "ACCEPTED"
            elif (not qa_obs.passes_qa) and qa_rep.passes_qa:
                repair_info["repair_acceptance_decision"] = "ACCEPTED"
            else:
                repair_info["repair_acceptance_decision"] = (
                    f"REJECTED (fse={qa_rep.free_space_violation_ratio:.3f} vs {qa_obs.free_space_violation_ratio:.3f}, "
                    f"unsup={qa_rep.unsupported_surface_ratio:.3f} vs {qa_obs.unsupported_surface_ratio:.3f}, "
                    f"dbl={qa_rep.double_shell_ratio:.3f} vs {qa_obs.double_shell_ratio:.3f}, "
                    f"nm={qa_rep.non_manifold_edges} vs {qa_obs.non_manifold_edges})"
                )
            repair_info["accepted_patch_triangle_count"] = int(max(0, len(mesh_rep.triangles) - len(mesh_obs.triangles)))
            _log(
                f"    Global TSDF Repaired QA: decision={repair_info['repair_acceptance_decision']} | "
                f"V={qa_rep.vertex_count:,} F={qa_rep.triangle_count:,} (+{repair_info['accepted_patch_triangle_count']:,}) | "
                f"open={qa_obs.open_boundary_edges}->{qa_rep.open_boundary_edges} | "
                f"holes={qa_obs.hole_count}->{qa_rep.hole_count} ({qa_obs.hole_area_m2:.3f}->{qa_rep.hole_area_m2:.3f}m2) | "
                f"dbl={qa_obs.double_shell_ratio:.4f}->{qa_rep.double_shell_ratio:.4f}"
            )
        else:
            repair_info["repair_acceptance_decision"] = "BYPASSED_NO_ACCEPTED_PATCHES"

    # Persist gap classifications manifest + diagnostic visuals safely
    gap_json_path = meshes_dir / "gap_classifications.json"
    gap_json_path.write_text(json.dumps({
        "total_holes_detected": len(gap_classifications),
        "accepted_count": sum(1 for g in gap_classifications if g.get("decision_status") == "ACCEPTED"),
        "preserved_case_c_count": sum(1 for g in gap_classifications if g.get("decision_status") == "PRESERVED"),
        "rejected_per_patch_qa_count": sum(1 for g in gap_classifications if str(g.get("decision_status", "")).startswith("REJECTED")),
        "gaps": gap_classifications,
    }, indent=2), encoding="utf-8")
    vis_arts: Dict[str, Optional[str]] = {"gap_classifications_json": str(gap_json_path.resolve())}
    try:
        diag_png = render_gap_diagnostic_views(mesh_obs, gap_classifications, out_renders_dir / "gap_diagnostics.png")
        vis_arts["gap_diagnostics_png"] = diag_png
    except Exception:
        vis_arts["gap_diagnostics_png"] = None
    try:
        comp_png = render_before_after_repair_comparison(
            mesh_obs, mesh_rep, int(len(mesh_obs.triangles)), biggest_cent,
            out_renders_dir / "before_after_repair_comparison.png",
        )
        vis_arts["before_after_comparison_png"] = comp_png
    except Exception:
        vis_arts["before_after_comparison_png"] = None
    repair_info["visual_artifacts"] = vis_arts

    # 3. Select authoritative final observed/repaired mesh (NO forced PASS!)
    if repair_info["repair_acceptance_decision"] == "ACCEPTED" and mesh_rep is not None and qa_rep is not None:
        selected_mesh = mesh_rep
        selected_report = qa_rep
    else:
        selected_mesh = mesh_obs
        selected_report = qa_obs

    if selected_report.passes_qa:
        selection_status = "PASS"
        rationale = f"{selected_report.candidate_name} ({selected_report.method}) passed all global TSDF & physical QA gates."
    elif (
        selected_report.triangle_count >= 100
        and selected_report.obs_support_ratio >= 0.65
        and selected_report.surface_completeness_ratio >= 0.40
        and selected_report.free_space_violation_ratio <= 0.18
    ):
        selection_status = "PARTIAL_MODEL"
        rationale = (
            f"Global TSDF partial recovery ({selected_report.failure_reasons}); "
            "diagnostic mesh retained without forcing PASS."
        )
    else:
        selection_status = "QA_FAILED"
        rationale = f"Global TSDF failed Mesh QA gates ({selected_report.failure_reasons}); rejecting model."

    summary = {
        "selection_status": selection_status,
        "selected_candidate": selected_report.candidate_name,
        "selected_method": selected_report.method,
        "selection_rationale": rationale,
        "tsdf_integration_stats": tsdf_stats,
        "conditional_poisson_repair": repair_info,
        "candidates": {
            "global_tsdf_observed": qa_obs.to_dict(),
            "global_tsdf_repaired": qa_rep.to_dict() if qa_rep is not None else None,
        },
    }
    return {
        "selected_mesh": selected_mesh,
        "selected_qa": selected_report,
        "selection_status": selection_status,
        "mesh_observed": mesh_obs,
        "mesh_repaired": mesh_rep,
        "qa_observed": qa_obs,
        "qa_repaired": qa_rep,
        "repair_info": repair_info,
        "tsdf_stats": tsdf_stats,
        "summary": summary,
    }
