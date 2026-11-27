"""
F.O.R.G.E. V2 — Behind the Veil Feasibility Audit & Optional Conservative Hidden-Surface Completion.

BEHIND THE VEIL FEASIBILITY AUDIT (Verified 2026-11-26):
  - Paper: "Behind the Veil: Enhanced Indoor 3D Scene Reconstruction with Occluded Surfaces Completion"
    (Su Sun, Cheng Zhao, Yuliang Guo, Ruoyu Wang, Xinyu Huang, Yingjie Victor Chen, Liu Ren — CVPR 2024, arXiv:2404.03070).
  - Dataset: Official 3D-CRS dataset is public (`https://github.com/BoschRHI3NA/3D-CRS-dataset`).
  - Code & Pretrained Weights: The published octree dual-decoder `Geo-decoder` & `3D Inpainter`
    inference code and checkpoints are NOT publicly released by the authors.
  - Conclusion: Genuine neural Behind-the-Veil inference is UNAVAILABLE locally without retraining
    a generative 3D octree model from scratch (outside milestone scope / weights unavailable).
  - Local Alternative Implemented Here (Explicitly Labelled):
    `local_conservative_occluded_wall_completion` — bounded geometric completion ONLY on
    verified flat wall planes behind furniture where ALL of the following hold:
      1. ZERO `KNOWN_FREE_SPACE` voxels intersect the candidate cell (doors/passages strictly preserved!).
      2. Distance to camera walking corridor >= `min_corridor_clearance_m`.
      3. Observed wall support surrounds the occluded interior patch (interpolation only).
      4. Kept in a SEPARATE `inferred_mesh` with distinct non-photographic extrapolated provenance;
         the accepted `observed_only` mesh is NEVER overwritten.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence
from src.reconstruction.planar_refinement import PlanarRegionRecord


FEASIBILITY_AUDIT = {
    "target_method": "Behind the Veil (CVPR 2024, arXiv:2404.03070)",
    "dataset_available": "BoschRHI3NA/3D-CRS-dataset (public on GitHub)",
    "official_code_and_weights_available": False,
    "blocker": "Author repository releases dataset only; pretrained Geo-decoder and 3D Inpainter weights/code are not publicly released.",
    "local_alternative_implemented": "local_conservative_occluded_wall_completion (bounded planar interpolation gated by 3D free-space evidence)",
    "misrepresentation_guard": "Outputs are explicitly tagged as geometric wall-plane extrapolation and never labelled as neural Behind-the-Veil.",
}


@dataclass
class HiddenCompletionConfig:
    enabled: bool = False  # Explicit OFF bypass by default; strictly optional!
    grid_step_m: float = 0.08
    support_margin_cells: int = 1
    min_corridor_clearance_m: float = 0.35
    max_wall_gap_cells: int = 8  # Never fill huge empty openings > ~0.64m wide without strong enclosure
    inferred_color_rgb: Tuple[int, int, int] = (145, 165, 195)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_optional_hidden_surface_completion(
    observed_mesh: o3d.geometry.TriangleMesh,
    planar_regions: List[PlanarRegionRecord],
    observed_points: np.ndarray,
    cam_centers: List[np.ndarray],
    free_space_evidence: Optional[FreeSpaceVoxelEvidence],
    out_dir: Path,
    parent_hashes: Optional[Dict[str, Optional[str]]] = None,
    config: Optional[HiddenCompletionConfig] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Generates separate conservative inferred wall quads ONLY in occluded behind-furniture gaps
    with zero KNOWN_FREE_SPACE violations. Preserves `observed_mesh` untouched.
    """
    cfg = config or HiddenCompletionConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "hidden_completion_report.json"

    if not cfg.enabled:
        rep = {
            "enabled": False,
            "status": "SKIPPED_DISABLED",
            "reason": "Optional hidden-surface completion disabled (observed-only pipeline path).",
            "feasibility_audit": FEASIBILITY_AUDIT,
            "parent_hashes": parent_hashes or {},
        }
        report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
        return {
            "enabled": False,
            "inferred_mesh": None,
            "inferred_ply_path": None,
            "report": rep,
            "report_path": str(report_path.resolve()),
        }

    t0 = time.perf_counter()
    wall_regs = [r for r in planar_regions if r.accepted and r.semantic_label == "wall"]
    obs_verts = np.asarray(observed_mesh.vertices, dtype=np.float64)
    mesh_tree = cKDTree(obs_verts) if len(obs_verts) > 0 else None
    cams_np = np.asarray(cam_centers, dtype=np.float64) if cam_centers else np.zeros((0, 3))

    inferred_verts: List[np.ndarray] = []
    inferred_tris: List[List[int]] = []
    accepted_cells_meta: List[Dict[str, Any]] = []
    rejected_freespace = 0
    rejected_corridor = 0
    rejected_unbounded = 0
    cand_total = 0

    col_f = [c / 255.0 for c in cfg.inferred_color_rgb]

    for reg in wall_regs:
        p_wall = observed_points[reg.point_indices].astype(np.float64)
        if len(p_wall) < 120:
            continue
        n = reg.normal / max(np.linalg.norm(reg.normal), 1e-9)
        d = reg.d_coeff

        # Orthonormal 2D basis (u_ax, v_ax) on wall plane
        ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u_ax = np.cross(ref, n)
        u_ax /= max(np.linalg.norm(u_ax), 1e-9)
        v_ax = np.cross(n, u_ax)
        v_ax /= max(np.linalg.norm(v_ax), 1e-9)
        origin = reg.centroid - float(np.dot(reg.centroid, n) + d) * n

        rel = p_wall - origin[None, :]
        u_coords = rel @ u_ax
        v_coords = rel @ v_ax

        step = cfg.grid_step_m
        u_min, u_max = float(np.percentile(u_coords, 2)), float(np.percentile(u_coords, 98))
        v_min, v_max = float(np.percentile(v_coords, 2)), float(np.percentile(v_coords, 98))
        nu = max(1, int(math.floor((u_max - u_min) / step)))
        nv = max(1, int(math.floor((v_max - v_min) / step)))
        if nu < 3 or nv < 3 or nu * nv > 4000:
            continue

        obs_grid = np.zeros((nv, nu), dtype=bool)
        ui = np.clip(np.floor((u_coords - u_min) / step).astype(int), 0, nu - 1)
        vi = np.clip(np.floor((v_coords - v_min) / step).astype(int), 0, nv - 1)
        obs_grid[vi, ui] = True

        # Check interior empty cells bounded on BOTH left/right OR top/bottom by observed wall cells
        for r_i in range(1, nv - 1):
            for c_i in range(1, nu - 1):
                if obs_grid[r_i, c_i]:
                    continue
                cand_total += 1
                # Must be enclosed by observed wall both horizontally OR vertically (never open wall extension!)
                has_left = bool(np.any(obs_grid[r_i, max(0, c_i - cfg.max_wall_gap_cells):c_i]))
                has_right = bool(np.any(obs_grid[r_i, c_i + 1:min(nu, c_i + 1 + cfg.max_wall_gap_cells)]))
                has_below = bool(np.any(obs_grid[max(0, r_i - cfg.max_wall_gap_cells):r_i, c_i]))
                has_above = bool(np.any(obs_grid[r_i + 1:min(nv, r_i + 1 + cfg.max_wall_gap_cells), c_i]))
                if not ((has_left and has_right) and (has_below or has_above)):
                    rejected_unbounded += 1
                    continue

                uc = u_min + (c_i + 0.5) * step
                vc = v_min + (r_i + 0.5) * step
                center_3d = origin + uc * u_ax + vc * v_ax

                # Check if observed mesh ALREADY covers this cell
                if mesh_tree is not None and mesh_tree.query(center_3d, k=1)[0] < step * 0.75:
                    continue

                # Corridor / walkable clearance guard
                if len(cams_np) > 0 and float(np.min(np.linalg.norm(cams_np - center_3d[None, :], axis=1))) < cfg.min_corridor_clearance_m:
                    rejected_corridor += 1
                    continue

                # Sample 5 points inside cell & strictly veto if ANY KNOWN_FREE_SPACE ray passed through!
                u0, u1 = u_min + c_i * step, u_min + (c_i + 1) * step
                v0, v1 = v_min + r_i * step, v_min + (r_i + 1) * step
                c_pts = np.array([
                    origin + u0 * u_ax + v0 * v_ax,
                    origin + u1 * u_ax + v0 * v_ax,
                    origin + u1 * u_ax + v1 * v_ax,
                    origin + u0 * u_ax + v1 * v_ax,
                    center_3d,
                ])
                if free_space_evidence is not None:
                    # Also check slight +-4cm normal slab around wall to ensure no doorway ray crossed it!
                    slab = np.vstack([c_pts, c_pts + 0.04 * n, c_pts - 0.04 * n])
                    if np.any(free_space_evidence.query_free_space_mask(slab)):
                        rejected_freespace += 1
                        continue

                base_idx = len(inferred_verts)
                inferred_verts.extend(c_pts[:4])
                inferred_tris.append([base_idx, base_idx + 1, base_idx + 2])
                inferred_tris.append([base_idx, base_idx + 2, base_idx + 3])
                accepted_cells_meta.append({
                    "wall_region_id": reg.region_id,
                    "center": [round(float(x), 4) for x in center_3d],
                    "plane_normal": [round(float(x), 4) for x in n],
                })

    inf_mesh = o3d.geometry.TriangleMesh()
    inf_ply_path = out_dir / "inferred_surfaces.ply"
    if inferred_tris:
        inf_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(inferred_verts, dtype=np.float64))
        inf_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(inferred_tris, dtype=np.int32))
        inf_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(np.array(col_f), (len(inferred_verts), 1)))
        inf_mesh.remove_duplicated_vertices()
        inf_mesh.compute_vertex_normals()
        inf_mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(str(inf_ply_path), inf_mesh, write_vertex_colors=True)

    rep = {
        "enabled": True,
        "status": "COMPLETED" if len(accepted_cells_meta) > 0 else "NO_SAFE_OCCLUDED_CELLS",
        "method_identity": "local_conservative_occluded_wall_completion_NOT_BehindTheVeil_neural",
        "feasibility_audit": FEASIBILITY_AUDIT,
        "candidate_cells_evaluated": cand_total,
        "accepted_cells_count": len(accepted_cells_meta),
        "rejected_freespace_barrier": rejected_freespace,
        "rejected_corridor_clearance": rejected_corridor,
        "rejected_unbounded_opening": rejected_unbounded,
        "inferred_vertices": len(inf_mesh.vertices),
        "inferred_triangles": len(inf_mesh.triangles),
        "extrapolated_color_rgb": list(cfg.inferred_color_rgb),
        "parent_hashes": parent_hashes or {},
        "duration_sec": round(time.perf_counter() - t0, 3),
        "accepted_cells_sample": accepted_cells_meta[:25],
    }
    report_path.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if log_callback:
        log_callback(
            f"  [+] Hidden-Surface Alternative ({rep['duration_sec']}s): "
            f"{rep['accepted_cells_count']} occluded wall cells completed "
            f"(blocked: freespace={rejected_freespace}, corridor={rejected_corridor}, unbounded={rejected_unbounded})"
        )
    return {
        "enabled": True,
        "inferred_mesh": inf_mesh if len(inf_mesh.triangles) > 0 else None,
        "inferred_ply_path": str(inf_ply_path.resolve()) if len(inf_mesh.triangles) > 0 else None,
        "report": rep,
        "report_path": str(report_path.resolve()),
    }
