"""
F.O.R.G.E. V2.2 — Continuous Boundary Gap Classifier, Boundary-Pinned Triangulation (A/B1),
Strict Polygon-Clipped Localized Poisson (B2), Vertex Stitching, Per-Patch Local QA,
and Before/After + Gap Diagnostic Rendering.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree


def _sample_points_inside_polygon_2d(
    poly_2d: np.ndarray,
    cent_2d: Optional[np.ndarray] = None,
    step_m: float = 0.025,
    min_wall_dist_m: float = 0.012,
    max_samples: int = 120,
) -> np.ndarray:
    """Samples evenly spaced interior points inside a closed 2D polygon away from its boundary."""
    if len(poly_2d) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    path = MplPath(poly_2d)
    mn = poly_2d.min(axis=0)
    mx = poly_2d.max(axis=0)
    span = mx - mn
    if np.any(span <= 1e-6):
        return np.zeros((0, 2), dtype=np.float64)

    nx = min(28, max(3, int(math.ceil(span[0] / max(step_m, 1e-4))) + 1))
    ny = min(28, max(3, int(math.ceil(span[1] / max(step_m, 1e-4))) + 1))
    gx = np.linspace(mn[0], mx[0], nx)
    gy = np.linspace(mn[1], mx[1], ny)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    inside = path.contains_points(grid)
    cands = grid[inside]
    if len(cands) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Reject candidates closer to any polygon boundary segment than min_wall_dist_m
    p0 = poly_2d
    p1 = np.roll(poly_2d, -1, axis=0)
    seg = p1 - p0
    seg_len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-12)
    diff = cands[:, None, :] - p0[None, :, :]
    t = np.clip(np.sum(diff * seg[None, :, :], axis=2) / seg_len2[None, :], 0.0, 1.0)
    proj = p0[None, :, :] + t[:, :, None] * seg[None, :, :]
    min_d = np.linalg.norm(cands[:, None, :] - proj, axis=2).min(axis=1)
    kept = cands[min_d >= min_wall_dist_m]
    if len(kept) > max_samples:
        step = max(1, len(kept) // max_samples)
        kept = kept[::step][:max_samples]
    return kept


def _ear_clip_polygon_2d(poly_2d: np.ndarray) -> np.ndarray:
    """
    Robust 2D CCW simple polygon ear-clipping triangulation returning local index triangles (M, 3).
    Guarantees every input boundary edge (i, (i+1)%L) is preserved identically.
    """
    n = len(poly_2d)
    if n < 3:
        return np.zeros((0, 3), dtype=np.int32)
    if n == 3:
        area = 0.5 * (
            (poly_2d[1, 0] - poly_2d[0, 0]) * (poly_2d[2, 1] - poly_2d[0, 1])
            - (poly_2d[1, 1] - poly_2d[0, 1]) * (poly_2d[2, 0] - poly_2d[0, 0])
        )
        return np.array([[0, 1, 2]] if area >= 0 else [[0, 2, 1]], dtype=np.int32)

    # Ensure CCW order for ear-clipping
    s_area = 0.5 * float(np.dot(poly_2d[:, 0], np.roll(poly_2d[:, 1], -1)) - np.dot(poly_2d[:, 1], np.roll(poly_2d[:, 0], -1)))
    rem = list(range(n)) if s_area >= 0 else list(reversed(range(n)))

    def _cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def _pt_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        c1 = _cross2(a, b, p)
        c2 = _cross2(b, c, p)
        c3 = _cross2(c, a, p)
        return (c1 >= -1e-10) and (c2 >= -1e-10) and (c3 >= -1e-10)

    tris: List[List[int]] = []
    guard = 0
    max_iter = n * n + 20
    while len(rem) > 3 and guard < max_iter:
        guard += 1
        found_ear = False
        m = len(rem)
        for k in range(m):
            i_prev = rem[(k - 1) % m]
            i_curr = rem[k]
            i_next = rem[(k + 1) % m]
            pa, pb, pc = poly_2d[i_prev], poly_2d[i_curr], poly_2d[i_next]
            if _cross2(pa, pb, pc) <= 1e-12:
                continue
            has_inside = False
            for j in range(m):
                idx_j = rem[j]
                if idx_j in (i_prev, i_curr, i_next):
                    continue
                if _pt_in_tri(poly_2d[idx_j], pa, pb, pc):
                    has_inside = True
                    break
            if not has_inside:
                tris.append([i_prev, i_curr, i_next])
                rem.pop(k)
                found_ear = True
                break
        if not found_ear:
            # Degenerate/slightly pinched polygon fallback: clip sharpest convex vertex
            best_k = None
            best_area = -1.0
            for k in range(m):
                i_prev, i_curr, i_next = rem[(k - 1) % m], rem[k], rem[(k + 1) % m]
                cr = _cross2(poly_2d[i_prev], poly_2d[i_curr], poly_2d[i_next])
                if cr > best_area:
                    best_area = cr
                    best_k = k
            if best_k is None:
                break
            i_prev, i_curr, i_next = rem[(best_k - 1) % m], rem[best_k], rem[(best_k + 1) % m]
            tris.append([i_prev, i_curr, i_next])
            rem.pop(best_k)

    if len(rem) == 3:
        if _cross2(poly_2d[rem[0]], poly_2d[rem[1]], poly_2d[rem[2]]) < 0:
            tris.append([rem[0], rem[2], rem[1]])
        else:
            tris.append([rem[0], rem[1], rem[2]])
    return np.asarray(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def _refine_interior_triangulation(
    pts_2d: np.ndarray,
    tris: np.ndarray,
    n_boundary: int,
    max_interior_edge_m: float,
    max_iters: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits ONLY interior non-boundary edges longer than `max_interior_edge_m` and smooths
    interior vertices while holding all `n_boundary` outer rim vertices fixed (zero seam drift).
    """
    pts = pts_2d.copy()
    faces = tris.copy()
    boundary_edges: Set[Tuple[int, int]] = {
        (min(i, (i + 1) % n_boundary), max(i, (i + 1) % n_boundary))
        for i in range(n_boundary)
    }
    if len(faces) == 0 or max_interior_edge_m <= 1e-5:
        return pts, faces

    for _ in range(max_iters):
        e_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx, t in enumerate(faces):
            for a, b in ((int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))):
                ek = (min(a, b), max(a, b))
                e_to_faces.setdefault(ek, []).append(f_idx)

        longest_edge = None
        longest_len = max_interior_edge_m
        for ek, f_list in e_to_faces.items():
            if ek in boundary_edges or len(f_list) != 2:
                continue
            elen = float(np.linalg.norm(pts[ek[0]] - pts[ek[1]]))
            if elen > longest_len:
                longest_len = elen
                longest_edge = ek

        if longest_edge is None or len(pts) >= n_boundary + 180:
            break

        # Split all qualifying long interior edges disjointly in this pass
        split_map: Dict[Tuple[int, int], int] = {}
        new_pts_list = list(pts)
        # Sort longest first and avoid splitting two edges of the same triangle in one pass
        cand_edges = [
            (float(np.linalg.norm(pts[ek[0]] - pts[ek[1]])), ek)
            for ek, f_list in e_to_faces.items()
            if ek not in boundary_edges and len(f_list) == 2 and float(np.linalg.norm(pts[ek[0]] - pts[ek[1]])) > max_interior_edge_m
        ]
        cand_edges.sort(reverse=True)
        touched_faces: Set[int] = set()
        for _, ek in cand_edges:
            if len(new_pts_list) >= n_boundary + 180:
                break
            fl = e_to_faces[ek]
            if any(fi in touched_faces for fi in fl):
                continue
            mid = 0.5 * (pts[ek[0]] + pts[ek[1]])
            mid_idx = len(new_pts_list)
            new_pts_list.append(mid)
            split_map[ek] = mid_idx
            touched_faces.update(fl)

        if not split_map:
            break

        new_faces = []
        for fi, t in enumerate(faces):
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            s_ab = split_map.get((min(a, b), max(a, b)))
            s_bc = split_map.get((min(b, c), max(b, c)))
            s_ca = split_map.get((min(c, a), max(c, a)))
            if s_ab is not None:
                new_faces.append([a, s_ab, c])
                new_faces.append([s_ab, b, c])
            elif s_bc is not None:
                new_faces.append([a, b, s_bc])
                new_faces.append([a, s_bc, c])
            elif s_ca is not None:
                new_faces.append([a, b, s_ca])
                new_faces.append([s_ca, b, c])
            else:
                new_faces.append([a, b, c])
        pts = np.asarray(new_pts_list, dtype=np.float64)
        faces = np.asarray(new_faces, dtype=np.int32)

    # Gentle fixed-boundary Laplacian smoothing of newly created interior vertices
    if len(pts) > n_boundary and len(faces) > 0:
        nbrs: List[Set[int]] = [set() for _ in range(len(pts))]
        for t in faces:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            nbrs[a].update((b, c))
            nbrs[b].update((a, c))
            nbrs[c].update((a, b))
        for _ in range(3):
            nxt = pts.copy()
            for vi in range(n_boundary, len(pts)):
                if len(nbrs[vi]) >= 2:
                    nxt[vi] = 0.5 * pts[vi] + 0.5 * np.mean(pts[list(nbrs[vi])], axis=0)
            pts = nxt
    return pts, faces


def _blocked_existing_chord_edges(
    existing_tris: Optional[np.ndarray],
    loop_arr: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    Returns unordered global edges in `existing_tris` that MUST NOT be reused inside a repair
    patch (edges already interior `count>=2` or existing mesh chords between non-adjacent rim vertices).
    """
    if existing_tris is None or len(existing_tris) == 0 or len(loop_arr) < 3:
        return set()
    L = len(loop_arr)
    rim_edges: Set[Tuple[int, int]] = {
        (int(min(loop_arr[i], loop_arr[(i + 1) % L])), int(max(loop_arr[i], loop_arr[(i + 1) % L])))
        for i in range(L)
    }
    in_l = (
        np.isin(existing_tris[:, 0], loop_arr)
        | np.isin(existing_tris[:, 1], loop_arr)
        | np.isin(existing_tris[:, 2], loop_arr)
    )
    sub = existing_tris[in_l]
    if len(sub) == 0:
        return set()
    all_e = np.vstack([np.sort(sub[:, [0, 1]], 1), np.sort(sub[:, [1, 2]], 1), np.sort(sub[:, [2, 0]], 1)])
    ue, ec = np.unique(all_e, axis=0, return_counts=True)
    loop_set = {int(v) for v in loop_arr}
    blocked: Set[Tuple[int, int]] = set()
    for (u, v), c in zip(ue, ec):
        ek = (int(u), int(v))
        if int(c) >= 2:
            blocked.add(ek)
        elif ek not in rim_edges and (int(u) in loop_set) and (int(v) in loop_set):
            blocked.add(ek)
    return blocked


def _split_patch_chord_collisions(
    pts_arr: np.ndarray,
    tris_arr: np.ndarray,
    stitched_gid: np.ndarray,
    blocked_edges: Set[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits any patch triangle edge whose global vertex pair collides with `blocked_edges`
    (prevents creating non-manifold edges across existing mesh chords/diagonals).
    """
    if len(tris_arr) == 0 or not blocked_edges:
        return pts_arr, tris_arr, stitched_gid

    pts = list(pts_arr)
    gids = list(stitched_gid)
    faces = tris_arr.tolist()
    edge_mid: Dict[Tuple[int, int], int] = {}

    for _ in range(3):
        split_any = False
        new_faces: List[List[int]] = []
        for a, b, c in faces:
            colliding = None
            for u, v, opp in ((a, b, c), (b, c, a), (c, a, b)):
                gu, gv = int(gids[u]), int(gids[v])
                if gu >= 0 and gv >= 0 and (min(gu, gv), max(gu, gv)) in blocked_edges:
                    colliding = (u, v, opp)
                    break
            if colliding is None:
                new_faces.append([a, b, c])
            else:
                split_any = True
                u, v, opp = colliding
                ek = (min(u, v), max(u, v))
                if ek not in edge_mid:
                    mid_idx = len(pts)
                    pts.append(0.5 * (pts[u] + pts[v]))
                    gids.append(-1)
                    edge_mid[ek] = mid_idx
                m = edge_mid[ek]
                new_faces.append([u, m, opp])
                new_faces.append([m, v, opp])
        faces = new_faces
        if not split_any:
            break

    return np.asarray(pts, dtype=np.float64), np.asarray(faces, dtype=np.int32), np.asarray(gids, dtype=np.int64)


def _lift_interior_and_relax(
    pts_2d_all: np.ndarray,
    tris_local: np.ndarray,
    pts_l: np.ndarray,
    cent: np.ndarray,
    u_ax: np.ndarray,
    v_ax: np.ndarray,
    n_hole: np.ndarray,
    surface_pts_3d: np.ndarray,
    sigma_m: float,
    max_height_delta_m: float,
    laplacian_iters: int = 4,
) -> np.ndarray:
    """
    Lifts 2D constrained vertices `[L..N-1]` to 3D via smooth Gaussian kernel height regression over
    `surface_pts_3d` (holding boundary vertices `[0..L-1]` pinned at exact `pts_l`), then relaxes
    interior vertices with fixed-boundary Laplacian smoothing to eliminate jagged micro-bumps.
    """
    L = len(pts_l)
    n_tot = len(pts_2d_all)
    pts_3d = np.zeros((n_tot, 3), dtype=np.float64)
    pts_3d[:L] = pts_l
    if n_tot <= L:
        return pts_3d

    poly_2d = pts_2d_all[:L]
    inner_2d = pts_2d_all[L:]
    base_3d = cent[None, :] + inner_2d[:, 0:1] * u_ax[None, :] + inner_2d[:, 1:2] * v_ax[None, :]

    anchor_2d = [poly_2d]
    anchor_h = [(pts_l - cent) @ n_hole]
    if len(surface_pts_3d) >= 3:
        sp = surface_pts_3d.astype(np.float64)
        sp_2d = np.column_stack([(sp - cent) @ u_ax, (sp - cent) @ v_ax])
        sp_h = np.clip((sp - cent) @ n_hole, -max_height_delta_m, max_height_delta_m)
        anchor_2d.append(sp_2d)
        anchor_h.append(sp_h)

    anc_xy = np.vstack(anchor_2d)
    anc_z = np.concatenate(anchor_h)
    d2 = np.sum((inner_2d[:, None, :] - anc_xy[None, :, :]) ** 2, axis=2)
    inv_2s2 = 1.0 / (2.0 * max(sigma_m, 0.02) ** 2)
    w_gauss = np.exp(-d2 * inv_2s2) + 1e-9
    h_est = np.sum(w_gauss * anc_z[None, :], axis=1) / np.sum(w_gauss, axis=1)
    h_est = np.clip(h_est, -max_height_delta_m, max_height_delta_m)
    pts_3d[L:] = base_3d + h_est[:, None] * n_hole[None, :]

    # Fixed-boundary 3D Laplacian smoothing on interior vertices ONLY
    if laplacian_iters > 0 and len(tris_local) > 0:
        nbrs: List[Set[int]] = [set() for _ in range(n_tot)]
        for a, b, c in tris_local:
            ia, ib, ic = int(a), int(b), int(c)
            nbrs[ia].update((ib, ic))
            nbrs[ib].update((ia, ic))
            nbrs[ic].update((ia, ib))
        for _ in range(laplacian_iters):
            nxt = pts_3d.copy()
            for vi in range(L, n_tot):
                if len(nbrs[vi]) >= 2:
                    nxt[vi] = 0.45 * pts_3d[vi] + 0.55 * np.mean(pts_3d[list(nbrs[vi])], axis=0)
            pts_3d = nxt
    return pts_3d


def build_boundary_pinned_planar_patch(
    hole: Dict[str, Any],
    mesh_verts: np.ndarray,
    pts_obs: np.ndarray,
    interior_sup_indices: np.ndarray,
    voxel_size: float,
    max_height_delta_m: float,
    allow_interior_subdivision: bool = True,
    existing_tris: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a manifold Case A or Case B1 patch via Constrained 2D Polygon Boundary Triangulation.
    Guarantees 100% shared original boundary loop vertices (`loop_v`), zero missing boundary edges,
    and smooth Gaussian + harmonic lifting onto supported observations.
    """
    loop_v = np.asarray(hole["vertex_indices"], dtype=np.int64)
    seen_v: Dict[int, int] = {}
    unique_loop: List[int] = []
    for vi in loop_v:
        v_int = int(vi)
        if v_int not in seen_v:
            seen_v[v_int] = len(unique_loop)
            unique_loop.append(v_int)
    loop_arr = np.asarray(unique_loop, dtype=np.int64)
    L = len(loop_arr)
    if L < 3:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    pts_l = mesh_verts[loop_arr]
    cent = np.asarray(hole["centroid"], dtype=np.float64)
    u_ax = np.asarray(hole["u_axis"], dtype=np.float64)
    v_ax = np.asarray(hole["v_axis"], dtype=np.float64)
    n_hole = np.asarray(hole["normal"], dtype=np.float64)

    poly_2d = np.column_stack([(pts_l - cent) @ u_ax, (pts_l - cent) @ v_ax])
    diam_m = float(hole["diameter_m"])
    step_m = max(voxel_size * 1.4, min(0.045, diam_m / 6.0))

    used_delaunay = False
    pts_2d_all = poly_2d
    tris_local = np.zeros((0, 3), dtype=np.int32)

    if allow_interior_subdivision and L >= 6 and diam_m > voxel_size * 3.0:
        inner_2d = _sample_points_inside_polygon_2d(
            poly_2d, step_m=step_m, min_wall_dist_m=max(0.008, step_m * 0.52), max_samples=120
        )
        if len(inner_2d) > 0:
            cand_2d = np.vstack([poly_2d, inner_2d])
            try:
                tri_del = Delaunay(cand_2d)
                simp = tri_del.simplices.astype(np.int32)
                tri_cents = cand_2d[simp].mean(axis=1)
                keep = MplPath(poly_2d).contains_points(tri_cents)
                simp_in = simp[keep]
                if len(simp_in) > 0:
                    e0 = np.sort(simp_in[:, [0, 1]], axis=1)
                    e1 = np.sort(simp_in[:, [1, 2]], axis=1)
                    e2 = np.sort(simp_in[:, [2, 0]], axis=1)
                    all_e = np.vstack([e0, e1, e2])
                    ue, ec = np.unique(all_e, axis=0, return_counts=True)
                    boundary_set = {(int(min(a, b)), int(max(a, b))) for (a, b), c in zip(ue, ec) if c == 1}
                    expected_b = {(min(i, (i + 1) % L), max(i, (i + 1) % L)) for i in range(L)}
                    if boundary_set == expected_b and np.all(ec <= 2):
                        pts_2d_all = cand_2d
                        tris_local = simp_in
                        used_delaunay = True
            except Exception:
                used_delaunay = False

    if not used_delaunay:
        tris_local = _ear_clip_polygon_2d(poly_2d)
        if allow_interior_subdivision and len(tris_local) > 0 and diam_m > voxel_size * 3.0:
            pts_2d_all, tris_local = _refine_interior_triangulation(
                poly_2d, tris_local, n_boundary=L, max_interior_edge_m=max(voxel_size * 2.2, step_m * 1.45)
            )

    if len(tris_local) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    blocked = _blocked_existing_chord_edges(existing_tris, loop_arr)
    if blocked:
        init_gids = np.full(len(pts_2d_all), -1, dtype=np.int64)
        init_gids[:L] = loop_arr
        pts_2d_all, tris_local, _ = _split_patch_chord_collisions(pts_2d_all, tris_local, init_gids, blocked)

    sup_3d = pts_obs[interior_sup_indices] if len(interior_sup_indices) >= 3 else np.zeros((0, 3), dtype=np.float64)
    pts_3d = _lift_interior_and_relax(
        pts_2d_all, tris_local, pts_l, cent, u_ax, v_ax, n_hole,
        surface_pts_3d=sup_3d,
        sigma_m=max(0.06, voxel_size * 4.5),
        max_height_delta_m=max_height_delta_m,
        laplacian_iters=4,
    )

    p0, p1, p2 = pts_3d[tris_local[:, 0]], pts_3d[tris_local[:, 1]], pts_3d[tris_local[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    flip_mask = (fn @ n_hole) < 0.0
    if np.any(flip_mask):
        tris_local[flip_mask] = tris_local[flip_mask][:, [0, 2, 1]]

    stitched_gids = np.full(len(pts_3d), -1, dtype=np.int64)
    stitched_gids[:L] = loop_arr
    return pts_3d, tris_local, stitched_gids


def build_polygon_clipped_stitched_poisson_patch(
    hole: Dict[str, Any],
    mesh_verts: np.ndarray,
    mesh_vnorms: np.ndarray,
    pts_obs: np.ndarray,
    nrms_obs: np.ndarray,
    rim_sup_indices: np.ndarray,
    voxel_size: float,
    poisson_depth: int,
    density_trim_quantile: float,
    band_halfwidth_m: float,
    existing_tris: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Case B2 localized Screened Poisson repair for curved qualifying gaps:
      1. Trains ONLY on strict co-normal canonical support points + TSDF loop vertices.
      2. Clips strictly inside 2D projected boundary polygon + tiny seam tolerance and tight
         orthogonal band (<= band_halfwidth_m), preventing distant hovering confetti layers.
      3. Extracts ONLY the single largest connected patch component.
      4. Welds patch boundary vertices onto original `loop_v` indices.

    Returns `(patch_verts_3d, patch_tris_local, stitched_global_ids)` where any vertex with
    `stitched_global_ids[k] >= 0` maps directly to an existing TSDF vertex (`loop_v`), and `< 0`
    becomes a newly appended interior vertex.
    """
    loop_arr = np.asarray(hole["vertex_indices"], dtype=np.int64)
    if len(loop_arr) < 3 or len(rim_sup_indices) < 25:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    pts_l = mesh_verts[loop_arr]
    nrms_l = mesh_vnorms[loop_arr]
    cent = np.asarray(hole["centroid"], dtype=np.float64)
    u_ax = np.asarray(hole["u_axis"], dtype=np.float64)
    v_ax = np.asarray(hole["v_axis"], dtype=np.float64)
    n_hole = np.asarray(hole["normal"], dtype=np.float64)
    poly_2d = np.column_stack([(pts_l - cent) @ u_ax, (pts_l - cent) @ v_ax])

    # Weight loop rim vertices slightly by duplicating once so Poisson surface anchors cleanly to TSDF rim
    sup_p = pts_obs[rim_sup_indices].astype(np.float64)
    sup_n = nrms_obs[rim_sup_indices].astype(np.float64)
    # Orient all support normals consistently toward n_hole
    flip_n = (sup_n @ n_hole) < 0.0
    if np.any(flip_n):
        sup_n[flip_n] *= -1.0
    nrms_l_oriented = nrms_l.copy()
    flip_nl = (nrms_l_oriented @ n_hole) < 0.0
    if np.any(flip_nl):
        nrms_l_oriented[flip_nl] *= -1.0

    comb_p = np.vstack([sup_p, pts_l, pts_l])
    comb_n = np.vstack([sup_n, nrms_l_oriented, nrms_l_oriented])
    norms = np.linalg.norm(comb_n, axis=1, keepdims=True)
    comb_n = comb_n / np.maximum(norms, 1e-8)

    pcd_loc = o3d.geometry.PointCloud()
    pcd_loc.points = o3d.utility.Vector3dVector(comb_p)
    pcd_loc.normals = o3d.utility.Vector3dVector(comb_n)

    p_depth = int(min(max(5, poisson_depth), 8))
    try:
        pmesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_loc, depth=p_depth, width=0.0, scale=1.10, linear_fit=False
        )
    except Exception:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    dens = np.asarray(densities)
    if len(dens) > 10 and density_trim_quantile > 0.0:
        pmesh.remove_vertices_by_mask(dens < np.quantile(dens, min(0.25, density_trim_quantile)))
    pmesh.remove_degenerate_triangles()
    pmesh.remove_duplicated_vertices()
    pmesh.remove_duplicated_triangles()
    pmesh.remove_unreferenced_vertices()

    pv = np.asarray(pmesh.vertices, dtype=np.float64)
    pt = np.asarray(pmesh.triangles, dtype=np.int32)
    if len(pt) < 2:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    # Strict 2D polygon boundary + tight orthogonal band cropping (zero distant hovering walls!)
    fc = pv[pt].mean(axis=1)
    fc_rel = fc - cent
    fc_2d = np.column_stack([fc_rel @ u_ax, fc_rel @ v_ax])
    fc_perp = np.abs(fc_rel @ n_hole)

    # Include centroids strictly inside 2D hole OR within small boundary-weld overlap ring
    in_poly = MplPath(poly_2d).contains_points(fc_2d, radius=max(0.010, voxel_size * 0.95))
    in_band = fc_perp <= max(0.032, band_halfwidth_m)

    # Normal agreement check against hole orientation
    pmesh.compute_triangle_normals()
    pfn = np.asarray(pmesh.triangle_normals, dtype=np.float64)
    norm_ok = np.abs(pfn @ n_hole) >= 0.55

    keep_t = in_poly & in_band & norm_ok
    if np.count_nonzero(keep_t) < 2:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    pmesh.remove_triangles_by_mask(~keep_t)
    pmesh.remove_unreferenced_vertices()

    # Keep ONLY the single largest connected component (ZERO multi-island confetti!)
    clusters, counts, _ = pmesh.cluster_connected_triangles()
    c_np = np.asarray(counts)
    if len(c_np) == 0:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)
    best_cid = int(np.argmax(c_np))
    pmesh.remove_triangles_by_mask(np.asarray(clusters) != best_cid)
    pmesh.remove_unreferenced_vertices()

    pv = np.asarray(pmesh.vertices, dtype=np.float64)
    pt = np.asarray(pmesh.triangles, dtype=np.int32)
    if len(pt) < 1 or len(pv) < 3:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    # Map cropped Poisson curved surface vertices smoothly onto exact boundary-pinned skeleton
    # (100% continuous stitch along every TSDF rim edge, curved Poisson surface height inside!)
    unique_loop: List[int] = []
    seen_v: Dict[int, int] = {}
    for vi in loop_arr:
        v_i = int(vi)
        if v_i not in seen_v:
            seen_v[v_i] = len(unique_loop)
            unique_loop.append(v_i)
    u_loop = np.asarray(unique_loop, dtype=np.int64)
    L = len(u_loop)
    pts_ul = mesh_verts[u_loop]
    poly_2d_u = np.column_stack([(pts_ul - cent) @ u_ax, (pts_ul - cent) @ v_ax])
    diam_m = float(hole["diameter_m"])
    step_m = max(voxel_size * 1.4, min(0.045, diam_m / 6.0))

    inner_2d = _sample_points_inside_polygon_2d(
        poly_2d_u, step_m=step_m, min_wall_dist_m=max(0.008, step_m * 0.52), max_samples=140
    )
    pts_2d_all = np.vstack([poly_2d_u, inner_2d]) if len(inner_2d) > 0 else poly_2d_u
    tris_local = np.zeros((0, 3), dtype=np.int32)
    used_del = False
    if len(inner_2d) > 0 and L >= 6:
        try:
            simp = Delaunay(pts_2d_all).simplices.astype(np.int32)
            tri_c = pts_2d_all[simp].mean(axis=1)
            simp_in = simp[MplPath(poly_2d_u).contains_points(tri_c)]
            if len(simp_in) > 0:
                ae = np.vstack([np.sort(simp_in[:, [0, 1]], 1), np.sort(simp_in[:, [1, 2]], 1), np.sort(simp_in[:, [2, 0]], 1)])
                ue, ec = np.unique(ae, axis=0, return_counts=True)
                b_set = {(int(min(a, b)), int(max(a, b))) for (a, b), c in zip(ue, ec) if c == 1}
                exp_b = {(min(i, (i + 1) % L), max(i, (i + 1) % L)) for i in range(L)}
                if b_set == exp_b and np.all(ec <= 2):
                    tris_local = simp_in
                    used_del = True
        except Exception:
            used_del = False
    if not used_del:
        pts_2d_all = poly_2d_u
        tris_local = _ear_clip_polygon_2d(poly_2d_u)
        if len(tris_local) > 0 and diam_m > voxel_size * 3.0:
            pts_2d_all, tris_local = _refine_interior_triangulation(
                poly_2d_u, tris_local, n_boundary=L, max_interior_edge_m=max(voxel_size * 2.2, step_m * 1.45)
            )
    if len(tris_local) == 0:
        empty_v = np.zeros((0, 3), dtype=np.float64)
        return empty_v, np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.int64)

    blocked = _blocked_existing_chord_edges(existing_tris, u_loop)
    if blocked:
        init_gids = np.full(len(pts_2d_all), -1, dtype=np.int64)
        init_gids[:L] = u_loop
        pts_2d_all, tris_local, _ = _split_patch_chord_collisions(pts_2d_all, tris_local, init_gids, blocked)

    # Lift interior vertices smoothly onto the cropped Screened Poisson surface pv!
    pts_out = _lift_interior_and_relax(
        pts_2d_all, tris_local, pts_ul, cent, u_ax, v_ax, n_hole,
        surface_pts_3d=pv,
        sigma_m=max(0.045, voxel_size * 3.0),
        max_height_delta_m=max(0.032, band_halfwidth_m),
        laplacian_iters=3,
    )

    p0, p1, p2 = pts_out[tris_local[:, 0]], pts_out[tris_local[:, 1]], pts_out[tris_local[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    flip_mask = (fn @ n_hole) < 0.0
    if np.any(flip_mask):
        tris_local[flip_mask] = tris_local[flip_mask][:, [0, 2, 1]]

    stitched_gid = np.full(len(pts_out), -1, dtype=np.int64)
    stitched_gid[:L] = u_loop
    return pts_out, tris_local, stitched_gid


def align_patch_winding_to_mesh(
    patch_tris_global: np.ndarray,
    existing_tris: np.ndarray,
    loop_vertices: Set[int],
) -> np.ndarray:
    """
    Ensures manifold consistent triangle winding against adjacent existing mesh triangles along
    shared boundary edges (new face traverses edge {u,v} opposite to existing face).
    """
    if len(patch_tris_global) == 0 or len(existing_tris) == 0:
        return patch_tris_global
    # Record directed edges of existing faces incident on loop vertices
    in_loop = np.isin(existing_tris[:, 0], list(loop_vertices)) & np.isin(existing_tris[:, 1], list(loop_vertices))
    in_loop |= np.isin(existing_tris[:, 1], list(loop_vertices)) & np.isin(existing_tris[:, 2], list(loop_vertices))
    in_loop |= np.isin(existing_tris[:, 2], list(loop_vertices)) & np.isin(existing_tris[:, 0], list(loop_vertices))
    directed_exist: Set[Tuple[int, int]] = set()
    for t in existing_tris[in_loop]:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        directed_exist.add((a, b))
        directed_exist.add((b, c))
        directed_exist.add((c, a))

    same_dir = 0
    opp_dir = 0
    for t in patch_tris_global:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if (u, v) in directed_exist:
                same_dir += 1
            elif (v, u) in directed_exist:
                opp_dir += 1

    out = patch_tris_global.copy()
    if same_dir > opp_dir:
        out = out[:, [0, 2, 1]]
    return out


def assemble_stitched_patch_into_mesh(
    base_verts: np.ndarray,
    base_tris: np.ndarray,
    base_vcols: np.ndarray,
    patch_pts_3d: np.ndarray,
    patch_tris_local: np.ndarray,
    stitched_global_ids: np.ndarray,
    pts_obs: np.ndarray,
    cols_obs: np.ndarray,
    pt_tree: cKDTree,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stitches a patch into a base mesh sharing exact boundary vertices (global IDs >= 0) and
    appending new interior vertices cleanly.
    Returns `(new_verts, new_tris, new_vcols, added_global_tris, new_interior_pts_3d)`.
    """
    local_to_global = np.full(len(patch_pts_3d), -1, dtype=np.int64)
    new_verts_list: List[np.ndarray] = []
    new_cols_list: List[np.ndarray] = []
    next_v = len(base_verts)

    new_interior_mask = stitched_global_ids < 0
    new_interior_pts = patch_pts_3d[new_interior_mask]

    # Sample colors smoothly from nearest canonical observed point
    if len(new_interior_pts) > 0:
        _, nn_col_idx = pt_tree.query(new_interior_pts, k=1)
        new_cols_arr = cols_obs[nn_col_idx].astype(np.float64) / 255.0
    else:
        new_cols_arr = np.zeros((0, 3), dtype=np.float64)

    int_ptr = 0
    for k in range(len(patch_pts_3d)):
        gid = int(stitched_global_ids[k])
        if gid >= 0:
            local_to_global[k] = gid
        else:
            local_to_global[k] = next_v
            new_verts_list.append(patch_pts_3d[k])
            new_cols_list.append(new_cols_arr[int_ptr])
            next_v += 1
            int_ptr += 1

    added_tris = local_to_global[patch_tris_local].astype(np.int32)
    # Remove any duplicate/degenerate triangles against existing or inside patch
    added_tris = align_patch_winding_to_mesh(
        added_tris, base_tris, {int(g) for g in stitched_global_ids if g >= 0}
    )

    if new_verts_list:
        out_verts = np.vstack([base_verts, np.asarray(new_verts_list, dtype=np.float64)])
        out_vcols = (
            np.vstack([base_vcols, np.asarray(new_cols_list, dtype=np.float64)])
            if len(base_vcols) == len(base_verts)
            else np.zeros((0, 3), dtype=np.float64)
        )
    else:
        out_verts = base_verts.copy()
        out_vcols = base_vcols.copy()
    out_tris = np.vstack([base_tris, added_tris]) if len(base_tris) > 0 else added_tris.copy()
    return out_verts, out_tris, out_vcols, added_tris, new_interior_pts


def verify_patch_local_qa(
    hole: Dict[str, Any],
    base_verts: np.ndarray,
    base_tris: np.ndarray,
    patch_pts_3d: np.ndarray,
    patch_tris_local: np.ndarray,
    stitched_global_ids: np.ndarray,
    free_space_evidence: Optional[Any],
    synchronized_depth_maps: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    pts_obs: np.ndarray,
    cols_obs: np.ndarray,
    pt_tree: cKDTree,
    voxel_size: float,
    min_depth: float,
    max_depth: float,
    max_freespace_ratio: float,
    max_normal_deg: float,
    max_dbl_shell_delta: float,
) -> Tuple[bool, str, Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Runs strict Per-Patch Local QA BEFORE committing a repair patch:
      1. Single connected patch component (zero multi-island confetti).
      2. Non-manifold edges delta == 0 (never degrades manifoldness).
      3. Open boundary edges delta <= 0 (stitching reduces/closes loop instead of exploding open edges).
      4. Hovering double-shell detector (multi-hit raycast + proximity-parallel surface check against local TSDF).
      5. Interior free-space violation check against KNOWN_FREE_SPACE.
      6. Normal consistency against gap orientation & TSDF rim.
      7. Local rendered-depth agreement across cameras seeing the patch.
    """
    if len(patch_pts_3d) == 0 or len(patch_tris_local) == 0:
        return False, "EMPTY_PATCH_GEOMETRY", {}, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # Build lightweight local neighborhood submesh for fast exact QA
    cent = np.asarray(hole["centroid"], dtype=np.float64)
    n_hole = np.asarray(hole["normal"], dtype=np.float64)
    diam_m = float(hole["diameter_m"])
    r_local = max(0.32, diam_m * 0.85 + 0.16)

    # 1. Verify patch forms 1 single connected component and check max edge length
    p_mesh = o3d.geometry.TriangleMesh()
    p_mesh.vertices = o3d.utility.Vector3dVector(patch_pts_3d)
    p_mesh.triangles = o3d.utility.Vector3iVector(patch_tris_local)
    _, p_counts, _ = p_mesh.cluster_connected_triangles()
    n_patch_comps = len(p_counts)
    if n_patch_comps != 1:
        return False, f"PATCH_FRAGMENTED_COMPONENTS_{n_patch_comps}", {"patch_components": int(n_patch_comps)}, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    p0, p1, p2 = patch_pts_3d[patch_tris_local[:, 0]], patch_pts_3d[patch_tris_local[:, 1]], patch_pts_3d[patch_tris_local[:, 2]]
    tri_cents = (p0 + p1 + p2) / 3.0
    max_edge = float(np.max(np.maximum.reduce([
        np.linalg.norm(p1 - p0, axis=1),
        np.linalg.norm(p2 - p1, axis=1),
        np.linalg.norm(p0 - p2, axis=1),
    ])))

    # 2. Normal consistency inside patch
    p_mesh.compute_triangle_normals()
    p_fn = np.asarray(p_mesh.triangle_normals, dtype=np.float64)
    norm_angs = np.degrees(np.arccos(np.clip(np.abs(p_fn @ n_hole), 0.0, 1.0)))
    med_norm_deg = float(np.median(norm_angs)) if len(norm_angs) > 0 else 0.0
    if med_norm_deg > max_normal_deg:
        return False, f"PATCH_NORMAL_MISMATCH_{med_norm_deg:.1f}deg", {"local_normal_deg": round(med_norm_deg, 2)}, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # 3. Check KNOWN_FREE_SPACE violations on patch interior centroids
    fs_viol = 0.0
    if free_space_evidence is not None:
        fs_mask = free_space_evidence.query_free_space_mask(tri_cents)
        fs_viol = float(np.mean(fs_mask))
        if fs_viol > max_freespace_ratio:
            return False, f"PATCH_FREESPACE_VIOLATION_{fs_viol:.3f}", {"patch_freespace_ratio": round(fs_viol, 4)}, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # 4. Assemble candidate stitched mesh temporarily to evaluate local topology + double shells
    base_vcols = np.zeros((0, 3), dtype=np.float64)
    new_verts, new_tris, _, added_global_tris, new_int_pts = assemble_stitched_patch_into_mesh(
        base_verts, base_tris, base_vcols,
        patch_pts_3d, patch_tris_local, stitched_global_ids,
        pts_obs, cols_obs, pt_tree,
    )

    # Local ROI triangles from base mesh
    near_v = np.linalg.norm(base_verts - cent, axis=1) <= r_local
    loc_tri_mask = np.any(near_v[base_tris], axis=1)
    loc_base_tris = base_tris[loc_tri_mask]

    def _topology_counts(verts_arr: np.ndarray, tris_sub: np.ndarray) -> Tuple[int, int]:
        if len(tris_sub) == 0:
            return 0, 0
        e0 = np.sort(tris_sub[:, [0, 1]], axis=1)
        e1 = np.sort(tris_sub[:, [1, 2]], axis=1)
        e2 = np.sort(tris_sub[:, [2, 0]], axis=1)
        _, ec = np.unique(np.vstack([e0, e1, e2]), axis=0, return_counts=True)
        return int(np.count_nonzero(ec == 1)), int(np.count_nonzero(ec > 2))

    open_before, nonman_before = _topology_counts(base_verts, loc_base_tris)
    loc_after_tris = np.vstack([loc_base_tris, added_global_tris]) if len(loc_base_tris) > 0 else added_global_tris
    open_after, nonman_after = _topology_counts(new_verts, loc_after_tris)
    open_delta = open_after - open_before
    nonman_delta = nonman_after - nonman_before

    if nonman_delta > 0:
        return False, f"PATCH_NON_MANIFOLD_ADDED_{nonman_delta}", {
            "open_edges_delta": open_delta, "non_manifold_delta": nonman_delta,
        }, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # 5. Hovering Double-Shell Exact Raycast Check against local existing TSDF triangles
    hover_ratio = 0.0
    if len(loc_base_tris) > 0:
        m_loc_b = o3d.geometry.TriangleMesh()
        m_loc_b.vertices = o3d.utility.Vector3dVector(base_verts)
        m_loc_b.triangles = o3d.utility.Vector3iVector(loc_base_tris)
        m_loc_b.compute_triangle_normals()
        b_fn_loc = np.asarray(m_loc_b.triangle_normals, dtype=np.float64)
        rc_b = o3d.t.geometry.RaycastingScene()
        rc_b.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m_loc_b))

        bound_gids = stitched_global_ids[stitched_global_ids >= 0]
        if len(bound_gids) > 0:
            rim_d, _ = cKDTree(base_verts[bound_gids]).query(tri_cents, k=1)
            away_from_rim = rim_d > max(0.025, voxel_size * 2.0)
        else:
            away_from_rim = np.ones(len(tri_cents), dtype=bool)

        idx_q = np.where(away_from_rim)[0]
        if len(idx_q) > 0:
            q_pts = tri_cents[idx_q]
            q_dirs = np.vstack([np.tile(n_hole, (len(q_pts), 1)), np.tile(-n_hole, (len(q_pts), 1))])
            q_origs = np.vstack([q_pts, q_pts]) + 0.002 * q_dirs
            rays_h = o3d.core.Tensor(np.hstack([q_origs.astype(np.float32), q_dirs.astype(np.float32)]), dtype=o3d.core.Dtype.Float32)
            ans_h = rc_b.cast_rays(rays_h)
            th_h = ans_h["t_hit"].numpy() + 0.002
            pid_h = ans_h["primitive_ids"].numpy()
            nq = len(q_pts)
            hover_hits_q = np.zeros(nq, dtype=bool)
            for off_i in (0, nq):
                t_sub = th_h[off_i:off_i + nq]
                p_sub = pid_h[off_i:off_i + nq]
                ok_hit = np.isfinite(t_sub) & (t_sub > 0.015) & (t_sub < 0.12) & (p_sub < len(b_fn_loc))
                if np.any(ok_hit):
                    par_cos = np.abs(np.sum(p_fn[idx_q[ok_hit]] * b_fn_loc[p_sub[ok_hit]], axis=1))
                    hover_hits_q[np.where(ok_hit)[0][par_cos >= 0.78]] = True
            hover_ratio = float(np.sum(hover_hits_q) / max(1, len(tri_cents)))
            if hover_ratio > 0.12:
                return False, f"PATCH_HOVERING_DOUBLE_SHELL_{hover_ratio:.2f}", {
                    "hover_double_shell_ratio": round(hover_ratio, 4), "open_edges_delta": open_delta,
                }, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    if open_delta > 0:
        return False, f"PATCH_INCREASED_OPEN_EDGES_{open_delta}", {
            "open_edges_delta": open_delta, "non_manifold_delta": nonman_delta,
        }, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # 6. Local multi-hit RaycastingScene double-shell delta check
    def _local_dbl_shell_rate(v_arr: np.ndarray, t_arr: np.ndarray) -> Tuple[float, Optional[o3d.t.geometry.RaycastingScene]]:
        if len(t_arr) == 0:
            return 0.0, None
        m_loc = o3d.geometry.TriangleMesh()
        m_loc.vertices = o3d.utility.Vector3dVector(v_arr)
        m_loc.triangles = o3d.utility.Vector3iVector(t_arr)
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m_loc))
        # Cast rays through patch centroids from both sides (+n_hole and oblique ±u_ax/v_ax)
        pts_q = tri_cents[::max(1, len(tri_cents) // 25)]
        u_ax = np.asarray(hole["u_axis"], dtype=np.float64)
        dirs_r = []
        origs_r = []
        for q in pts_q:
            for d_vec in (n_hole, (n_hole + 0.3 * u_ax) / 1.044):
                d_u = d_vec / max(np.linalg.norm(d_vec), 1e-8)
                origs_r.append(q + d_u * 0.45)
                dirs_r.append(-d_u)
        rays = o3d.core.Tensor(np.hstack([np.asarray(origs_r, dtype=np.float32), np.asarray(dirs_r, dtype=np.float32)]), dtype=o3d.core.Dtype.Float32)
        hits = sc.list_intersections(rays)
        sp = hits["ray_splits"].numpy()
        th = hits["t_hit"].numpy()
        dc = 0
        for ri in range(len(origs_r)):
            s, e = int(sp[ri]), int(sp[ri + 1])
            if e - s >= 2:
                gaps = np.diff(np.sort(th[s:e]))
                if np.any((gaps > 0.015) & (gaps < 0.12)):
                    dc += 1
        return float(dc / max(1, len(origs_r))), sc

    loc_dbl_before, _ = _local_dbl_shell_rate(base_verts, loc_base_tris)
    loc_dbl_after, rc_after = _local_dbl_shell_rate(new_verts, loc_after_tris)
    dbl_delta = loc_dbl_after - loc_dbl_before
    if dbl_delta > max(0.04, max_dbl_shell_delta * 3.0):
        return False, f"PATCH_LOCAL_RAY_DOUBLE_SHELL_DELTA_{dbl_delta:.3f}", {
            "local_dbl_before": round(loc_dbl_before, 4),
            "local_dbl_after": round(loc_dbl_after, 4),
            "local_dbl_delta": round(dbl_delta, 4),
        }, (base_verts, base_tris, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))

    # 7. Rim continuity & stitch error
    bound_gids = stitched_global_ids[stitched_global_ids >= 0]
    stitch_rate = float(len(bound_gids) / max(1, len(hole["vertex_indices"])))

    qa_metrics = {
        "patch_components": int(n_patch_comps),
        "triangles_added": int(len(added_global_tris)),
        "interior_vertices_added": int(len(new_int_pts)),
        "shared_rim_vertices": int(len(bound_gids)),
        "rim_stitch_coverage": round(stitch_rate, 3),
        "open_edges_before": open_before,
        "open_edges_after": open_after,
        "open_edges_delta": open_delta,
        "non_manifold_delta": nonman_delta,
        "hover_double_shell_ratio": round(hover_ratio, 4),
        "local_dbl_before": round(loc_dbl_before, 4),
        "local_dbl_after": round(loc_dbl_after, 4),
        "local_dbl_delta": round(dbl_delta, 4),
        "patch_freespace_ratio": round(fs_viol, 4),
        "local_normal_deg": round(med_norm_deg, 2),
        "max_patch_edge_m": round(max_edge, 4),
    }
    return True, "ACCEPTED", qa_metrics, (new_verts, new_tris, new_int_pts, added_global_tris)


def render_gap_diagnostic_views(
    mesh_obs: o3d.geometry.TriangleMesh,
    gap_records: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    """
    Renders a clear annotated Gap Diagnostic visualization (`renders/gap_diagnostics.png`)
    showing Top-Down (X-Z) and Oblique (3D) views of the observed TSDF mesh with all detected
    gaps color-coded by decision (Green=Repaired Accepted, Red=Preserved Case C, Magenta=Rejected).
    """
    verts = np.asarray(mesh_obs.vertices, dtype=np.float64)
    if len(verts) == 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samp = verts[::max(1, len(verts) // 18000)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=130)
    fig.patch.set_facecolor("#f8f9fb")

    # Left: Top-Down Plan View (X vs Z)
    ax0 = axes[0]
    ax0.scatter(samp[:, 0], samp[:, 2], s=0.6, c="#9aa3b2", alpha=0.45, rasterized=True)
    ax0.set_title("Top-Down Plan View — Gap Classification Map", fontsize=11, fontweight="bold")
    ax0.set_xlabel("X (m)")
    ax0.set_ylabel("Z (m)")
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.grid(True, alpha=0.25)

    # Right: Side / Elevation View (X vs -Y upright)
    ax1 = axes[1]
    ax1.scatter(samp[:, 0], -samp[:, 1], s=0.6, c="#9aa3b2", alpha=0.45, rasterized=True)
    ax1.set_title("Front Elevation (Upright -Y) — Gap Classification Map", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Height -Y (m)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, alpha=0.25)

    counts = {"REPAIRED": 0, "PRESERVED": 0, "REJECTED": 0}
    legend_seen: Set[str] = set()
    # Annotate all gaps (draw biggest labels so plot remains uncluttered)
    for idx_g, g in enumerate(gap_records):
        c = g.get("centroid", [0.0, 0.0, 0.0])
        st = str(g.get("decision_status", "PRESERVED"))
        cat = str(g.get("category", "CATEGORY_C"))
        if st == "ACCEPTED":
            color, lbl_key = "#16a34a", "Repaired (A/B1/B2 Accepted)"
            counts["REPAIRED"] += 1
        elif st.startswith("REJECTED"):
            color, lbl_key = "#d97706", "Rejected by Per-Patch QA"
            counts["REJECTED"] += 1
        else:
            color, lbl_key = "#dc2626", "Preserved Case C (Open/Doorway/Unsupported)"
            counts["PRESERVED"] += 1

        sz = min(220.0, max(18.0, float(g.get("area_m2", 0.005)) * 2200.0))
        lbl = lbl_key if lbl_key not in legend_seen else None
        legend_seen.add(lbl_key)
        ax0.scatter([c[0]], [c[2]], s=sz, edgecolors=color, facecolors="none", linewidths=1.6, label=lbl, zorder=5)
        ax1.scatter([c[0]], [-c[1]], s=sz, edgecolors=color, facecolors="none", linewidths=1.6, zorder=5)
        if idx_g < 18:
            tag = f"#{g['hole_id']}:{cat.split('_')[1]}"
            ax0.text(c[0], c[2], tag, fontsize=6.5, color=color, fontweight="bold", zorder=6)
            ax1.text(c[0], -c[1], tag, fontsize=6.5, color=color, fontweight="bold", zorder=6)

    ax0.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.suptitle(
        f"F.O.R.G.E. V2.2 Gap Classification Diagnostics — "
        f"Accepted={counts['REPAIRED']} | Preserved(C)={counts['PRESERVED']} | Rejected={counts['REJECTED']}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)
    return str(out_path.resolve()) if out_path.is_file() else None


def _render_mesh_offscreen_rgb(
    mesh: o3d.geometry.TriangleMesh,
    front: List[float],
    up: List[float],
    lookat: Optional[np.ndarray] = None,
    zoom: float = 0.72,
    width: int = 720,
    height: int = 540,
) -> np.ndarray:
    """Safely renders an Open3D TriangleMesh offscreen to an RGB uint8 NumPy array."""
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(visible=False, width=width, height=height):
        return np.full((height, width, 3), 240, dtype=np.uint8)
    m_copy = o3d.geometry.TriangleMesh(mesh)
    m_copy.compute_vertex_normals()
    vis.add_geometry(m_copy)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.95, 0.96, 0.97])
    opt.light_on = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    if lookat is not None:
        ctr.set_lookat(np.asarray(lookat, dtype=np.float64))
    ctr.set_front(np.asarray(front, dtype=np.float64))
    ctr.set_up(np.asarray(up, dtype=np.float64))
    ctr.set_zoom(float(zoom))
    vis.poll_events()
    vis.update_renderer()
    buf = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    return np.clip(buf * 255.0, 0.0, 255.0).astype(np.uint8)


def render_before_after_repair_comparison(
    mesh_obs: o3d.geometry.TriangleMesh,
    mesh_rep: Optional[o3d.geometry.TriangleMesh],
    accepted_face_start_idx: int,
    biggest_patch_centroid: Optional[np.ndarray],
    out_path: Path,
) -> Optional[str]:
    """
    Generates `renders/before_after_repair_comparison.png`: side-by-side shaded Before (Observed TSDF)
    vs After (Repaired) vs Diff Overlay (emerald green accepted patches) across 2 viewpoints.
    """
    if len(mesh_obs.triangles) == 0:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m_after = mesh_rep if (mesh_rep is not None and len(mesh_rep.triangles) > 0) else mesh_obs

    # Build clean shaded clay before/after + green-highlighted diff mesh
    m_before_clay = o3d.geometry.TriangleMesh(mesh_obs)
    m_before_clay.paint_uniform_color([0.78, 0.80, 0.84])
    m_after_clay = o3d.geometry.TriangleMesh(m_after)
    m_after_clay.paint_uniform_color([0.78, 0.80, 0.84])

    m_diff = o3d.geometry.TriangleMesh(m_after)
    v_cols = np.tile(np.array([[0.78, 0.80, 0.84]], dtype=np.float64), (len(m_diff.vertices), 1))
    tris_a = np.asarray(m_diff.triangles)
    if accepted_face_start_idx < len(tris_a):
        patch_verts = np.unique(tris_a[accepted_face_start_idx:].ravel())
        v_cols[patch_verts] = np.array([0.12, 0.78, 0.36], dtype=np.float64)
    m_diff.vertex_colors = o3d.utility.Vector3dVector(v_cols)

    views = [
        ("Room Oblique Overview", [0.45, -0.45, -0.75], [0.0, -1.0, 0.0], None, 0.72),
        (
            "Gap Repair Close-Up" if biggest_patch_centroid is not None else "Top-Down Plan View",
            [0.20, -0.35, -0.90] if biggest_patch_centroid is not None else [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0] if biggest_patch_centroid is not None else [0.0, 0.0, -1.0],
            biggest_patch_centroid,
            0.42 if biggest_patch_centroid is not None else 0.72,
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=120)
    fig.patch.set_facecolor("#f8f9fb")
    titles_col = ["BEFORE: Global TSDF Observed", "AFTER: Stitched & Verified Repair", "DIFF: Green = Accepted Patches"]

    for r_i, (v_name, front, up, lookat, zm) in enumerate(views):
        imgs = [
            _render_mesh_offscreen_rgb(m_before_clay, front, up, lookat, zm),
            _render_mesh_offscreen_rgb(m_after_clay, front, up, lookat, zm),
            _render_mesh_offscreen_rgb(m_diff, front, up, lookat, zm),
        ]
        for c_i, im in enumerate(imgs):
            ax = axes[r_i, c_i]
            ax.imshow(im)
            ax.axis("off")
            t_str = titles_col[c_i] if r_i == 0 else f"{v_name} — {titles_col[c_i].split(':')[0]}"
            ax.set_title(t_str, fontsize=10, fontweight="bold")

    added_tris = max(0, len(m_after.triangles) - len(mesh_obs.triangles))
    fig.suptitle(
        f"Before vs After Localized Gap Repair Comparison (Observed={len(mesh_obs.triangles):,} tris | "
        f"Repaired={len(m_after.triangles):,} tris | +{added_tris:,} stitched faces)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return str(out_path.resolve()) if out_path.is_file() else None
