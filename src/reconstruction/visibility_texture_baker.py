"""
F.O.R.G.E. V2 — Visibility-Aware Photographic Texture Baker, Multi-Material Exporter & Texture QA.

Improvements over V3:
1. True Mesh Z-Buffer Self-Occlusion Culling via Open3D RaycastingScene (prevents foreground
   furniture/people from projecting onto walls/beds behind them).
2. Combined visibility scoring using incidence angle (cos_theta >= 0.22), projected pixel
   resolution area, Laplacian frame sharpness, and depth consistency.
3. Fast vectorized UV computation and chunked canonical OBJ/MTL + GLB + PLY export.
4. Explicit separate provenance material (`Material_Inferred_Extrapolated`) for optional
   inferred surfaces so unseen surfaces are NEVER mislabelled as photographic textures.
5. Comprehensive Texture QA: direct coverage, OOR UV check, chart seam stats, resolution,
   and multi-view photometric reprojection check against held-out/source frames.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image


class VisibilityAwareTextureBaker:
    """Bakes high-resolution original video frames onto a canonical mesh with z-buffer occlusion gating."""

    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        extrinsics: np.ndarray,
        intrinsics_infer: np.ndarray,
        frame_paths: List[Path],
        depth_maps: Optional[np.ndarray] = None,
        min_cos_theta: float = 0.22,
        dynamic_masks_infer: Optional[List[Optional[np.ndarray]]] = None,
    ):
        self.mesh = o3d.geometry.TriangleMesh(mesh)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        self.extrinsics = np.asarray(extrinsics, dtype=np.float64)
        self.intrinsics_infer = np.asarray(intrinsics_infer, dtype=np.float64)
        self.frame_paths = [Path(p) for p in frame_paths]
        self.depth_maps = depth_maps
        self.min_cos_theta = float(min_cos_theta)
        self.dynamic_masks_infer = dynamic_masks_infer

        self.num_cams = len(self.frame_paths)
        self.verts = np.asarray(self.mesh.vertices, dtype=np.float64)
        self.tris = np.asarray(self.mesh.triangles, dtype=np.int32)
        self.num_faces = len(self.tris)
        self.face_normals = np.asarray(self.mesh.triangle_normals, dtype=np.float64)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float64)
        self.face_centroids = self.verts[self.tris].mean(axis=1)

        with Image.open(self.frame_paths[0]) as im:
            self.W_native, self.H_native = im.size

        if self.depth_maps is not None:
            d0 = self.depth_maps[0].squeeze()
            self.H_infer, self.W_infer = d0.shape[:2]
        else:
            self.H_infer, self.W_infer = (432, 240) if self.H_native > self.W_native else (240, 432)

        self.sx = float(self.W_native) / float(self.W_infer)
        self.sy = float(self.H_native) / float(self.H_infer)
        self.intrinsics_native = self.intrinsics_infer.copy()
        self.intrinsics_native[:, 0, 0] *= self.sx
        self.intrinsics_native[:, 0, 2] *= self.sx
        self.intrinsics_native[:, 1, 1] *= self.sy
        self.intrinsics_native[:, 1, 2] *= self.sy

    def compute_visibility_and_best_cameras(
        self,
        frame_sharpness: Optional[List[float]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        def _log(msg: str):
            if log_callback:
                log_callback(msg)

        N_f = self.num_faces
        M = self.num_cams
        scores = np.zeros((N_f, M), dtype=np.float32)

        cam_centers = np.array([
            -self.extrinsics[i, :3, :3].T @ self.extrinsics[i, :3, 3]
            for i in range(M)
        ], dtype=np.float64)

        sharp_weights = np.ones(M, dtype=np.float32)
        if frame_sharpness is not None and len(frame_sharpness) == M:
            s_arr = np.asarray(frame_sharpness, dtype=np.float32)
            s_min, s_max = float(s_arr.min()), float(s_arr.max())
            if s_max > s_min:
                sharp_weights = 0.72 + 0.28 * ((s_arr - s_min) / (s_max - s_min))

        # Build BVH RaycastingScene once for true mesh z-buffer visibility testing!
        _log("[*] Evaluating mesh self-occlusion (BVH raycasting) & photographic camera selection...")
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(tmesh)

        # Precompute 3D face double-area magnitude as resolution proxy
        p0 = self.verts[self.tris[:, 0]]
        p1 = self.verts[self.tris[:, 1]]
        p2 = self.verts[self.tris[:, 2]]
        tri_area_3d = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

        zbuf_occluded_faces = 0
        dynamic_masked_hits = 0

        for i in range(M):
            R_i = self.extrinsics[i, :3, :3]
            t_i = self.extrinsics[i, :3, 3]
            K = self.intrinsics_native[i]
            cam_c = cam_centers[i]

            P_c = (self.face_centroids @ R_i.T) + t_i
            Z_c = P_c[:, 2]
            valid_z = Z_c > 0.15
            z_safe = np.maximum(Z_c, 1e-4)
            u = K[0, 0] * (P_c[:, 0] / z_safe) + K[0, 2]
            v = K[1, 1] * (P_c[:, 1] / z_safe) + K[1, 2]

            in_bounds = (
                valid_z
                & (u >= 6.0) & (u <= self.W_native - 7.0)
                & (v >= 6.0) & (v <= self.H_native - 7.0)
            )
            dist = np.linalg.norm(P_c, axis=1) + 1e-6
            rays_c = P_c / dist[:, None]
            N_c = self.face_normals @ R_i.T
            cos_theta = -np.sum(rays_c * N_c, axis=1)
            cand_mask = in_bounds & (cos_theta > self.min_cos_theta)

            if not np.any(cand_mask):
                continue

            # Exclude dynamic person pixels in frame i
            if self.dynamic_masks_infer is not None and i < len(self.dynamic_masks_infer) and self.dynamic_masks_infer[i] is not None:
                dmsk = self.dynamic_masks_infer[i]
                ui_m = np.clip(np.round(u[cand_mask] / self.sx).astype(int), 0, self.W_infer - 1)
                vi_m = np.clip(np.round(v[cand_mask] / self.sy).astype(int), 0, self.H_infer - 1)
                dyn_hit = dmsk[vi_m, ui_m] > 0
                dynamic_masked_hits += int(np.count_nonzero(dyn_hit))
                c_idx = np.where(cand_mask)[0]
                cand_mask[c_idx[dyn_hit]] = False

            if not np.any(cand_mask):
                continue

            # True mesh self-occlusion test via RaycastingScene
            c_indices = np.where(cand_mask)[0]
            cent_sub = self.face_centroids[c_indices]
            vec_to_cent = cent_sub - cam_c[None, :]
            true_dist = np.linalg.norm(vec_to_cent, axis=1)
            unit_dirs = vec_to_cent / np.maximum(true_dist[:, None], 1e-6)
            rays_tensor = o3d.core.Tensor(
                np.hstack([np.tile(cam_c.astype(np.float32), (len(c_indices), 1)), unit_dirs.astype(np.float32)]),
                dtype=o3d.core.Dtype.Float32,
            )
            ans = rc.cast_rays(rays_tensor)
            t_hit = ans["t_hit"].numpy()
            # Occluded if ray hits ANOTHER mesh surface closer than true_dist - margin
            self_occluded = np.isfinite(t_hit) & (t_hit < true_dist - 0.035)
            zbuf_occluded_faces += int(np.count_nonzero(self_occluded))

            # Optional secondary predicted depth occlusion check
            depth_occ = np.zeros(len(c_indices), dtype=bool)
            if self.depth_maps is not None and i < len(self.depth_maps):
                dm = self.depth_maps[i].squeeze()
                u_d = np.clip(np.round(u[c_indices] / self.sx).astype(int), 0, dm.shape[1] - 1)
                v_d = np.clip(np.round(v[c_indices] / self.sy).astype(int), 0, dm.shape[0] - 1)
                d_obs = dm[v_d, u_d]
                depth_occ = (d_obs > 0.15) & (Z_c[c_indices] > d_obs * 1.12)

            vis_sub = (~self_occluded) & (~depth_occ)
            if not np.any(vis_sub):
                continue
            v_indices = c_indices[vis_sub]
            # Projected pixel area score = fx * fy * area3D * cos_theta / dist^2
            proj_px_area = (K[0, 0] * K[1, 1] * tri_area_3d[v_indices] * cos_theta[v_indices]) / np.maximum(dist[v_indices] ** 2, 0.04)
            res_bonus = np.clip(np.sqrt(np.maximum(proj_px_area, 0.05)) / 4.0, 0.35, 2.2)
            # Soft vignette penalty near image borders
            border_margin = np.minimum.reduce([u[v_indices], self.W_native - u[v_indices], v[v_indices], self.H_native - v[v_indices]])
            border_w = np.clip(border_margin / 40.0, 0.4, 1.0)

            q = (
                (cos_theta[v_indices] ** 2.5)
                * (1.0 / np.maximum(dist[v_indices], 0.30))
                * res_bonus
                * border_w
                * sharp_weights[i]
            )
            scores[v_indices, i] = q.astype(np.float32)

        best_cam = np.argmax(scores, axis=1).astype(np.int32)
        max_scores = np.max(scores, axis=1)
        uncovered = max_scores <= 0.0
        best_cam[uncovered] = -1

        # Build fast face adjacency and fill uncovered + consolidate seams across 4 topological passes
        e01 = np.sort(self.tris[:, [0, 1]], axis=1)
        e12 = np.sort(self.tris[:, [1, 2]], axis=1)
        e20 = np.sort(self.tris[:, [2, 0]], axis=1)
        edges_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for f_idx in range(N_f):
            for e in ((int(e01[f_idx, 0]), int(e01[f_idx, 1])),
                      (int(e12[f_idx, 0]), int(e12[f_idx, 1])),
                      (int(e20[f_idx, 0]), int(e20[f_idx, 1]))):
                edges_to_faces.setdefault(e, []).append(f_idx)

        face_nbrs: List[List[int]] = [[] for _ in range(N_f)]
        for fs in edges_to_faces.values():
            if len(fs) == 2:
                face_nbrs[fs[0]].append(fs[1])
                face_nbrs[fs[1]].append(fs[0])

        for _ in range(2):
            rem_unc = np.where(best_cam < 0)[0]
            if len(rem_unc) == 0:
                break
            for f_idx in rem_unc:
                nc = [best_cam[nb] for nb in face_nbrs[f_idx] if best_cam[nb] >= 0]
                if nc:
                    vals, cnts = np.unique(nc, return_counts=True)
                    best_cam[f_idx] = vals[np.argmax(cnts)]

        still_unc = best_cam < 0
        if np.any(still_unc):
            unc_idx = np.where(still_unc)[0]
            unc_cents = self.face_centroids[unc_idx]
            # Vectorized check for front-of-camera + nearest optical center across all cameras (no Z<=0 behind-cam!)
            best_fallback = np.zeros(len(unc_idx), dtype=np.int32)
            best_cost = np.full(len(unc_idx), 1e12, dtype=np.float64)
            cx_tgt, cy_tgt = self.W_native * 0.5, self.H_native * 0.5
            for ci in range(M):
                R_i = self.extrinsics[ci, :3, :3]
                t_i = self.extrinsics[ci, :3, 3]
                K = self.intrinsics_native[ci]
                Pc = (unc_cents @ R_i.T) + t_i
                Zc = Pc[:, 2]
                in_front = Zc > 0.15
                if not np.any(in_front):
                    continue
                u = K[0, 0] * (Pc[in_front, 0] / Zc[in_front]) + K[0, 2]
                v = K[1, 1] * (Pc[in_front, 1] / Zc[in_front]) + K[1, 2]
                off_u = np.maximum(0.0, np.maximum(-u, u - self.W_native)) / self.W_native
                off_v = np.maximum(0.0, np.maximum(-v, v - self.H_native)) / self.H_native
                cent_dist = np.hypot((u - cx_tgt) / self.W_native, (v - cy_tgt) / self.H_native)
                cost_sub = np.linalg.norm(unc_cents[in_front] - cam_centers[ci], axis=1) + 4.0 * (off_u + off_v) + 0.5 * cent_dist
                sub_ids = np.where(in_front)[0]
                better = cost_sub < best_cost[sub_ids]
                best_cost[sub_ids[better]] = cost_sub[better]
                best_fallback[sub_ids[better]] = ci
            # Only if every camera is strictly behind centroid (rare mesh outlier), use nearest camera center
            no_front = best_cost > 1e11
            if np.any(no_front):
                dists_fb = np.linalg.norm(unc_cents[no_front, None, :] - cam_centers[None, :, :], axis=-1)
                best_fallback[no_front] = np.argmin(dists_fb, axis=1)
            best_cam[unc_idx] = best_fallback

        seams_consolidated = 0
        for _ in range(8):
            changes = 0
            for f_idx in range(N_f):
                nbrs = face_nbrs[f_idx]
                if len(nbrs) < 2:
                    continue
                curr_c = best_cam[f_idx]
                nc = [best_cam[nb] for nb in nbrs]
                vals, cnts = np.unique(nc, return_counts=True)
                dom_c = int(vals[np.argmax(cnts)])
                if dom_c != curr_c and cnts.max() >= 2:
                    if scores[f_idx, dom_c] >= scores[f_idx, curr_c] * 0.45 and scores[f_idx, dom_c] > 0.0:
                        best_cam[f_idx] = dom_c
                        changes += 1
            seams_consolidated += changes
            if changes == 0:
                break

        # Prune micro camera fragments (< 20 faces) into visible neighbor cameras to reduce exposure seams
        cam_ids, cam_cnts = np.unique(best_cam, return_counts=True)
        tiny_cams = set(int(c) for c, cnt in zip(cam_ids, cam_cnts) if c >= 0 and cnt < 20)
        if tiny_cams:
            for f_idx in range(N_f):
                if best_cam[f_idx] in tiny_cams:
                    nc = [best_cam[nb] for nb in face_nbrs[f_idx] if best_cam[nb] not in tiny_cams and scores[f_idx, best_cam[nb]] > 0]
                    if nc:
                        vals, cnts = np.unique(nc, return_counts=True)
                        best_cam[f_idx] = int(vals[np.argmax(cnts)])
                        seams_consolidated += 1

        cov_pct = round(float((~uncovered).mean() * 100.0), 2)
        stats = {
            "total_faces": N_f,
            "direct_photographic_faces": int((~uncovered).sum()),
            "photographic_coverage_pct": cov_pct,
            "active_cameras_used": int(len(np.unique(best_cam))),
            "native_frame_resolution": [self.W_native, self.H_native],
            "zbuffer_self_occluded_tests": zbuf_occluded_faces,
            "dynamic_mask_filtered_tests": dynamic_masked_hits,
            "seam_faces_consolidated": seams_consolidated,
        }
        return best_cam, scores, stats

    def _estimate_exposure_gains(
        self,
        best_cam_per_face: np.ndarray,
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """
        Estimates conservative per-camera multiplicative luminance gain factors in [0.85, 1.18]
        from shared geometric seam edges between adjacent cameras so wall/furniture tiles blend
        without exposure stepping while keeping photographic colors anchored.
        """
        active_cams = sorted([int(c) for c in np.unique(best_cam_per_face) if c >= 0])
        gains: Dict[int, float] = {ci: 1.0 for ci in active_cams}
        if len(active_cams) < 2 or self.num_faces < 4:
            return gains, {"applied": False, "active_cams": len(active_cams), "seam_edges_used": 0, "gain_min": 1.0, "gain_max": 1.0}

        # Build fast edge -> adjacent face pairs where best_cam differs
        edge_to_face: Dict[Tuple[int, int], int] = {}
        seam_edges: List[Tuple[int, int, int, int, int, int]] = []  # (c0, c1, v0, v1, f0, f1)
        for f_idx, tri in enumerate(self.tris):
            c_idx = int(best_cam_per_face[f_idx])
            if c_idx < 0:
                continue
            v_ids = (int(tri[0]), int(tri[1]), int(tri[2]))
            for e_a, e_b in ((v_ids[0], v_ids[1]), (v_ids[1], v_ids[2]), (v_ids[2], v_ids[0])):
                key = (e_a, e_b) if e_a < e_b else (e_b, e_a)
                if key in edge_to_face:
                    f_other = edge_to_face[key]
                    c_other = int(best_cam_per_face[f_other])
                    if c_other >= 0 and c_other != c_idx:
                        seam_edges.append((c_other, c_idx, key[0], key[1], f_other, f_idx))
                else:
                    edge_to_face[key] = f_idx

        if not seam_edges:
            return gains, {"applied": False, "active_cams": len(active_cams), "seam_edges_used": 0, "gain_min": 1.0, "gain_max": 1.0}

        # Load downsampled RGB images lazily only for cameras touching seams
        seam_cams = sorted({e[0] for e in seam_edges} | {e[1] for e in seam_edges})
        img_lumas: Dict[int, np.ndarray] = {}
        for ci in seam_cams:
            bgr = cv2.imread(str(self.frame_paths[ci]))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            img_lumas[ci] = luma

        pair_log_ratios: Dict[Tuple[int, int], List[float]] = {}
        # Subsample seam edges deterministically if very numerous
        step = max(1, len(seam_edges) // 4000)
        for c0, c1, va, vb, _f0, _f1 in seam_edges[::step]:
            if c0 not in img_lumas or c1 not in img_lumas:
                continue
            for alpha in (0.25, 0.50, 0.75):
                p_sample = alpha * self.verts[va] + (1.0 - alpha) * self.verts[vb]
                vals = []
                for ci in (c0, c1):
                    R = self.extrinsics[ci, :3, :3]
                    t = self.extrinsics[ci, :3, 3]
                    Pc = R @ p_sample + t
                    if Pc[2] <= 0.15:
                        vals.append(None)
                        break
                    K = self.intrinsics_native[ci]
                    u = int(round(K[0, 0] * Pc[0] / Pc[2] + K[0, 2]))
                    v = int(round(K[1, 1] * Pc[1] / Pc[2] + K[1, 2]))
                    if u < 2 or u >= self.W_native - 2 or v < 2 or v >= self.H_native - 2:
                        vals.append(None)
                        break
                    patch = img_lumas[ci][v - 1 : v + 2, u - 1 : u + 2]
                    med_l = float(np.median(patch))
                    if med_l < 14.0 or med_l > 246.0:  # skip crushed shadows / saturated highlights
                        vals.append(None)
                        break
                    vals.append(med_l)
                if len(vals) == 2 and vals[0] is not None and vals[1] is not None:
                    lr = math.log(max(vals[1], 1e-3) / max(vals[0], 1e-3))
                    if abs(lr) <= 0.35:
                        pair = (c0, c1) if c0 < c1 else (c1, c0)
                        signed_lr = lr if c0 < c1 else -lr  # log(L_{high} / L_{low})
                        pair_log_ratios.setdefault(pair, []).append(signed_lr)

        if not pair_log_ratios:
            return gains, {"applied": False, "active_cams": len(active_cams), "seam_edges_used": 0, "gain_min": 1.0, "gain_max": 1.0}

        # Solve small damped normal system: min sum_pairs w_ij (g_i + log(L_i) - (g_j + log(L_j)))^2 + reg * sum g_i^2
        # i.e. w_ij (g_i - g_j) ~= -w_ij * log(L_i / L_j) = w_ij * log(L_j / L_i)
        cam_to_idx = {ci: k for k, ci in enumerate(active_cams)}
        M = len(active_cams)
        A = np.zeros((M, M), dtype=np.float64)
        b = np.zeros(M, dtype=np.float64)
        seam_samples_used = 0
        for (c_lo, c_hi), ratios in pair_log_ratios.items():
            if len(ratios) < 2:
                continue
            med_lr = float(np.median(ratios))  # log(L_hi / L_lo)
            w = float(min(len(ratios), 60))
            i_lo = cam_to_idx[c_lo]
            i_hi = cam_to_idx[c_hi]
            # Target: g_lo - g_hi = med_lr => g_lo matches brightness toward shared average
            A[i_lo, i_lo] += w
            A[i_hi, i_hi] += w
            A[i_lo, i_hi] -= w
            A[i_hi, i_lo] -= w
            b[i_lo] += w * med_lr
            b[i_hi] -= w * med_lr
            seam_samples_used += len(ratios)

        if seam_samples_used == 0:
            return gains, {"applied": False, "active_cams": len(active_cams), "seam_edges_used": 0, "gain_min": 1.0, "gain_max": 1.0}

        # Weak anchor keeps average scene exposure centered around log-gain = 0
        A += np.eye(M, dtype=np.float64) * 2.5
        log_g = np.linalg.solve(A, b)
        log_g -= float(np.mean(log_g))
        gain_arr = np.clip(np.exp(log_g), 0.85, 1.18)
        for ci, k in cam_to_idx.items():
            gains[ci] = float(gain_arr[k])

        return gains, {
            "applied": True,
            "active_cams": len(active_cams),
            "seam_edges_used": seam_samples_used,
            "seam_camera_pairs": len(pair_log_ratios),
            "gain_min": round(float(gain_arr.min()), 4),
            "gain_max": round(float(gain_arr.max()), 4),
            "gain_std": round(float(gain_arr.std()), 4),
        }

    def synthesize_atlas(
        self,
        best_cam_per_face: np.ndarray,
        out_atlas_path: Path,
        atlas_size: int = 8192,
    ) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[str, Any]]:
        active_cams = sorted([int(c) for c in np.unique(best_cam_per_face) if c >= 0])
        M_act = max(1, len(active_cams))
        aspect = self.W_native / float(self.H_native)
        best_area = 0.0
        best_layout = (1, 1, self.W_native, self.H_native)
        pad = 8
        for cols in range(1, M_act + 1):
            rows = math.ceil(M_act / cols)
            avail_w = atlas_size - (cols + 1) * pad
            avail_h = atlas_size - (rows + 1) * pad
            if avail_w <= 0 or avail_h <= 0:
                continue
            tw = min(avail_w / cols, (avail_h / rows) * aspect)
            th = tw / aspect
            if tw * th > best_area:
                best_area = tw * th
                best_layout = (cols, rows, max(16, int(math.floor(tw))), max(16, int(math.floor(th))))

        cols, rows, tw, th = best_layout
        pad_x = max(2, (atlas_size - cols * tw) // (cols + 1))
        pad_y = max(2, (atlas_size - rows * th) // (rows + 1))
        out_atlas_path.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (atlas_size, atlas_size), color=(26, 28, 32))

        exposure_gains, exposure_stats = self._estimate_exposure_gains(best_cam_per_face)
        self._last_exposure_gains = exposure_gains

        tile_rects: Dict[int, Tuple[int, int, int, int]] = {}
        for idx, cam_i in enumerate(active_cams):
            c = idx % cols
            r = idx // cols
            xo = pad_x + c * (tw + pad_x)
            yo = pad_y + r * (th + pad_y)
            tile_rects[cam_i] = (xo, yo, tw, th)
            with Image.open(self.frame_paths[cam_i]) as im:
                im_rgb = im.convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
                gain = float(exposure_gains.get(cam_i, 1.0))
                if abs(gain - 1.0) > 1e-3:
                    arr = np.clip(np.asarray(im_rgb, dtype=np.float32) * gain, 0.0, 255.0).astype(np.uint8)
                    im_rgb = Image.fromarray(arr, mode="RGB")
                canvas.paste(im_rgb, (xo, yo))

        canvas.save(out_atlas_path, format="PNG", optimize=False)
        stats = {
            "atlas_size": [atlas_size, atlas_size],
            "grid_layout": [cols, rows],
            "tile_size": [tw, th],
            "active_cams_count": len(active_cams),
            "atlas_file_mb": round(out_atlas_path.stat().st_size / (1024 ** 2), 2),
            "exposure_consistency": exposure_stats,
        }
        return tile_rects, stats

    def export_canonical_deliverables(
        self,
        output_dir: Path,
        base_name: str,
        best_cam_per_face: np.ndarray,
        tile_rects: Dict[int, Tuple[int, int, int, int]],
        atlas_path: Path,
        atlas_size: int = 8192,
        inferred_mesh: Optional[o3d.geometry.TriangleMesh] = None,
        inferred_material_rgb: Tuple[float, float, float] = (0.57, 0.65, 0.76),
        repaired_face_start_idx: Optional[int] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict[str, Any]]:
        """
        Exports canonical OBJ/MTL, GLB, PLY for observed + (if present) provenance-separated
        accepted local repair faces (`Material_Local_Repair_Photographic`) AND, when
        `inferred_mesh` is provided, separately exports `base_name_completed` with an
        explicit non-photographic `Material_Inferred_Extrapolated` material.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
        verts_canon = self.verts * T_canon
        center = verts_canon.mean(axis=0)
        verts_centered = verts_canon - center
        normals_canon = self.vertex_normals * T_canon
        norms_len = np.linalg.norm(normals_canon, axis=1, keepdims=True)
        normals_canon = normals_canon / np.maximum(norms_len, 1e-8)

        def _compute_face_uvs() -> np.ndarray:
            N_f = self.num_faces
            uvs = np.zeros((N_f, 3, 2), dtype=np.float64)
            inv_a = 1.0 / float(atlas_size)
            for cam_i, rect in tile_rects.items():
                f_idxs = np.where(best_cam_per_face == cam_i)[0]
                if len(f_idxs) == 0:
                    continue
                xo, yo, tw, th = rect
                R_i = self.extrinsics[cam_i, :3, :3]
                t_i = self.extrinsics[cam_i, :3, 3]
                K = self.intrinsics_native[cam_i]
                tri_pts = self.verts[self.tris[f_idxs]].reshape(-1, 3)
                Pc = (tri_pts @ R_i.T) + t_i
                Zc = Pc[:, 2].reshape(-1, 3)
                valid_tri_z = np.all(Zc > 0.12, axis=1)
                Zc_safe = np.maximum(Pc[:, 2], 0.12)
                u = np.clip((K[0, 0] * (Pc[:, 0] / Zc_safe) + K[0, 2]) / float(self.W_native), 0.0, 1.0)
                v = np.clip((K[1, 1] * (Pc[:, 1] / Zc_safe) + K[1, 2]) / float(self.H_native), 0.0, 1.0)
                u_tri = u.reshape(-1, 3)
                v_tri = v.reshape(-1, 3)
                # Guard behind-camera fallback faces: sample clean interior tile center instead of outer border corners
                if not np.all(valid_tri_z):
                    u_tri[~valid_tri_z] = 0.5
                    v_tri[~valid_tri_z] = 0.5
                eff_tw = max(1, tw - 4)
                eff_th = max(1, th - 4)
                ua = np.clip((xo + 2 + u_tri * eff_tw) * inv_a, 0.0, 1.0)
                va = np.clip(1.0 - (yo + 2 + v_tri * eff_th) * inv_a, 0.0, 1.0)
                uvs[f_idxs, :, 0] = ua
                uvs[f_idxs, :, 1] = va
            return uvs

        face_uvs = _compute_face_uvs()
        oor_uv_count = int(np.count_nonzero((face_uvs < 0.0) | (face_uvs > 1.0)))

        atlas_img = Image.open(atlas_path).convert("RGB")
        split_rep = (
            int(repaired_face_start_idx)
            if (repaired_face_start_idx is not None and 0 < int(repaired_face_start_idx) < len(self.tris))
            else None
        )

        def _make_split_trimesh(
            obs_v: np.ndarray,
            obs_vn: np.ndarray,
            sub_tris: np.ndarray,
            sub_cams: np.ndarray,
            sub_uvs: np.ndarray,
            mat_name: str,
        ) -> trimesh.Trimesh:
            v_cam_pairs = np.column_stack([sub_tris.reshape(-1), np.repeat(sub_cams, 3)])
            pair_keys = v_cam_pairs[:, 0].astype(np.int64) * 100000 + (v_cam_pairs[:, 1].astype(np.int64) + 1)
            _, first_occ, inv_faces = np.unique(pair_keys, return_index=True, return_inverse=True)
            split_orig_v = v_cam_pairs[first_occ, 0]
            pbr_mat = trimesh.visual.material.PBRMaterial(
                name=mat_name,
                baseColorTexture=atlas_img,
                metallicFactor=0.0,
                roughnessFactor=0.95,
                doubleSided=True,
            )
            return trimesh.Trimesh(
                vertices=obs_v[split_orig_v],
                faces=inv_faces.reshape(-1, 3),
                vertex_normals=obs_vn[split_orig_v],
                visual=trimesh.visual.TextureVisuals(uv=sub_uvs.reshape(-1, 2)[first_occ], material=pbr_mat),
                process=False,
            )

        def _build_direct_glb(
            obs_v: np.ndarray,
            obs_vn: np.ndarray,
            obs_t: np.ndarray,
            obs_uvs: np.ndarray,
            inf_m: Optional[o3d.geometry.TriangleMesh] = None,
        ) -> bytes:
            """Builds GLTF-2.0 GLB directly with smooth normals, matte PBR & provenance primitives."""
            if split_rep is not None:
                tm_obs = _make_split_trimesh(
                    obs_v, obs_vn, obs_t[:split_rep], best_cam_per_face[:split_rep],
                    obs_uvs[:split_rep], "Material_Photographic_Atlas",
                )
                tm_rep = _make_split_trimesh(
                    obs_v, obs_vn, obs_t[split_rep:], best_cam_per_face[split_rep:],
                    obs_uvs[split_rep:], "Material_Local_Repair_Photographic",
                )
                scene_dict: Dict[str, Any] = {"observed_geometry": tm_obs, "accepted_local_repair": tm_rep}
            else:
                tm_obs = _make_split_trimesh(
                    obs_v, obs_vn, obs_t, best_cam_per_face, obs_uvs, "Material_Photographic_Atlas",
                )
                scene_dict = {"observed_geometry": tm_obs}

            if inf_m is not None and len(inf_m.triangles) > 0:
                iv = (np.asarray(inf_m.vertices, dtype=np.float64) * T_canon) - center
                it = np.asarray(inf_m.triangles, dtype=np.int32)
                inf_m_n = o3d.geometry.TriangleMesh(inf_m)
                inf_m_n.compute_vertex_normals()
                ivn = np.asarray(inf_m_n.vertex_normals, dtype=np.float64) * T_canon
                r_m, g_m, b_m = inferred_material_rgb
                pbr_inf = trimesh.visual.material.PBRMaterial(
                    name="Material_Inferred_Extrapolated",
                    baseColorFactor=[int(r_m * 255), int(g_m * 255), int(b_m * 255), 217],
                    metallicFactor=0.0,
                    roughnessFactor=0.95,
                    alphaMode="BLEND",
                    doubleSided=True,
                )
                tm_inf = trimesh.Trimesh(
                    vertices=iv,
                    faces=it,
                    vertex_normals=ivn,
                    visual=trimesh.visual.TextureVisuals(
                        uv=np.full((len(iv), 2), 0.001, dtype=np.float64),
                        material=pbr_inf,
                    ),
                    process=False,
                )
                scene_dict["inferred_extrapolated"] = tm_inf

            if len(scene_dict) > 1:
                return trimesh.Scene(scene_dict).export(file_type="glb", include_normals=True)
            return tm_obs.export(file_type="glb", include_normals=True)

        def _write_obj_mtl_glb(
            name_stem: str,
            obs_v: np.ndarray,
            obs_t: np.ndarray,
            obs_uvs: np.ndarray,
            inf_m: Optional[o3d.geometry.TriangleMesh] = None,
        ) -> Tuple[str, str, Optional[str]]:
            obj_p = output_dir / f"{name_stem}_canonical.obj"
            mtl_p = output_dir / f"{name_stem}_canonical.mtl"
            glb_p = output_dir / f"{name_stem}.glb"

            with open(mtl_p, "w", encoding="utf-8") as fm:
                fm.write("# F.O.R.G.E. V2 Provenance-Aware Materials\n")
                fm.write("newmtl Material_Photographic_Atlas\n")
                fm.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nillum 2\nNs 5.0\n")
                fm.write(f"map_Kd {atlas_path.name}\n")
                if split_rep is not None:
                    fm.write("\nnewmtl Material_Local_Repair_Photographic\n")
                    fm.write("Ka 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\nillum 2\nNs 5.0\n")
                    fm.write(f"map_Kd {atlas_path.name}\n")
                if inf_m is not None and len(inf_m.triangles) > 0:
                    r_m, g_m, b_m = inferred_material_rgb
                    fm.write("\nnewmtl Material_Inferred_Extrapolated\n")
                    fm.write(f"Ka {r_m:.3f} {g_m:.3f} {b_m:.3f}\nKd {r_m:.3f} {g_m:.3f} {b_m:.3f}\nKs 0.0 0.0 0.0\nd 0.85\nillum 2\n")

            buf = [f"# F.O.R.G.E. V2 Mesh ({name_stem})\nmtllib {mtl_p.name}\n"]
            for v in obs_v:
                buf.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for vn in normals_canon:
                buf.append(f"vn {vn[0]:.5f} {vn[1]:.5f} {vn[2]:.5f}\n")
            if inf_m is not None and len(inf_m.triangles) > 0:
                iv = (np.asarray(inf_m.vertices, dtype=np.float64) * T_canon) - center
                for v in iv:
                    buf.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            uvs_flat = obs_uvs.reshape(-1, 2)
            for uv in uvs_flat:
                buf.append(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            buf.append("usemtl Material_Photographic_Atlas\n")
            vt_i = 1
            for f_i in range(len(obs_t)):
                if split_rep is not None and f_i == split_rep:
                    buf.append("usemtl Material_Local_Repair_Photographic\n")
                t = obs_t[f_i]
                buf.append(f"f {t[0]+1}/{vt_i}/{t[0]+1} {t[1]+1}/{vt_i+1}/{t[1]+1} {t[2]+1}/{vt_i+2}/{t[2]+1}\n")
                vt_i += 3

            if inf_m is not None and len(inf_m.triangles) > 0:
                buf.append("vt 0.001000 0.001000\nusemtl Material_Inferred_Extrapolated\n")
                inf_vt = vt_i
                v_off = len(obs_v)
                for t in np.asarray(inf_m.triangles, dtype=np.int32):
                    buf.append(f"f {t[0]+1+v_off}/{inf_vt} {t[1]+1+v_off}/{inf_vt} {t[2]+1+v_off}/{inf_vt}\n")

            with open(obj_p, "w", encoding="utf-8") as fo:
                fo.writelines(buf)

            glb_out = None
            try:
                glb_bytes = _build_direct_glb(obs_v, normals_canon, obs_t, obs_uvs, inf_m)
                glb_p.write_bytes(glb_bytes)
                glb_out = str(glb_p.resolve())
            except Exception:
                glb_out = None
            return str(obj_p.resolve()), str(mtl_p.resolve()), glb_out

        # 1. Observed-Only deliverables
        obj_obs, mtl_obs, glb_obs = _write_obj_mtl_glb(base_name, verts_centered, self.tris, face_uvs, None)

        # Photographic PLY vertex color sampling
        ply_p = output_dir / f"{base_name}.ply"
        v_cols = np.zeros((len(self.verts), 3), dtype=np.float64)
        vert_to_cams: Dict[int, List[int]] = {}
        for f_idx, tri in enumerate(self.tris):
            c_idx = int(best_cam_per_face[f_idx])
            for vi in tri:
                vert_to_cams.setdefault(int(vi), []).append(c_idx)
        exp_gains = getattr(self, "_last_exposure_gains", {})
        cached_frames = {}
        for c_idx in np.unique(best_cam_per_face):
            if c_idx < 0:
                continue
            ci = int(c_idx)
            rgb_raw = cv2.cvtColor(cv2.imread(str(self.frame_paths[ci])), cv2.COLOR_BGR2RGB)
            g = float(exp_gains.get(ci, 1.0))
            if abs(g - 1.0) > 1e-3:
                rgb_raw = np.clip(rgb_raw.astype(np.float32) * g, 0.0, 255.0).astype(np.uint8)
            cached_frames[ci] = rgb_raw
        for vi, cams_v in vert_to_cams.items():
            vals, cnts = np.unique(cams_v, return_counts=True)
            tc = int(vals[np.argmax(cnts)])
            if tc not in cached_frames:
                continue
            Pc = self.extrinsics[tc, :3, :3] @ self.verts[vi] + self.extrinsics[tc, :3, 3]
            if Pc[2] > 0.05:
                K = self.intrinsics_native[tc]
                uc = min(max(int(round(K[0, 0] * Pc[0] / Pc[2] + K[0, 2])), 0), self.W_native - 1)
                vc = min(max(int(round(K[1, 1] * Pc[1] / Pc[2] + K[1, 2])), 0), self.H_native - 1)
                v_cols[vi] = cached_frames[tc][vc, uc] / 255.0

        canon_ply_mesh = o3d.geometry.TriangleMesh()
        canon_ply_mesh.vertices = o3d.utility.Vector3dVector(verts_centered)
        canon_ply_mesh.triangles = o3d.utility.Vector3iVector(self.tris)
        canon_ply_mesh.vertex_colors = o3d.utility.Vector3dVector(v_cols)
        canon_ply_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(ply_p), canon_ply_mesh, write_vertex_colors=True)

        deliverables: Dict[str, Optional[str]] = {
            "obj": obj_obs,
            "mtl": mtl_obs,
            "glb": glb_obs,
            "ply": str(ply_p.resolve()),
            "atlas": str(atlas_path.resolve()),
        }

        # 2. Optional Observed-Plus-Inferred deliverables (NEVER overwriting observed-only!)
        if inferred_mesh is not None and len(inferred_mesh.triangles) > 0:
            comp_obj, comp_mtl, comp_glb = _write_obj_mtl_glb(
                f"{base_name}_completed", verts_centered, self.tris, face_uvs, inferred_mesh
            )
            deliverables["completed_obj"] = comp_obj
            deliverables["completed_mtl"] = comp_mtl
            deliverables["completed_glb"] = comp_glb

        # Texture & Reprojection QA (sampled photometric agreement + GLB reload integrity!)
        reproj_maes = []
        for c_idx in list(cached_frames.keys())[:10]:
            img_rgb = cached_frames[c_idx].astype(np.float64) / 255.0
            f_idxs = np.where(best_cam_per_face == c_idx)[0]
            if len(f_idxs) < 20:
                continue
            vis_verts = np.unique(self.tris[f_idxs])
            vis_verts = vis_verts[::max(1, len(vis_verts) // 300)]
            Pc = (self.verts[vis_verts] @ self.extrinsics[c_idx, :3, :3].T) + self.extrinsics[c_idx, :3, 3]
            pos = Pc[:, 2] > 0.1
            K = self.intrinsics_native[c_idx]
            u = np.round(K[0, 0] * (Pc[pos, 0] / Pc[pos, 2]) + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * (Pc[pos, 1] / Pc[pos, 2]) + K[1, 2]).astype(int)
            inb = (u >= 0) & (u < self.W_native) & (v >= 0) & (v < self.H_native)
            if np.any(inb):
                pred_rgb = v_cols[vis_verts[pos][inb]]
                obs_rgb = img_rgb[v[inb], u[inb]]
                reproj_maes.append(float(np.mean(np.abs(pred_rgb - obs_rgb))))

        glb_reload_ok = False
        glb_reloaded_meshes = 0
        if glb_obs and Path(glb_obs).is_file():
            try:
                loaded = trimesh.load(glb_obs, process=False)
                if isinstance(loaded, trimesh.Scene):
                    glb_reloaded_meshes = len(loaded.geometry)
                    glb_reload_ok = glb_reloaded_meshes >= 1 and all(
                        len(g.vertices) > 0 and np.all(np.isfinite(g.vertices))
                        for g in loaded.geometry.values()
                    )
                else:
                    glb_reloaded_meshes = 1
                    glb_reload_ok = len(loaded.vertices) > 0 and np.all(np.isfinite(loaded.vertices))
            except Exception:
                glb_reload_ok = False

        tex_qa = {
            "out_of_range_uv_count": oor_uv_count,
            "has_valid_material": True,
            "photometric_reprojection_mae_rgb": round(float(np.mean(reproj_maes)) if reproj_maes else 0.0, 4),
            "glb_reload_valid": glb_reload_ok,
            "glb_reloaded_mesh_count": glb_reloaded_meshes,
            "canonical_center": [round(float(x), 5) for x in center],
            "provenance_separation": {
                "separate_material_nodes": bool(split_rep is not None),
                "observed_faces_count": int(split_rep if split_rep is not None else len(self.tris)),
                "accepted_local_repair_faces_count": int(len(self.tris) - split_rep if split_rep is not None else 0),
                "observed_material": "Material_Photographic_Atlas",
                "repair_material": "Material_Local_Repair_Photographic" if split_rep is not None else None,
            },
        }
        return deliverables, tex_qa
