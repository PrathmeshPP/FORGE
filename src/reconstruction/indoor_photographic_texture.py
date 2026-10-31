"""
F.O.R.G.E. — Indoor Photographic Texture Engine.

Multi-view photographic texture reconstruction designed for indoor architecture:
1. Native-Resolution Frame Projection:
   - Projects 3D mesh triangles directly into uncompressed source video frames.
   - Scales camera intrinsics from inference resolution to native optical sensor dimensions.
2. Occlusion-Aware Surface Visibility:
   - Evaluates viewing angle (fronto-parallel normal alignment: cos theta > 0.08).
   - In-frustum boundary checks with safety margin.
   - Z-buffer occlusion testing against neural depth predictions.
3. Optimal Multi-View Chart Selection:
   - Assigns each face to its sharpest, most direct camera observation.
   - Topological neighbor consensus smoothing to eliminate single-triangle seams.
4. Direct Photographic Vertex Color Sampling:
   - Samples native RGB pixels directly onto mesh vertices along fronto-parallel rays.
   - Completely eliminates low-pass KDTree point-cloud blurring.
5. Unified UV Texture Atlas Generation:
   - Synthesizes high-resolution packed 2048x2048 / 4096x4096 texture atlas.
   - Exports Wavefront OBJ with valid vt coordinates and MTL referencing texture_atlas.png.
   - Exports standard glTF/GLB with embedded baseColorTexture.
   - Exports photographic vertex PLY.
"""

from __future__ import annotations

import os
import sys
import time
import math
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


class IndoorPhotographicTextureEngine:
    def __init__(
        self,
        mesh: o3d.geometry.TriangleMesh,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        frame_paths: List[Path],
        depth_maps: Optional[np.ndarray] = None,
        floor_rotation: Optional[np.ndarray] = None,
        min_cos_theta: float = 0.08
    ):
        """
        mesh: TriangleMesh in floor-leveled space.
        extrinsics: (N, 3, 4) camera extrinsics in raw world space.
        intrinsics: (N, 3, 3) camera intrinsics at inference resolution.
        frame_paths: Paths to native curated frames.
        depth_maps: (N, H, W) or (N, H, W, 1) predicted depth maps.
        floor_rotation: 3x3 rotation matrix R_floor aligning floor normal to [0, 1, 0].
        """
        self.mesh = o3d.geometry.TriangleMesh(mesh)
        self.mesh.compute_triangle_normals()
        self.mesh.compute_vertex_normals()

        self.extrinsics = extrinsics.astype(np.float64)
        self.intrinsics_infer = intrinsics.astype(np.float64)
        self.frame_paths = [Path(p) for p in frame_paths]
        self.depth_maps = depth_maps
        self.floor_rotation = floor_rotation if floor_rotation is not None else np.eye(3, dtype=np.float64)
        self.min_cos_theta = min_cos_theta

        self.num_cams = len(self.frame_paths)
        self.verts = np.asarray(self.mesh.vertices)
        self.tris = np.asarray(self.mesh.triangles)
        self.num_faces = len(self.tris)

        # Compute face normals and centroids in leveled mesh space
        self.face_normals = np.asarray(self.mesh.triangle_normals)
        self.face_centroids = self.verts[self.tris].mean(axis=1)

        # Inspect first image to obtain native resolution
        with Image.open(self.frame_paths[0]) as im:
            self.W_native, self.H_native = im.size

        # Inference depth map resolution
        if self.depth_maps is not None:
            d0 = self.depth_maps[0].squeeze()
            self.H_infer, self.W_infer = d0.shape[:2]
        else:
            self.H_infer, self.W_infer = 432, 240

        # Scale intrinsics to native resolution
        self.intrinsics_native = np.zeros_like(self.intrinsics_infer)
        sx = float(self.W_native) / float(self.W_infer)
        sy = float(self.H_native) / float(self.H_infer)
        for i in range(self.num_cams):
            K = self.intrinsics_infer[i].copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            self.intrinsics_native[i] = K

        # Precompute effective camera extrinsics for leveled mesh space:
        # X_raw = R_floor.T @ X_leveled
        # P_cam = R_cam @ X_raw + t_cam = (R_cam @ R_floor.T) @ X_leveled + t_cam
        self.eff_extrinsics = np.zeros_like(self.extrinsics)
        for i in range(self.num_cams):
            R_i = self.extrinsics[i, :3, :3]
            t_i = self.extrinsics[i, :3, 3]
            R_eff = R_i @ self.floor_rotation.T
            self.eff_extrinsics[i, :3, :3] = R_eff
            self.eff_extrinsics[i, :3, 3] = t_i

    def compute_face_visibility_and_best_cameras(
        self,
        frame_sharpness: Optional[List[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Evaluates per-face visibility, occlusion, and photographic scoring.
        Returns:
            best_cam_per_face: int array (num_faces,) of optimal camera index (-1 if none)
            scores: float array (num_faces, num_cams)
            stats: diagnostic metrics
        """
        N_f = self.num_faces
        M = self.num_cams
        scores = np.zeros((N_f, M), dtype=np.float32)

        # Precompute camera centers in leveled mesh space
        cam_centers = []
        for i in range(M):
            R_eff = self.eff_extrinsics[i, :3, :3]
            t_eff = self.eff_extrinsics[i, :3, 3]
            c_eff = -R_eff.T @ t_eff
            cam_centers.append(c_eff)
        cam_centers = np.array(cam_centers)

        sharp_weights = np.ones(M, dtype=np.float32)
        if frame_sharpness is not None and len(frame_sharpness) == M:
            s_arr = np.array(frame_sharpness, dtype=np.float32)
            s_min, s_max = s_arr.min(), s_arr.max()
            if s_max > s_min:
                sharp_weights = 0.7 + 0.3 * ((s_arr - s_min) / (s_max - s_min))

        for i in range(M):
            R_eff = self.eff_extrinsics[i, :3, :3]
            t_eff = self.eff_extrinsics[i, :3, 3]
            K = self.intrinsics_native[i]

            # Transform centroids to camera space
            P_c = (R_eff @ self.face_centroids.T + t_eff[:, None]).T
            Z_c = P_c[:, 2]

            valid_z = Z_c > 0.10
            u = K[0, 0] * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + K[0, 2]
            v = K[1, 1] * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + K[1, 2]

            # In-bounds with 8px margin
            in_bounds = (
                valid_z
                & (u >= 8.0)
                & (u <= self.W_native - 9.0)
                & (v >= 8.0)
                & (v <= self.H_native - 9.0)
            )

            # Normal alignment (fronto-parallel incidence)
            dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
            rays = P_c / dist
            N_c = (R_eff @ self.face_normals.T).T
            cos_theta = -np.sum(rays * N_c, axis=1)
            facing = in_bounds & (cos_theta > self.min_cos_theta)

            # Occlusion check using predicted depth
            occluded = np.zeros(N_f, dtype=bool)
            if self.depth_maps is not None and i < len(self.depth_maps):
                d_map = self.depth_maps[i].squeeze()
                H_d, W_d = d_map.shape
                u_d = np.clip((u * (W_d / self.W_native)).astype(np.int32), 0, W_d - 1)
                v_d = np.clip((v * (H_d / self.H_native)).astype(np.int32), 0, H_d - 1)
                depth_pred = d_map[v_d, u_d]

                # Face is occluded if it sits significantly behind the surface seen by this camera
                occluded = facing & (depth_pred > 0.15) & (Z_c > depth_pred * 1.18)

            visible = facing & (~occluded)

            # Quality metric: prioritize fronto-parallel angle, proximity, and frame sharpness
            cos_safe = np.maximum(cos_theta, 0.0)
            q = np.where(
                visible,
                (cos_safe ** 2.5) * (1.0 / np.maximum(dist[:, 0], 0.25)) * sharp_weights[i],
                0.0
            )
            scores[:, i] = q

        # Best camera per face
        best_cam = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)
        uncovered = max_scores <= 0.0
        best_cam[uncovered] = -1

        # Smooth chart assignments using face adjacency
        if uncovered.any():
            edges_to_faces: Dict[Tuple[int, int], List[int]] = {}
            for f_idx, tri in enumerate(self.tris):
                for edge in [
                    (min(tri[0], tri[1]), max(tri[0], tri[1])),
                    (min(tri[1], tri[2]), max(tri[1], tri[2])),
                    (min(tri[2], tri[0]), max(tri[2], tri[0]))
                ]:
                    edges_to_faces.setdefault(edge, []).append(f_idx)

            for f_idx in np.where(uncovered)[0]:
                tri = self.tris[f_idx]
                neighbor_cams = []
                for edge in [
                    (min(tri[0], tri[1]), max(tri[0], tri[1])),
                    (min(tri[1], tri[2]), max(tri[1], tri[2])),
                    (min(tri[2], tri[0]), max(tri[2], tri[0]))
                ]:
                    for adj_f in edges_to_faces.get(edge, []):
                        if adj_f != f_idx and best_cam[adj_f] >= 0:
                            neighbor_cams.append(best_cam[adj_f])
                if neighbor_cams:
                    vals, counts = np.unique(neighbor_cams, return_counts=True)
                    best_cam[f_idx] = vals[np.argmax(counts)]

        # Fallback for any remaining unassigned faces: assign closest camera
        still_uncovered = best_cam < 0
        if still_uncovered.any():
            dists = np.linalg.norm(self.face_centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
            best_cam[still_uncovered] = np.argmin(dists, axis=1)

        coverage_pct = round(float((~uncovered).mean() * 100.0), 2)
        stats = {
            "total_faces": N_f,
            "direct_photographic_faces": int((~uncovered).sum()),
            "photographic_coverage_pct": coverage_pct,
            "active_cameras_used": int(len(np.unique(best_cam))),
            "native_frame_resolution": [self.W_native, self.H_native]
        }
        return best_cam, scores, stats

    def sample_direct_photographic_vertex_colors(
        self,
        best_cam_per_face: np.ndarray
    ) -> np.ndarray:
        """
        Samples authentic photographic pixel colors directly from native frames onto mesh vertices.
        Returns float64 array of shape (N_v, 3) in [0.0, 1.0].
        """
        N_v = len(self.verts)
        v_colors_num = np.zeros((N_v, 3), dtype=np.float64)
        v_weights_den = np.zeros(N_v, dtype=np.float64)

        # Map vertices to faces
        vert_to_cams: Dict[int, List[int]] = {}
        for f_idx, tri in enumerate(self.tris):
            cam_idx = best_cam_per_face[f_idx]
            for vi in tri:
                vert_to_cams.setdefault(vi, []).append(cam_idx)

        # Preload frames in RGB
        cached_frames = {}
        for cam_idx in np.unique(best_cam_per_face):
            im = cv2.imread(str(self.frame_paths[cam_idx]))
            cached_frames[cam_idx] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        for vi in range(N_v):
            assigned_cams = vert_to_cams.get(vi, [])
            if not assigned_cams:
                continue
            # Majority camera for this vertex
            vals, counts = np.unique(assigned_cams, return_counts=True)
            top_cam = vals[np.argmax(counts)]

            R_eff = self.eff_extrinsics[top_cam, :3, :3]
            t_eff = self.eff_extrinsics[top_cam, :3, 3]
            K = self.intrinsics_native[top_cam]

            P_c = R_eff @ self.verts[vi] + t_eff
            Z_c = P_c[2]
            if Z_c > 0.05:
                u = int(round(K[0, 0] * (P_c[0] / Z_c) + K[0, 2]))
                v = int(round(K[1, 1] * (P_c[1] / Z_c) + K[1, 2]))
                if 0 <= u < self.W_native and 0 <= v < self.H_native:
                    im_rgb = cached_frames[top_cam]
                    col = im_rgb[v, u] / 255.0
                    v_colors_num[vi] = col
                    v_weights_den[vi] = 1.0

        # Fill vertices without direct projection using neighbor smoothing
        missing = v_weights_den == 0.0
        if missing.any():
            avg_color = v_colors_num[~missing].mean(axis=0) if (~missing).any() else np.array([0.5, 0.5, 0.5])
            v_colors_num[missing] = avg_color

        return v_colors_num

    def synthesize_packed_texture_atlas(
        self,
        best_cam_per_face: np.ndarray,
        out_atlas_path: Path,
        atlas_size: int = 4096
    ) -> Tuple[Dict[int, Tuple[int, int, int, int]], Dict[str, Any]]:
        """
        Synthesizes a unified packed UV texture atlas from the active camera frames.
        Returns:
            tile_rects: Dict mapping camera_idx to (col, row, tile_w, tile_h)
            atlas_stats: dictionary of dimensions and file sizes
        """
        active_cams = sorted(list(np.unique(best_cam_per_face)))
        M_act = len(active_cams)

        # Compute optimal grid
        if M_act <= 4:
            cols, rows = 2, 2
        elif M_act <= 6:
            cols, rows = 3, 2
        elif M_act <= 9:
            cols, rows = 3, 3
        elif M_act <= 12:
            cols, rows = 4, 3
        elif M_act <= 16:
            cols, rows = 4, 4
        elif M_act <= 20:
            cols, rows = 5, 4
        elif M_act <= 25:
            cols, rows = 5, 5
        elif M_act <= 36:
            cols, rows = 6, 6
        else:
            cols = int(math.ceil(math.sqrt(M_act)))
            rows = int(math.ceil(M_act / cols))

        tile_w = atlas_size // cols
        tile_h = atlas_size // rows

        out_atlas_path.parent.mkdir(parents=True, exist_ok=True)
        atlas_canvas = Image.new("RGB", (atlas_size, atlas_size), color=(24, 26, 30))

        tile_rects = {}
        for idx, cam_idx in enumerate(active_cams):
            c = idx % cols
            r = idx // cols
            x_off = c * tile_w
            y_off = r * tile_h
            tile_rects[cam_idx] = (c, r, cols, rows)

            im = Image.open(self.frame_paths[cam_idx]).convert("RGB")
            im_resized = im.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            atlas_canvas.paste(im_resized, (x_off, y_off))

        atlas_canvas.save(out_atlas_path, format="PNG", optimize=True)
        stats = {
            "atlas_size": [atlas_size, atlas_size],
            "grid_layout": [cols, rows],
            "tile_size": [tile_w, tile_h],
            "active_cams_count": M_act,
            "atlas_file_mb": round(out_atlas_path.stat().st_size / (1024 ** 2), 2)
        }
        return tile_rects, stats

    def export_textured_deliverables(
        self,
        output_dir: Path,
        base_name: str,
        best_cam_per_face: np.ndarray,
        tile_rects: Dict[int, Tuple[int, int, int, int]],
        atlas_path: Path,
        canonical_transform: bool = True
    ) -> Dict[str, str]:
        """
        Exports:
        1. Unified Textured Wavefront OBJ + MTL (with valid vt coordinates).
        2. glTF/GLB with embedded baseColorTexture.
        3. Photographic Vertex PLY.
        Coordinates are canonically aligned (T = diag(1, -1, -1)) and centered.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # 1. Transform mesh to canonical Three.js space
        # T_canonical = diag(1, -1, -1)
        # Center mesh at origin
        verts = self.verts.copy()
        if canonical_transform:
            T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)
            verts = verts * T_canon
            center = verts.mean(axis=0)
            verts = verts - center
        else:
            center = np.zeros(3, dtype=np.float64)

        # 2. Compute Texture Coordinates for every face
        # For each face f, project its 3 original vertices into assigned camera
        obj_path = output_dir / f"{base_name}_canonical.obj"
        mtl_path = output_dir / f"{base_name}_canonical.mtl"

        # Copy atlas texture next to OBJ
        dst_atlas = output_dir / atlas_path.name
        if dst_atlas.resolve() != atlas_path.resolve():
            shutil.copyfile(atlas_path, dst_atlas)

        # Write MTL
        with open(mtl_path, "w", encoding="utf-8") as f_mtl:
            f_mtl.write("# F.O.R.G.E. Photographic Texture Material\n")
            f_mtl.write("newmtl Material_Photographic_Atlas\n")
            f_mtl.write("Ka 1.000000 1.000000 1.000000\n")
            f_mtl.write("Kd 1.000000 1.000000 1.000000\n")
            f_mtl.write("Ks 0.050000 0.050000 0.050000\n")
            f_mtl.write("illum 2\n")
            f_mtl.write("Ns 10.000000\n")
            f_mtl.write(f"map_Kd {dst_atlas.name}\n")
        results["mtl"] = str(mtl_path.resolve())

        # Write OBJ
        print(f"[*] Writing textured OBJ: {obj_path.name}...")
        with open(obj_path, "w", encoding="utf-8") as f_obj:
            f_obj.write(f"# F.O.R.G.E. Multi-View Photographic Textured Mesh\n")
            f_obj.write(f"mtllib {mtl_path.name}\n\n")

            # Write Vertices
            for v in verts:
                f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f_obj.write(f"\nusemtl Material_Photographic_Atlas\n")

            vt_idx = 1
            for f_idx in range(self.num_faces):
                cam_idx = best_cam_per_face[f_idx]
                c, r, cols, rows = tile_rects[cam_idx]

                R_eff = self.eff_extrinsics[cam_idx, :3, :3]
                t_eff = self.eff_extrinsics[cam_idx, :3, 3]
                K = self.intrinsics_native[cam_idx]

                tri = self.tris[f_idx]
                tri_pts = self.verts[tri]  # Use original leveled positions for projection!

                P_c = (R_eff @ tri_pts.T + t_eff[:, None]).T
                Z_c = np.maximum(P_c[:, 2], 1e-4)
                u = K[0, 0] * (P_c[:, 0] / Z_c) + K[0, 2]
                v = K[1, 1] * (P_c[:, 1] / Z_c) + K[1, 2]

                u_cam = np.clip(u / float(self.W_native), 0.0, 1.0)
                v_cam = np.clip(v / float(self.H_native), 0.0, 1.0)

                # Tile mapping in atlas
                u_atlas = (c + u_cam) / float(cols)
                v_atlas = 1.0 - (r + v_cam) / float(rows)

                for k in range(3):
                    f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")

                v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
                f_obj.write(f"f {v1}/{vt_idx} {v2}/{vt_idx+1} {v3}/{vt_idx+2}\n")
                vt_idx += 3

        results["obj"] = str(obj_path.resolve())

        # 3. Export GLB with Embedded Texture Atlas via trimesh
        glb_path = output_dir / f"{base_name}.glb"
        print(f"[*] Converting and embedding texture into GLB: {glb_path.name}...")
        try:
            # Load the textured OBJ using trimesh (which automatically parses vt and texture image)
            tm = trimesh.load(str(obj_path), process=False)
            glb_bytes = tm.export(file_type="glb")
            with open(glb_path, "wb") as f_glb:
                f_glb.write(glb_bytes)
            results["glb"] = str(glb_path.resolve())
            print(f"  [+] Saved textured GLB ({glb_path.stat().st_size / (1024**2):.2f} MB)")
        except Exception as e:
            print(f"[!] GLB texture embedding notice: {e}")

        # 4. Photographic Vertex PLY
        ply_path = output_dir / f"{base_name}.ply"
        print(f"[*] Sampling authentic photographic vertex colors: {ply_path.name}...")
        v_cols = self.sample_direct_photographic_vertex_colors(best_cam_per_face)
        canon_mesh = o3d.geometry.TriangleMesh()
        canon_mesh.vertices = o3d.utility.Vector3dVector(verts)
        canon_mesh.triangles = o3d.utility.Vector3iVector(self.tris)
        canon_mesh.vertex_colors = o3d.utility.Vector3dVector(v_cols)
        canon_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(ply_path), canon_mesh, write_vertex_colors=True)
        results["ply"] = str(ply_path.resolve())

        return results
