r"""
F.O.R.G.E. — Phase 4: Perfected Object-Aware Reconstruction & 4K Photographic Texturing
Target: WhatsApp Video 2026-11-01 at 14.28.11.mp4
Output: outputs/indoor/digital_twin_reconstruction_video_20261101_142811/
"""

import sys
import os
import time
import json
import math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "outputs/indoor/digital_twin_reconstruction_video_20261101_142811"
FRAMES_DIR = EXP_DIR / "frames"
GEOM_DIR = EXP_DIR / "camera_geometry"
MESH_DIR = EXP_DIR / "meshes"
OBJ_DIR = EXP_DIR / "objects"
TEX_DIR = EXP_DIR / "textures"

for d in [MESH_DIR, OBJ_DIR, TEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76, flush=True)
print("  F.O.R.G.E. PHASE 4: PERFECTED 4K DIGITAL TWIN RECONSTRUCTION", flush=True)
print("=" * 76, flush=True)

# 1. Load Leveled Geometry & Scale by Metric Factor 3.18
SCALE_FACTOR = 3.18
leveled_data = np.load(GEOM_DIR / "camera_poses_leveled.npz")
extrinsics_raw = leveled_data["extrinsics"]
intrinsics_raw = leveled_data["intrinsics"]
N_frames = len(extrinsics_raw)

extrinsics_scaled = extrinsics_raw.copy()
for i in range(N_frames):
    extrinsics_scaled[i, :3, 3] *= SCALE_FACTOR

frame_files = sorted(list(FRAMES_DIR.glob("frame_*.png")))
sample_bgr = cv2.imread(str(frame_files[0]))
H_native, W_native = sample_bgr.shape[:2]  # 854 x 480
H_vggt, W_vggt = leveled_data["depths"].shape[1:3]

intrinsics_native = np.zeros_like(intrinsics_raw, dtype=np.float64)
sx = float(W_native) / float(W_vggt)
sy = float(H_native) / float(H_vggt)
for i in range(N_frames):
    K = intrinsics_raw[i].copy().astype(np.float64)
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    intrinsics_native[i] = K

# 2. Build 4096 x 4096 Photographic Atlas (4x4 Grid of 1024x1024 Tiles)
print("\n[*] Synthesizing 4096x4096 Photographic Texture Atlas (16 Tiles)...", flush=True)
ATLAS_DIM = 4096
TILE_DIM = 1024
GRID = 4

atlas_im = Image.new("RGB", (ATLAS_DIM, ATLAS_DIM), color=(235, 235, 230))

TILE_FRAME_MAP = {
    0: (0, "Entrance Door & Hydraulic Closer"),
    1: (2, "Left Noticeboard & Chair 1"),
    2: (4, "Wardrobe 1 (Simply Salted Box)"),
    3: (6, "Bed 1 (Left Cot & Blue Sheet)"),
    4: (10, "Desk 1 (Left Study Desk & Drawers)"),
    5: (14, "Cooler 1 (Air Cooler & Drying Rack)"),
    6: (18, "Window Wall & Curtains"),
    7: (22, "Wardrobe 2 (BAJAJ Box)"),
    8: (26, "Bed 2 (Maroon Floral Mattress)"),
    9: (28, "Bed 2 Footboard & Blanket"),
    10: (30, "Desk 2 & White Curved Chair"),
    11: (34, "Wardrobe 3 & Luggage"),
    12: (38, "Entrance View from Aisle"),
    13: (12, "Tiled Stone Flooring"),
    14: (8, "Left Wall & Sockets"),
    15: (-1, "Neutral Plaster (Unobserved)")
}

# Clean architectural plaster for Tile 15
neutral_tile = Image.new("RGB", (TILE_DIM, TILE_DIM), color=(242, 240, 235))
draw_n = ImageDraw.Draw(neutral_tile)
for py in range(0, TILE_DIM, 32):
    for px in range(0, TILE_DIM, 32):
        shade = 238 + ((px * 7 + py * 13) % 7)
        draw_n.rectangle([(px, py), (px + 32, py + 32)], fill=(shade, shade, shade - 2))

tile_meta = {}
for tile_id in range(16):
    gx = tile_id % GRID
    gy = tile_id // GRID
    f_idx, desc = TILE_FRAME_MAP[tile_id]

    if tile_id == 13 and (EXP_DIR / "diagnostics/pure_stone_floor.png").exists():
        tile_im = Image.open(str(EXP_DIR / "diagnostics/pure_stone_floor.png")).convert("RGB").resize((TILE_DIM, TILE_DIM), Image.Resampling.LANCZOS)
    elif f_idx >= 0 and f_idx < len(frame_files):
        src_im = Image.open(str(frame_files[f_idx])).convert("RGB")
        tile_im = src_im.resize((TILE_DIM, TILE_DIM), Image.Resampling.LANCZOS)
    else:
        tile_im = neutral_tile

    atlas_im.paste(tile_im, (gx * TILE_DIM, gy * TILE_DIM))
    tile_meta[tile_id] = {
        "grid_x": gx,
        "grid_y": gy,
        "frame_idx": f_idx,
        "desc": desc,
        "u_min": gx / float(GRID),
        "u_max": (gx + 1) / float(GRID),
        "v_min": (GRID - 1 - gy) / float(GRID),
        "v_max": (GRID - gy) / float(GRID)
    }

atlas_path = TEX_DIR / "room_photographic_atlas.png"
atlas_im.save(str(atlas_path), quality=93)
print(f"[+] Saved 4096x4096 Photographic Atlas: {atlas_path.name}", flush=True)

# 3. Helper Functions
def make_quad_grid(p00, p10, p11, p01, res=0.10):
    v_u = p10 - p00
    v_v = p01 - p00
    len_u = np.linalg.norm(v_u)
    len_v = np.linalg.norm(v_v)

    nu = max(2, int(np.ceil(len_u / res)))
    nv = max(2, int(np.ceil(len_v / res)))

    u_steps = np.linspace(0.0, 1.0, nu + 1)
    v_steps = np.linspace(0.0, 1.0, nv + 1)

    verts = []
    for j in range(nv + 1):
        for i in range(nu + 1):
            pt = p00 + u_steps[i] * v_u + v_steps[j] * v_v
            verts.append(pt)
    verts = np.array(verts, dtype=np.float32)

    faces = []
    for j in range(nv):
        for i in range(nu):
            idx0 = j * (nu + 1) + i
            idx1 = idx0 + 1
            idx2 = (j + 1) * (nu + 1) + (i + 1)
            idx3 = idx2 - 1
            faces.append([idx0, idx1, idx2])
            faces.append([idx0, idx2, idx3])
    faces = np.array(faces, dtype=np.int32)
    return verts, faces

def make_box_mesh(min_pt, max_pt, res=0.10):
    x0, y0, z0 = min_pt
    x1, y1, z1 = max_pt

    quads = [
        # Bottom (-Y)
        (np.array([x0, y0, z1]), np.array([x1, y0, z1]), np.array([x1, y0, z0]), np.array([x0, y0, z0])),
        # Top (+Y)
        (np.array([x0, y1, z0]), np.array([x1, y1, z0]), np.array([x1, y1, z1]), np.array([x0, y1, z1])),
        # Front (-Z)
        (np.array([x1, y0, z0]), np.array([x0, y0, z0]), np.array([x0, y1, z0]), np.array([x1, y1, z0])),
        # Back (+Z)
        (np.array([x0, y0, z1]), np.array([x1, y0, z1]), np.array([x1, y1, z1]), np.array([x0, y1, z1])),
        # Left (-X)
        (np.array([x0, y0, z0]), np.array([x0, y0, z1]), np.array([x0, y1, z1]), np.array([x0, y1, z0])),
        # Right (+X)
        (np.array([x1, y0, z1]), np.array([x1, y0, z0]), np.array([x1, y1, z0]), np.array([x1, y1, z1]))
    ]

    all_v = []
    all_f = []
    offset = 0
    for q in quads:
        v, f = make_quad_grid(q[0], q[1], q[2], q[3], res=res)
        all_v.append(v)
        all_f.append(f + offset)
        offset += len(v)

    return np.vstack(all_v), np.vstack(all_f)

# 4. Smart UV Projection
def project_mesh_uvs(mesh_verts, mesh_faces, mode="auto", target_tile=None):
    """
    Computes photographic UVs for mesh:
    mode='floor': Maps uniformly across Tile 13 (photographic stone floor)
    mode='ceiling': Maps to Tile 15 (clean plaster)
    mode='furniture': Uses target_tile for primary faces and top faces, candidate views for others
    mode='auto': Standard multi-view winner-take-all
    """
    N_f = len(mesh_faces)
    v0 = mesh_verts[mesh_faces[:, 0]]
    v1 = mesh_verts[mesh_faces[:, 1]]
    v2 = mesh_verts[mesh_faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12)

    unrolled_verts = []
    unrolled_uvs = []
    unrolled_normals = []
    unrolled_faces = []

    if mode == "floor":
        # Map floor continuously to Tile 13 (Pure stone slab floor)
        gx, gy = 1, 3
        for fi in range(N_f):
            tri_v = mesh_verts[mesh_faces[fi]]
            fn = face_normals[fi]
            tri_uv = []
            for vi in range(3):
                # Scale so floor tile pattern repeats naturally every 1.50m
                u_norm = ((tri_v[vi, 0] + 10.0) / 1.50) % 1.0
                v_norm = ((tri_v[vi, 2] + 10.0) / 1.50) % 1.0
                u_atlas = (gx + u_norm) / float(GRID)
                v_atlas = 1.0 - ((gy + v_norm) / float(GRID))
                tri_uv.append([u_atlas, v_atlas])

            base_idx = len(unrolled_verts)
            unrolled_verts.extend(tri_v)
            unrolled_uvs.extend(tri_uv)
            unrolled_normals.extend([fn, fn, fn])
            unrolled_faces.append([base_idx, base_idx + 1, base_idx + 2])

        return (
            np.array(unrolled_verts, dtype=np.float32),
            np.array(unrolled_faces, dtype=np.int32),
            np.array(unrolled_uvs, dtype=np.float32),
            np.array(unrolled_normals, dtype=np.float32)
        )

    if mode == "ceiling":
        # Map ceiling to Tile 15 (Neutral plaster)
        gx, gy = 3, 3
        for fi in range(N_f):
            tri_v = mesh_verts[mesh_faces[fi]]
            fn = face_normals[fi]
            tri_uv = []
            for vi in range(3):
                u_norm = (tri_v[vi, 0] / 2.0) % 1.0
                v_norm = (tri_v[vi, 2] / 2.0) % 1.0
                u_atlas = (gx + u_norm) / float(GRID)
                v_atlas = 1.0 - ((gy + v_norm) / float(GRID))
                tri_uv.append([u_atlas, v_atlas])

            base_idx = len(unrolled_verts)
            unrolled_verts.extend(tri_v)
            unrolled_uvs.extend(tri_uv)
            unrolled_normals.extend([fn, fn, fn])
            unrolled_faces.append([base_idx, base_idx + 1, base_idx + 2])

        return (
            np.array(unrolled_verts, dtype=np.float32),
            np.array(unrolled_faces, dtype=np.int32),
            np.array(unrolled_uvs, dtype=np.float32),
            np.array(unrolled_normals, dtype=np.float32)
        )

    # General / Furniture mode
    best_tile = np.full(N_f, 15, dtype=int)
    best_score = np.zeros(N_f, dtype=np.float32)

    # If target_tile specified, candidate tiles prioritize it
    candidate_tiles = [target_tile] if (target_tile is not None and target_tile < 15) else list(range(15))

    for tile_id in candidate_tiles:
        f_idx = TILE_FRAME_MAP[tile_id][0]
        R = extrinsics_scaled[f_idx, :3, :3]
        t = extrinsics_scaled[f_idx, :3, 3]
        K = intrinsics_native[f_idx]
        C = - R.T @ t

        v_view = centroids - C
        dist = np.linalg.norm(v_view, axis=1, keepdims=True)
        v_view = v_view / (dist + 1e-12)
        cos_theta = - np.sum(face_normals * v_view, axis=1)

        X_cam = (R @ centroids.T).T + t
        z = X_cam[:, 2]

        valid_z = z > 0.15
        u = (K[0, 0] * X_cam[:, 0] / (z + 1e-12)) + K[0, 2]
        v = (K[1, 1] * X_cam[:, 1] / (z + 1e-12)) + K[1, 2]

        in_frustum = (
            valid_z &
            (u >= 0.02 * W_native) & (u <= 0.98 * W_native) &
            (v >= 0.02 * H_native) & (v <= 0.98 * H_native) &
            (cos_theta > 0.10)
        )

        score = cos_theta * (1.0 / (dist.squeeze(-1) + 1.0))
        score[~in_frustum] = 0.0

        better = score > best_score
        best_tile[better] = tile_id
        best_score[better] = score[better]

    # Fill unassigned faces with candidate views or target tile
    unassigned = best_tile == 15
    if np.any(unassigned) and target_tile is not None:
        # Check all other tiles
        for tile_id in range(15):
            f_idx = TILE_FRAME_MAP[tile_id][0]
            R = extrinsics_scaled[f_idx, :3, :3]
            t = extrinsics_scaled[f_idx, :3, 3]
            K = intrinsics_native[f_idx]
            C = - R.T @ t

            v_view = centroids[unassigned] - C
            dist = np.linalg.norm(v_view, axis=1, keepdims=True)
            v_view = v_view / (dist + 1e-12)
            cos_theta = - np.sum(face_normals[unassigned] * v_view, axis=1)

            X_cam = (R @ centroids[unassigned].T).T + t
            z = X_cam[:, 2]
            valid_z = z > 0.15
            u = (K[0, 0] * X_cam[:, 0] / (z + 1e-12)) + K[0, 2]
            v = (K[1, 1] * X_cam[:, 1] / (z + 1e-12)) + K[1, 2]

            in_frustum = (
                valid_z &
                (u >= 0.02 * W_native) & (u <= 0.98 * W_native) &
                (v >= 0.02 * H_native) & (v <= 0.98 * H_native) &
                (cos_theta > 0.10)
            )

            score = cos_theta * (1.0 / (dist.squeeze(-1) + 1.0))
            score[~in_frustum] = 0.0

            idx_sub = np.where(unassigned)[0]
            better_sub = score > best_score[unassigned]
            best_tile[idx_sub[better_sub]] = tile_id
            best_score[idx_sub[better_sub]] = score[better_sub]

    # If any faces still unassigned on furniture (e.g. top faces), project them using target_tile
    still_unassigned = best_tile == 15
    if np.any(still_unassigned) and target_tile is not None and target_tile < 15:
        best_tile[still_unassigned] = target_tile

    for fi in range(N_f):
        t_id = best_tile[fi]
        t_info = tile_meta[t_id]
        gx = t_info["grid_x"]
        gy = t_info["grid_y"]
        f_idx = t_info["frame_idx"]

        tri_v = mesh_verts[mesh_faces[fi]]
        fn = face_normals[fi]

        tri_uv = []
        is_top = fn[1] > 0.70
        if is_top and target_tile is not None and target_tile < 15:
            x_min, x_max = mesh_verts[:, 0].min(), mesh_verts[:, 0].max()
            z_min, z_max = mesh_verts[:, 2].min(), mesh_verts[:, 2].max()
            for vi in range(3):
                u_norm = np.clip(0.15 + 0.70 * (tri_v[vi, 0] - x_min) / (x_max - x_min + 1e-5), 0.05, 0.95)
                v_norm = np.clip(0.15 + 0.70 * (tri_v[vi, 2] - z_min) / (z_max - z_min + 1e-5), 0.05, 0.95)
                u_atlas = (gx + u_norm) / float(GRID)
                v_atlas = 1.0 - ((gy + v_norm) / float(GRID))
                tri_uv.append([u_atlas, v_atlas])
        elif f_idx >= 0 and f_idx < N_frames:
            R = extrinsics_scaled[f_idx, :3, :3]
            t = extrinsics_scaled[f_idx, :3, 3]
            K = intrinsics_native[f_idx]

            X_c = (R @ tri_v.T).T + t
            z_c = X_c[:, 2]

            for vi in range(3):
                if z_c[vi] > 0.10:
                    u_px = (K[0, 0] * X_c[vi, 0] / z_c[vi]) + K[0, 2]
                    v_px = (K[1, 1] * X_c[vi, 1] / z_c[vi]) + K[1, 2]
                    u_norm = np.clip(u_px / W_native, 0.005, 0.995)
                    v_norm = np.clip(v_px / H_native, 0.005, 0.995)
                else:
                    u_norm, v_norm = 0.5, 0.5

                u_atlas = (gx + u_norm) / float(GRID)
                v_atlas = 1.0 - ((gy + v_norm) / float(GRID))
                tri_uv.append([u_atlas, v_atlas])
        else:
            for vi in range(3):
                u_norm = (tri_v[vi, 0] * 1.5) % 1.0
                v_norm = (tri_v[vi, 2] * 1.5) % 1.0
                u_atlas = (gx + u_norm) / float(GRID)
                v_atlas = 1.0 - ((gy + v_norm) / float(GRID))
                tri_uv.append([u_atlas, v_atlas])

        base_idx = len(unrolled_verts)
        unrolled_verts.extend(tri_v)
        unrolled_uvs.extend(tri_uv)
        unrolled_normals.extend([fn, fn, fn])
        unrolled_faces.append([base_idx, base_idx + 1, base_idx + 2])

    return (
        np.array(unrolled_verts, dtype=np.float32),
        np.array(unrolled_faces, dtype=np.int32),
        np.array(unrolled_uvs, dtype=np.float32),
        np.array(unrolled_normals, dtype=np.float32)
    )

# 5. Build Architectural Shell
print("\n[*] Constructing Inward-Facing Architectural Shell (Floor, 4 Walls, Ceiling, Openings)...", flush=True)

X_R, X_L = -2.55, 2.65  # Right wing at -X, Left wing at +X
Y_B, Y_T = 0.00, 2.85
Z_W, Z_D = -3.25, 3.25  # Window at -Z, Door at +Z

shell_parts = []

# Floor
v_fl, f_fl = make_quad_grid(
    np.array([X_R, Y_B, Z_D]), np.array([X_L, Y_B, Z_D]),
    np.array([X_L, Y_B, Z_W]), np.array([X_R, Y_B, Z_W]), res=0.10
)
uv_fl = project_mesh_uvs(v_fl, f_fl, mode="floor")
shell_parts.append(uv_fl)

# Ceiling
v_ce, f_ce = make_quad_grid(
    np.array([X_R, Y_T, Z_W]), np.array([X_L, Y_T, Z_W]),
    np.array([X_L, Y_T, Z_D]), np.array([X_R, Y_T, Z_D]), res=0.10
)
uv_ce = project_mesh_uvs(v_ce, f_ce, mode="ceiling")
shell_parts.append(uv_ce)

# Left Wall (+X = +2.65, Simply Salted wardrobe, Bed 1, Desk 1)
v_lw, f_lw = make_quad_grid(
    np.array([X_L, Y_B, Z_D]), np.array([X_L, Y_B, Z_W]),
    np.array([X_L, Y_T, Z_W]), np.array([X_L, Y_T, Z_D]), res=0.10
)
uv_lw = project_mesh_uvs(v_lw, f_lw, mode="auto", target_tile=14)
shell_parts.append(uv_lw)

# Right Wall (-X = -2.55, Wardrobes 2&3, Bed 2, Desk 2)
v_rw, f_rw = make_quad_grid(
    np.array([X_R, Y_B, Z_W]), np.array([X_R, Y_B, Z_D]),
    np.array([X_R, Y_T, Z_D]), np.array([X_R, Y_T, Z_W]), res=0.10
)
uv_rw = project_mesh_uvs(v_rw, f_rw, mode="auto", target_tile=7)
shell_parts.append(uv_rw)

# Window Wall (-Z = -3.25)
# Left of window (1.25 to X_L)
v_ww1, f_ww1 = make_quad_grid(np.array([1.25, Y_B, Z_W]), np.array([X_L, Y_B, Z_W]), np.array([X_L, Y_T, Z_W]), np.array([1.25, Y_T, Z_W]), res=0.10)
shell_parts.append(project_mesh_uvs(v_ww1, f_ww1, mode="auto", target_tile=6))

# Right of window (X_R to -1.25)
v_ww2, f_ww2 = make_quad_grid(np.array([X_R, Y_B, Z_W]), np.array([-1.25, Y_B, Z_W]), np.array([-1.25, Y_T, Z_W]), np.array([X_R, Y_T, Z_W]), res=0.10)
shell_parts.append(project_mesh_uvs(v_ww2, f_ww2, mode="auto", target_tile=6))

# Below window (-1.25 to 1.25, Y: 0 to 0.70)
v_ww3, f_ww3 = make_quad_grid(np.array([-1.25, Y_B, Z_W]), np.array([1.25, Y_B, Z_W]), np.array([1.25, 0.70, Z_W]), np.array([-1.25, 0.70, Z_W]), res=0.10)
shell_parts.append(project_mesh_uvs(v_ww3, f_ww3, mode="auto", target_tile=6))

# Above window (-1.25 to 1.25, Y: 2.35 to 2.85)
v_ww4, f_ww4 = make_quad_grid(np.array([-1.25, 2.35, Z_W]), np.array([1.25, 2.35, Z_W]), np.array([1.25, Y_T, Z_W]), np.array([-1.25, Y_T, Z_W]), res=0.10)
shell_parts.append(project_mesh_uvs(v_ww4, f_ww4, mode="auto", target_tile=6))

# Door Wall (+Z = +3.25)
# Left of door (0.55 to X_L)
v_dw1, f_dw1 = make_quad_grid(np.array([X_L, Y_B, Z_D]), np.array([0.55, Y_B, Z_D]), np.array([0.55, Y_T, Z_D]), np.array([X_L, Y_T, Z_D]), res=0.10)
shell_parts.append(project_mesh_uvs(v_dw1, f_dw1, mode="auto", target_tile=1))

# Right of door (X_R to -0.55)
v_dw2, f_dw2 = make_quad_grid(np.array([-0.55, Y_B, Z_D]), np.array([X_R, Y_B, Z_D]), np.array([X_R, Y_T, Z_D]), np.array([-0.55, Y_T, Z_D]), res=0.10)
shell_parts.append(project_mesh_uvs(v_dw2, f_dw2, mode="auto", target_tile=0))

# Above door transom (-0.55 to 0.55, Y: 2.15 to 2.85)
v_dw3, f_dw3 = make_quad_grid(np.array([0.55, 2.15, Z_D]), np.array([-0.55, 2.15, Z_D]), np.array([-0.55, Y_T, Z_D]), np.array([0.55, Y_T, Z_D]), res=0.10)
shell_parts.append(project_mesh_uvs(v_dw3, f_dw3, mode="auto", target_tile=0))

# Merge shell parts
all_shell_v, all_shell_f, all_shell_uv, all_shell_n = [], [], [], []
off_s = 0
for sv, sf, suv, sn in shell_parts:
    all_shell_v.append(sv)
    all_shell_f.append(sf + off_s)
    all_shell_uv.append(suv)
    all_shell_n.append(sn)
    off_s += len(sv)

arch_u_v = np.vstack(all_shell_v)
arch_u_f = np.vstack(all_shell_f)
arch_u_uv = np.vstack(all_shell_uv)
arch_u_norm = np.vstack(all_shell_n)

mat_arch = trimesh.visual.material.PBRMaterial(
    name="mat_architectural_shell",
    baseColorTexture=Image.open(str(atlas_path)),
    roughnessFactor=0.85,
    metallicFactor=0.05
)
vis_arch = trimesh.visual.TextureVisuals(uv=arch_u_uv, material=mat_arch)
mesh_arch = trimesh.Trimesh(
    vertices=arch_u_v,
    faces=arch_u_f,
    vertex_normals=arch_u_norm,
    visual=vis_arch,
    validate=False,
    process=False
)
arch_glb_path = MESH_DIR / "architectural_shell.glb"
mesh_arch.export(str(arch_glb_path))
print(f"[+] Saved Architectural Shell GLB: {arch_glb_path.name} ({len(arch_u_f):,} triangles)", flush=True)

# 6. Reconstruct Furniture Objects
print("\n[*] Reconstructing Furniture Objects with Aligned Metric Boundaries...", flush=True)

objects_dict = {}

def register_object(name, min_pt, max_pt, target_tile=None, res=0.08):
    v, f = make_box_mesh(min_pt, max_pt, res=res)
    center = (min_pt + max_pt) / 2.0
    dims = max_pt - min_pt
    objects_dict[name] = {
        "verts": v,
        "faces": f,
        "center": center,
        "dimensions": dims,
        "bounding_box": [list(min_pt), list(max_pt)],
        "target_tile": target_tile
    }
    print(f"  [+] {name:18s}: {len(v):5d} verts, {len(f):5d} tris | Size: {dims[0]:.2f}m x {dims[1]:.2f}m x {dims[2]:.2f}m | Tile: {target_tile}", flush=True)

# Entrance Door Leaf (Z = +3.20)
register_object("Door_001", np.array([-0.50, 0.00, 3.18]), np.array([0.50, 2.12, 3.24]), target_tile=0)

# Window Curtains (Z = -3.20)
register_object("Curtains_001", np.array([-1.30, 0.05, -3.24]), np.array([1.30, 2.65, -3.14]), target_tile=6)

# LEFT WING (+X):
# Wardrobe 1 (Simply Salted Box)
register_object("Wardrobe_001", np.array([1.60, 0.00, 1.25]), np.array([2.50, 2.15, 2.25]), target_tile=2)

# Wooden Chair 1
register_object("Chair_001", np.array([1.15, 0.00, 2.10]), np.array([1.65, 0.85, 2.55]), target_tile=1, res=0.06)

# Bed 1 (Left Cot & Blue Sheet)
register_object("Bed_001", np.array([1.15, 0.00, -1.05]), np.array([2.15, 0.55, 0.95]), target_tile=3)

# Desk 1 (Left Study Desk & Drawers)
register_object("Desk_001", np.array([1.35, 0.00, -2.65]), np.array([2.45, 0.78, -1.45]), target_tile=4)

# RIGHT WING (-X):
# Wardrobe 3 (Front Right with Luggage)
register_object("Wardrobe_003", np.array([-2.55, 0.00, 1.35]), np.array([-1.65, 2.15, 2.35]), target_tile=11)

# Wardrobe 2 (Middle Right with BAJAJ Box)
register_object("Wardrobe_002", np.array([-2.55, 0.00, -0.55]), np.array([-1.65, 2.10, 0.45]), target_tile=7)

# Bed 2 (Right Cot & Maroon Floral Mattress)
register_object("Bed_002", np.array([-2.15, 0.00, -1.05]), np.array([-1.15, 0.55, 0.95]), target_tile=8)

# Desk 2 (Right Study Desk & Purple Dot Bin)
register_object("Desk_002", np.array([-2.45, 0.00, -2.65]), np.array([-1.35, 0.78, -1.45]), target_tile=10)

# Chair 2 (Desk Chair with White Curved Backrest)
register_object("Chair_002", np.array([-1.50, 0.00, -2.20]), np.array([-1.00, 0.95, -1.70]), target_tile=10, res=0.06)

# AISLE CENTER:
# Evaporative Air Cooler
register_object("Cooler_001", np.array([-0.30, 0.00, -2.35]), np.array([0.30, 0.90, -1.80]), target_tile=5)

# Clothes Drying Rack
register_object("ClothesRack_001", np.array([-0.35, 0.00, -2.85]), np.array([0.35, 1.65, -2.40]), target_tile=5, res=0.06)

# 7. Apply Photographic Textures & Export Each Object to GLB
print("\n[*] Projecting Photographic Textures and Exporting Individual Object GLBs...", flush=True)

manifest_objects = []

for obj_name, obj_data in objects_dict.items():
    ov = obj_data["verts"]
    of = obj_data["faces"]
    t_tile = obj_data["target_tile"]

    u_verts, u_faces, u_uvs, u_normals = project_mesh_uvs(ov, of, mode="furniture", target_tile=t_tile)

    mat = trimesh.visual.material.PBRMaterial(
        name=f"mat_{obj_name}",
        baseColorTexture=Image.open(str(atlas_path)),
        roughnessFactor=0.75,
        metallicFactor=0.10
    )
    visual = trimesh.visual.TextureVisuals(uv=u_uvs, material=mat)
    mesh_tri = trimesh.Trimesh(
        vertices=u_verts,
        faces=u_faces,
        vertex_normals=u_normals,
        visual=visual,
        validate=False,
        process=False
    )

    obj_glb_path = OBJ_DIR / f"{obj_name}.glb"
    mesh_tri.export(str(obj_glb_path))

    manifest_objects.append({
        "id": obj_name,
        "name": obj_name.replace("_", " "),
        "glb_path": f"objects/{obj_name}.glb",
        "center": [round(float(c), 3) for c in obj_data["center"]],
        "dimensions": [round(float(d), 3) for d in obj_data["dimensions"]],
        "bounding_box": obj_data["bounding_box"],
        "num_triangles": len(u_faces)
    })

print(f"[+] Exported {len(manifest_objects)} individual furniture GLB models to {OBJ_DIR}", flush=True)

# 8. Assemble Unified Complete Room Digital Twin Mesh
print("\n[*] Assembling Complete Unified Room Digital Twin (Shell + All Objects)...", flush=True)

all_room_v = [arch_u_v]
all_room_f = [arch_u_f]
all_room_uv = [arch_u_uv]
all_room_n = [arch_u_norm]

v_offset = len(arch_u_v)

for obj_name, obj_data in objects_dict.items():
    ov = obj_data["verts"]
    of = obj_data["faces"]
    t_tile = obj_data["target_tile"]
    u_verts, u_faces, u_uvs, u_normals = project_mesh_uvs(ov, of, mode="furniture", target_tile=t_tile)

    all_room_v.append(u_verts)
    all_room_f.append(u_faces + v_offset)
    all_room_uv.append(u_uvs)
    all_room_n.append(u_normals)

    v_offset += len(u_verts)

full_v = np.vstack(all_room_v)
full_f = np.vstack(all_room_f)
full_uv = np.vstack(all_room_uv)
full_n = np.vstack(all_room_n)

mat_full = trimesh.visual.material.PBRMaterial(
    name="mat_room_digital_twin",
    baseColorTexture=Image.open(str(atlas_path)),
    roughnessFactor=0.80,
    metallicFactor=0.05
)
vis_full = trimesh.visual.TextureVisuals(uv=full_uv, material=mat_full)
mesh_full = trimesh.Trimesh(
    vertices=full_v,
    faces=full_f,
    vertex_normals=full_n,
    visual=vis_full,
    validate=False,
    process=False
)

# Export Full Room GLB
room_glb_path = MESH_DIR / "room_digital_twin_photographic.glb"
mesh_full.export(str(room_glb_path))
print(f"[+] Exported Complete Room GLB: {room_glb_path.name} ({len(full_f):,} triangles, {len(full_v):,} vertices)", flush=True)

# Export Full Room OBJ/MTL
obj_path = MESH_DIR / "room_digital_twin_photographic.obj"
mtl_path = MESH_DIR / "room_digital_twin_photographic.mtl"

with open(mtl_path, "w") as f:
    f.write("newmtl mat_room_digital_twin\n")
    f.write("Ka 1.000 1.000 1.000\n")
    f.write("Kd 1.000 1.000 1.000\n")
    f.write("Ks 0.100 0.100 0.100\n")
    f.write("d 1.0\n")
    f.write("illum 2\n")
    f.write(f"map_Kd ../textures/{atlas_path.name}\n")

with open(obj_path, "w") as f:
    f.write(f"mtllib {mtl_path.name}\n")
    f.write("o Room_Digital_Twin\n")
    for v in full_v:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
    for vt in full_uv:
        f.write(f"vt {vt[0]:.5f} {vt[1]:.5f}\n")
    for vn in full_n:
        f.write(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}\n")
    f.write("usemtl mat_room_digital_twin\n")
    f.write("s 1\n")
    for tri in full_f:
        f.write(f"f {tri[0]+1}/{tri[0]+1}/{tri[0]+1} {tri[1]+1}/{tri[1]+1}/{tri[1]+1} {tri[2]+1}/{tri[2]+1}/{tri[2]+1}\n")

print(f"[+] Exported Complete Room OBJ & MTL: {obj_path.name}", flush=True)

# 9. Write Scene Manifest
scene_manifest = {
    "scene_name": "Hostel Room B 314 - Immersive Walkable Digital Twin",
    "video_source": "WhatsApp Video 2026-11-01 at 14.28.11.mp4",
    "sha256": "1c02de602ff34bd63d7332f662dfb6fb73de5676a05ca07c2704f1fbe6c6ce35",
    "primary_model_glb": "meshes/room_digital_twin_photographic.glb",
    "architectural_shell_glb": "meshes/architectural_shell.glb",
    "room_dimensions_m": {
        "width_x": round(float(X_L - X_R), 2),
        "height_y": round(float(Y_T - Y_B), 2),
        "length_z": round(float(Z_D - Z_W), 2)
    },
    "bounds": {
        "x_min": X_R, "x_max": X_L,
        "y_min": Y_B, "y_max": Y_T,
        "z_min": Z_W, "z_max": Z_D
    },
    "default_spawn": {
        "position": [0.0, 1.42, 2.5],
        "look_at": [0.0, 1.42, -2.5]
    },
    "objects": manifest_objects,
    "texture_atlas": {
        "path": f"textures/{atlas_path.name}",
        "resolution": [ATLAS_DIM, ATLAS_DIM],
        "tiles_grid": [GRID, GRID],
        "tiles": tile_meta
    }
}

with open(EXP_DIR / "diagnostics/scene_manifest.json", "w") as f:
    json.dump(scene_manifest, f, indent=2)

print(f"[+] Saved complete scene manifest: diagnostics/scene_manifest.json", flush=True)
print("=" * 76, flush=True)
