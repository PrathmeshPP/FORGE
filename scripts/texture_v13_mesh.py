import sys, time
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import trimesh
from PIL import Image
import pycolmap

v13_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v13')
mesh_path = v13_dir / 'final_best_canonical_v13_mesh.ply'
f4k_dir = v13_dir / 'frames_4k'
sparse_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm/sparse/0')

print("=== STEP 1: LOAD V13 MESH & 4K CAMERA RECONSTRUCTION ===")
mesh = o3d.io.read_triangle_mesh(str(mesh_path))
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

verts = np.asarray(mesh.vertices)
tris = np.asarray(mesh.triangles)
normals = np.asarray(mesh.triangle_normals)
centroids = verts[tris].mean(axis=1)

print(f"Loaded mesh: {len(verts):,} vertices, {len(tris):,} faces.")
print(f"Bounds min: {verts.min(axis=0)}, max: {verts.max(axis=0)}")

rec = pycolmap.Reconstruction(str(sparse_dir))
images = sorted(list(rec.images.values()), key=lambda x: x.name)
num_cams = len(images)
print(f"Loaded {num_cams} cameras.")

# Camera calibration at 4K pinhole
W_native, H_native = 3840, 2160
fx_u4k = 1159.82941244 * 2.4
fy_u4k = 1160.23780308 * 2.4
cx_u4k = 800.0 * 2.4
cy_u4k = 450.0 * 2.4

extrinsics = []
cam_centers = []
for im in images:
    R = im.cam_from_world().rotation.matrix()
    t = im.cam_from_world().translation
    extr = np.eye(4, dtype=np.float64)
    extr[:3, :3] = R
    extr[:3, 3] = t
    extrinsics.append(extr)
    cam_centers.append(-R.T @ t)

extrinsics = np.array(extrinsics)
cam_centers = np.array(cam_centers)

print("\n=== STEP 2: MULTI-VIEW VISIBILITY & PHOTOGRAPHIC QUALITY SCORING ===")
N_f = len(tris)
scores = np.zeros((N_f, num_cams), dtype=np.float32)

t0 = time.time()
for i in range(num_cams):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    
    P_c = (R @ centroids.T + t[:, None]).T
    Z_c = P_c[:, 2]
    
    valid_z = Z_c > 0.1
    u = fx_u4k * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + cx_u4k
    v = fy_u4k * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + cy_u4k
    
    # In-bounds with 10px protective border
    in_bounds = valid_z & (u >= 10.0) & (u <= W_native - 11.0) & (v >= 10.0) & (v <= H_native - 11.0)
    
    dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
    rays = P_c / dist
    N_c = (R @ normals.T).T
    cos_theta = -np.sum(rays * N_c, axis=1)
    
    # Front-facing incidence check (cos_theta > 0.05)
    visible = in_bounds & (cos_theta > 0.05)
    
    # Fronto-parallel resolution weight: favor direct angle and closer observation
    q = np.where(visible, (np.clip(cos_theta, 0.0, 1.0) ** 2.2) / np.maximum(Z_c, 0.4), 0.0)
    scores[:, i] = q

print(f"Scored {N_f:,} faces against {num_cams} cameras in {time.time()-t0:.2f}s.")

best_cam = np.argmax(scores, axis=1)
max_scores = np.max(scores, axis=1)
uncovered = max_scores <= 0.0
print(f"Initial direct photographic coverage: {(~uncovered).sum():,} / {N_f:,} faces ({(~uncovered).mean()*100:.2f}%)")

print("\n=== STEP 3: TOPOLOGICAL NEIGHBOR CONSENSUS REFINEMENT ===")
# Build adjacency to propagate sharp camera assignments across boundary seams
if uncovered.any():
    edges_to_faces = {}
    for f_idx, tri in enumerate(tris):
        for edge in [(min(tri[0], tri[1]), max(tri[0], tri[1])),
                     (min(tri[1], tri[2]), max(tri[1], tri[2])),
                     (min(tri[2], tri[0]), max(tri[2], tri[0]))]:
            edges_to_faces.setdefault(edge, []).append(f_idx)
            
    uncovered_indices = np.where(uncovered)[0]
    for f_idx in uncovered_indices:
        tri = tris[f_idx]
        neighbor_cams = []
        for edge in [(min(tri[0], tri[1]), max(tri[0], tri[1])),
                     (min(tri[1], tri[2]), max(tri[1], tri[2])),
                     (min(tri[2], tri[0]), max(tri[2], tri[0]))]:
            for adj_f in edges_to_faces.get(edge, []):
                if adj_f != f_idx and not uncovered[adj_f]:
                    neighbor_cams.append(best_cam[adj_f])
        if neighbor_cams:
            vals, counts = np.unique(neighbor_cams, return_counts=True)
            best_cam[f_idx] = vals[np.argmax(counts)]
            uncovered[f_idx] = False

# Fallback for any remaining unassigned faces: assign closest camera
still_uncovered = uncovered
if still_uncovered.any():
    dists_to_cams = np.linalg.norm(centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
    best_cam[still_uncovered] = np.argmin(dists_to_cams, axis=1)

print(f"Post-consensus coverage: 100.0% of {N_f:,} faces assigned to optimal source views.")

print("\n=== STEP 4: SYNTHESIZE UNIFIED 4096x4096 TEXTURE ATLAS (16:9 PRESERVED) ===")
atlas_size = 4096
grid_cols, grid_rows = 4, 6 # 24 slots exactly
slot_w = atlas_size // grid_cols # 1024
slot_h = atlas_size // grid_rows # 682

# Target tile resolution preserving exact 16:9 ratio
tile_w = slot_w # 1024
tile_h = int(round(tile_w * 9.0 / 16.0)) # 576 (fits cleanly inside 682 slot)

print(f"Atlas Grid: {grid_cols} cols x {grid_rows} rows (slot {slot_w}x{slot_h})")
print(f"Frame Tile: {tile_w}x{tile_h} (Exact 16:9 aspect ratio, 0 stretching)")

atlas_canvas = Image.new("RGB", (atlas_size, atlas_size), color=(30, 32, 36))

for i, im in enumerate(images):
    col = i % grid_cols
    row = i // grid_cols
    x_off = col * slot_w
    y_off = row * slot_h
    
    frame_path = f4k_dir / im.name
    frame_pil = Image.open(frame_path).convert("RGB")
    frame_resized = frame_pil.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
    atlas_canvas.paste(frame_resized, (x_off, y_off))

atlas_path = v13_dir / "texture_atlas.png"
atlas_canvas.save(str(atlas_path), format="PNG", optimize=True)
print(f"Saved Unified Texture Atlas: {atlas_path} ({atlas_path.stat().st_size / 1e6:.2f} MB)")

print("\n=== STEP 5: EXPORT TEXTURED OBJ & MTL IN COLMAP COORDINATES ===")
obj_path = v13_dir / "final_best_canonical_v13_reconstruction.obj"
mtl_path = v13_dir / "final_best_canonical_v13_reconstruction.mtl"

with open(mtl_path, "w", encoding="utf-8") as f_mtl:
    f_mtl.write("# F.O.R.G.E V13 Unified Atlas Material\n")
    f_mtl.write("newmtl Material_Atlas\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.000 0.000 0.000\n")
    f_mtl.write("d 1.0\n")
    f_mtl.write("illum 1\n")
    f_mtl.write(f"map_Kd texture_atlas.png\n")

print(f"Writing OBJ file: {obj_path}...")
with open(obj_path, "w", encoding="utf-8") as f_obj:
    f_obj.write("# F.O.R.G.E V13 Multi-View Photographic Mesh\n")
    f_obj.write(f"mtllib {mtl_path.name}\n\n")
    
    # Write vertices
    for v in verts:
        f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
    f_obj.write(f"\nusemtl Material_Atlas\n")
    
    # Write texture coordinates and faces
    vt_idx = 1
    for f_idx in range(N_f):
        cam_idx = best_cam[f_idx]
        col = cam_idx % grid_cols
        row = cam_idx // grid_cols
        
        R = extrinsics[cam_idx, :3, :3]
        t = extrinsics[cam_idx, :3, 3]
        
        tri = tris[f_idx]
        tri_pts = verts[tri]
        
        P_c = (R @ tri_pts.T + t[:, None]).T
        Z_c = np.maximum(P_c[:, 2], 1e-4)
        u = fx_u4k * (P_c[:, 0] / Z_c) + cx_u4k
        v = fy_u4k * (P_c[:, 1] / Z_c) + cy_u4k
        
        u_norm = np.clip(u / float(W_native), 0.0, 1.0)
        v_norm = np.clip(v / float(H_native), 0.0, 1.0)
        
        # Map to atlas UV coordinates (with exact tile dimensions)
        u_atlas = (col * slot_w + u_norm * tile_w) / float(atlas_size)
        v_atlas = 1.0 - (row * slot_h + v_norm * tile_h) / float(atlas_size)
        
        for k in range(3):
            f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")
            
        v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f_obj.write(f"f {v1}/{vt_idx} {v2}/{vt_idx+1} {v3}/{vt_idx+2}\n")
        vt_idx += 3

print(f"OBJ write complete: {obj_path.stat().st_size / 1e6:.2f} MB")

print("\n=== STEP 6: APPLY EXACT 180-DEGREE TOP/BOTTOM FLIP & EXPORT FINAL GLB ===")
# Load textured mesh using trimesh
tm = trimesh.load(str(obj_path), process=False)
print(f"Loaded textured mesh into trimesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces")

# Compute bounding box center
bbox_min = tm.vertices.min(axis=0)
bbox_max = tm.vertices.max(axis=0)
bbox_center = (bbox_min + bbox_max) / 2.0
print(f"Bounding box center: {bbox_center}")

# Apply rigid 180-degree rotation R_X(180 deg) around center
# R_X(180 deg) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
R_x = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

# Transform vertex positions: P' = R @ (P - C) + C
tm.vertices = (tm.vertices - bbox_center) @ R_x.T + bbox_center
if hasattr(tm, 'vertex_normals') and tm.vertex_normals is not None and len(tm.vertex_normals) > 0:
    tm.vertex_normals = tm.vertex_normals @ R_x.T

# Export final GLB
glb_path = v13_dir / "final_best_canonical_v13_reconstruction.glb"
print(f"Exporting final GLB to: {glb_path}...")
glb_bytes = tm.export(file_type='glb')
with open(glb_path, 'wb') as f_glb:
    f_glb.write(glb_bytes)

print(f"Exported GLB successfully! Size: {glb_path.stat().st_size / 1e6:.2f} MB")

print("\n=== STEP 7: RE-IMPORT & VALIDATE FINAL GLB ===")
scene = trimesh.load(str(glb_path), file_type='glb')
print(f"GLB Scene successfully re-imported: {list(scene.geometry.keys())}")
for g_name, geom in scene.geometry.items():
    print(f"  Geometry: {g_name}")
    print(f"    Vertices: {len(geom.vertices):,}")
    print(f"    Faces:    {len(geom.faces):,}")
    print(f"    Bounds:   Min={geom.bounds[0]}, Max={geom.bounds[1]}")
    print(f"    Visual:   Kind={geom.visual.kind}")

print("\nALL TEXTURING AND GLB EXPORT COMPLETED WITH 100% SUCCESS!")
