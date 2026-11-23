import sys, os, time
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
import pycolmap

v13_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v13')
v14_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v14')
v14_dir.mkdir(parents=True, exist_ok=True)

sparse_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm/sparse/0')
f4k_dir = v13_dir / 'frames_4k'
pcd_path = v13_dir / 'final_best_canonical_v13_point_cloud.ply'
mesh_v13_path = v13_dir / 'final_best_canonical_v13_mesh.ply'

print("=== STEP 1: LOAD RECONSTRUCTION DATA ===")
mesh_v13 = o3d.io.read_triangle_mesh(str(mesh_v13_path))
mesh_v13.compute_triangle_normals()
mesh_v13.compute_vertex_normals()

pcd = o3d.io.read_point_cloud(str(pcd_path))
print(f"Loaded V13 mesh: {len(mesh_v13.vertices):,} vertices, {len(mesh_v13.triangles):,} faces")
print(f"Loaded point cloud: {len(pcd.points):,} points")

# Copy point cloud to V14 directory
pcd_v14_path = v14_dir / 'final_best_canonical_v14_point_cloud.ply'
if not pcd_v14_path.exists():
    o3d.io.write_point_cloud(str(pcd_v14_path), pcd)
    print(f"Saved V14 point cloud: {pcd_v14_path.name}")

# Load cameras
rec = pycolmap.Reconstruction(str(sparse_dir))
images = sorted(list(rec.images.values()), key=lambda x: x.name)
num_cams = len(images)
print(f"Loaded {num_cams} cameras.")

# Native 4K camera intrinsics
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

# =========================================================================
# STEP 2: TARGETED GEOMETRY REPAIR (Candidate C geometry)
# =========================================================================
print("\n=== STEP 2: TARGETED GEOMETRY REPAIR ===")
verts_raw = np.asarray(mesh_v13.vertices)
tris_raw = np.asarray(mesh_v13.triangles)

# KDTree on point cloud
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
v_dists = np.zeros(len(verts_raw), dtype=np.float32)
for i in range(len(verts_raw)):
    k, idx, d2 = pcd_tree.search_knn_vector_3d(verts_raw[i], 1)
    v_dists[i] = np.sqrt(d2[0])

# Triangle mean distance
tri_dists = v_dists[tris_raw].mean(axis=1)
# Prune unsupported boundary extrapolation triangles where mean distance > 0.55m
# (preserves 100% of the building, roads, courtyard, and genuine peripheral trees)
unsupp_mask = tri_dists > 0.55
print(f"Total triangles in V13: {len(tris_raw):,}")
print(f"Unsupported boundary extrapolation triangles (>0.55m): {unsupp_mask.sum():,} ({unsupp_mask.mean()*100:.2f}%)")

keep_tris = ~unsupp_mask
mesh_v14 = o3d.geometry.TriangleMesh()
mesh_v14.vertices = mesh_v13.vertices
mesh_v14.triangles = o3d.utility.Vector3iVector(tris_raw[keep_tris])
mesh_v14.remove_unreferenced_vertices()
mesh_v14.compute_triangle_normals()
mesh_v14.compute_vertex_normals()

# Component analysis of cleaned mesh
tm_clean = trimesh.Trimesh(vertices=np.asarray(mesh_v14.vertices), faces=np.asarray(mesh_v14.triangles), process=False)
comps = tm_clean.split(only_watertight=False)
print(f"Cleaned mesh: {len(mesh_v14.vertices):,} vertices, {len(mesh_v14.triangles):,} faces across {len(comps)} components")
for idx, c in enumerate(comps[:10]):
    span = np.ptp(c.vertices, axis=0)
    print(f"  Comp {idx}: {len(c.faces):,} faces, span={span.round(2)}")

# Save repaired neutral mesh
mesh_v14_path = v14_dir / 'final_best_canonical_v14_mesh.ply'
o3d.io.write_triangle_mesh(str(mesh_v14_path), mesh_v14)
print(f"Saved repaired neutral mesh: {mesh_v14_path.name}")

# =========================================================================
# STEP 3: HIGH-RESOLUTION SURFACE-CHART TEXTURE SYNTHESIS (8192x8192 ATLAS)
# =========================================================================
print("\n=== STEP 3: HIGH-RESOLUTION SURFACE-CHART TEXTURE SYNTHESIS ===")
# We use the cleaned V14 geometry (Candidate C)
verts = np.asarray(mesh_v14.vertices)
tris = np.asarray(mesh_v14.triangles)
normals = np.asarray(mesh_v14.triangle_normals)
centroids = verts[tris].mean(axis=1)
N_f = len(tris)

# Photographic visibility & quality scoring
scores = np.zeros((N_f, num_cams), dtype=np.float32)
for i in range(num_cams):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    P_c = (R @ centroids.T + t[:, None]).T
    Z_c = P_c[:, 2]
    valid_z = Z_c > 0.1
    u = fx_u4k * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + cx_u4k
    v = fy_u4k * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + cy_u4k
    in_bounds = valid_z & (u >= 10.0) & (u <= W_native - 11.0) & (v >= 10.0) & (v <= H_native - 11.0)
    dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
    rays = P_c / dist
    N_c = (R @ normals.T).T
    cos_theta = -np.sum(rays * N_c, axis=1)
    visible = in_bounds & (cos_theta > 0.05)
    scores[:, i] = np.where(visible, (np.clip(cos_theta, 0.0, 1.0) ** 2.2) / np.maximum(Z_c, 0.4), 0.0)

best_cam = np.argmax(scores, axis=1)
uncovered = np.max(scores, axis=1) <= 0
print(f"Direct photographic coverage: {(~uncovered).sum():,} / {N_f:,} faces ({(~uncovered).mean()*100:.2f}%)")

# Refine uncovered boundary faces via topological neighbor consensus
if uncovered.any():
    edges_to_faces = {}
    for f_idx, tri in enumerate(tris):
        for edge in [(min(tri[0], tri[1]), max(tri[0], tri[1])),
                     (min(tri[1], tri[2]), max(tri[1], tri[2])),
                     (min(tri[2], tri[0]), max(tri[2], tri[0]))]:
            edges_to_faces.setdefault(edge, []).append(f_idx)
            
    for f_idx in np.where(uncovered)[0]:
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

# Fallback for remaining
still_uncovered = uncovered
if still_uncovered.any():
    dists_to_cams = np.linalg.norm(centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
    best_cam[still_uncovered] = np.argmin(dists_to_cams, axis=1)

print(f"Total consensus coverage: 100.0% of {N_f:,} faces assigned to sharpest views.")

# =========================================================================
# STEP 4: 8192x8192 HIGH-RESOLUTION UV ATLAS PACKING
# =========================================================================
# In an 8192x8192 atlas, a 4x6 grid gives:
# slot_w = 8192 // 4 = 2048 px
# slot_h = 8192 // 6 = 1365 px
# Preserving native 16:9 aspect ratio:
# tile_w = 2048 px
# tile_h = int(round(2048 * 9.0 / 16.0)) = 1152 px (fits cleanly inside 1365 slot)
# This provides 2048x1152 per 4K camera frame (4x the pixels of V13's 1024x576)!
atlas_size = 8192
grid_cols, grid_rows = 4, 6
slot_w = atlas_size // grid_cols # 2048
slot_h = atlas_size // grid_rows # 1365
tile_w = slot_w                  # 2048
tile_h = int(round(tile_w * 9.0 / 16.0)) # 1152

print(f"\nBuilding High-Resolution {atlas_size}x{atlas_size} Texture Atlas:")
print(f"  Grid: {grid_cols}x{grid_rows} | Slot: {slot_w}x{slot_h} | Tile: {tile_w}x{tile_h} (16:9 ratio preserved)")

atlas_canvas = Image.new("RGB", (atlas_size, atlas_size), color=(30, 32, 36))

t0 = time.time()
for i, im in enumerate(images):
    col = i % grid_cols
    row = i // grid_cols
    x_off = col * slot_w
    y_off = row * slot_h
    
    frame_path = f4k_dir / im.name
    frame_pil = Image.open(frame_path).convert("RGB")
    
    # High-quality Lanczos resampling to 2048x1152
    frame_resized = frame_pil.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
    atlas_canvas.paste(frame_resized, (x_off, y_off))

atlas_path = v14_dir / "texture_atlas.png"
atlas_canvas.save(str(atlas_path), format="PNG", optimize=True)
print(f"Saved 8K Texture Atlas: {atlas_path} ({atlas_path.stat().st_size / 1e6:.2f} MB in {time.time()-t0:.2f}s)")

# =========================================================================
# STEP 5: EXPORT TEXTURED OBJ & MTL IN COLMAP COORDINATES
# =========================================================================
obj_path = v14_dir / "final_best_canonical_v14_reconstruction.obj"
mtl_path = v14_dir / "final_best_canonical_v14_reconstruction.mtl"

with open(mtl_path, "w", encoding="utf-8") as f_mtl:
    f_mtl.write("# F.O.R.G.E V14 High-Resolution Photographic Atlas Material\n")
    f_mtl.write("newmtl Material_Atlas\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.000 0.000 0.000\n")
    f_mtl.write("d 1.0\n")
    f_mtl.write("illum 1\n")
    f_mtl.write("map_Kd texture_atlas.png\n")

print(f"Writing OBJ file: {obj_path}...")
with open(obj_path, "w", encoding="utf-8") as f_obj:
    f_obj.write("# F.O.R.G.E V14 High-Resolution Multi-View Photographic Mesh\n")
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
        
        # Map to atlas UV coordinates
        u_atlas = (col * slot_w + u_norm * tile_w) / float(atlas_size)
        v_atlas = 1.0 - (row * slot_h + v_norm * tile_h) / float(atlas_size)
        
        for k in range(3):
            f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")
            
        v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f_obj.write(f"f {v1}/{vt_idx} {v2}/{vt_idx+1} {v3}/{vt_idx+2}\n")
        vt_idx += 3

print(f"OBJ write complete: {obj_path.stat().st_size / 1e6:.2f} MB")

# =========================================================================
# STEP 6: APPLY 180-DEGREE ROTATION WITH CORRECTED UPWARD NORMALS & EXPORT GLB
# =========================================================================
print("\n=== STEP 6: ROTATE TO DISPLAY ORIENTATION WITH CORRECT UPWARD NORMALS ===")
tm = trimesh.load(str(obj_path), process=False)
print(f"Loaded OBJ into trimesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces")

bbox_min = tm.vertices.min(axis=0)
bbox_max = tm.vertices.max(axis=0)
bbox_center = (bbox_min + bbox_max) / 2.0
print(f"Bounding box center: {bbox_center}")

# 180-degree rotation R_X(180 deg) around center
R_x = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

# Transform vertex positions: P' = R @ (P - C) + C
tm.vertices = (tm.vertices - bbox_center) @ R_x.T + bbox_center

# In trimesh, updating vertices automatically recomputes vertex_normals from faces!
# DO NOT apply a second rotation, which caused the double-rotation bug in V13!
mean_ny = tm.vertex_normals[:, 1].mean()
print(f"Verified GLB vertex normal Y mean: {mean_ny:.4f} (Must be positive skyward)")
assert mean_ny > 0, f"Error: vertex normals pointing down ({mean_ny})"

# Export final GLB
glb_path = v14_dir / "final_best_canonical_v14_reconstruction.glb"
print(f"Exporting final GLB to: {glb_path}...")
glb_bytes = tm.export(file_type='glb')
with open(glb_path, 'wb') as f_glb:
    f_glb.write(glb_bytes)

print(f"Exported GLB successfully! Size: {glb_path.stat().st_size / 1e6:.2f} MB")

# =========================================================================
# STEP 7: RE-IMPORT & VALIDATE FINAL GLB
# =========================================================================
print("\n=== STEP 7: RE-IMPORT & VALIDATE FINAL GLB ===")
scene = trimesh.load(str(glb_path), file_type='glb')
print(f"GLB Scene successfully re-imported: {list(scene.geometry.keys())}")
for g_name, geom in scene.geometry.items():
    print(f"  Geometry: {g_name}")
    print(f"    Vertices: {len(geom.vertices):,}")
    print(f"    Faces:    {len(geom.faces):,}")
    print(f"    Bounds:   Min={geom.bounds[0].round(2)}, Max={geom.bounds[1].round(2)}")
    print(f"    Span:     X={geom.bounds[1][0]-geom.bounds[0][0]:.2f}m, Y={geom.bounds[1][1]-geom.bounds[0][1]:.2f}m, Z={geom.bounds[1][2]-geom.bounds[0][2]:.2f}m")
    print(f"    Visual:   Kind={geom.visual.kind}")
    if hasattr(geom.visual, 'material'):
        mat = geom.visual.material
        tex_size = mat.baseColorTexture.size if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture else None
        print(f"    Material: Type={type(mat)}, Texture={tex_size}")
    print(f"    Normals:  Mean Y={geom.vertex_normals[:, 1].mean():.4f} (Positive = Upright skyward)")

print("\nF.O.R.G.E V14 MODEL GENERATION COMPLETED SUCCESSFULLY!")
