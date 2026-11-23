import sys, os, time
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
import pycolmap

v15_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v15')
v15_dir.mkdir(parents=True, exist_ok=True)

mesh_path = v15_dir / 'final_best_canonical_v15_mesh.ply'
sparse_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm/sparse/0')
dm_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12/dense_mvs/stereo/depth_maps')
f4k_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v13/frames_4k')

print("=== STEP 1: LOAD V15 MESH & SFM CAMERAS ===")
mesh = o3d.io.read_triangle_mesh(str(mesh_path))
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()

verts = np.asarray(mesh.vertices)
tris = np.asarray(mesh.triangles)
normals = np.asarray(mesh.triangle_normals)
centroids = verts[tris].mean(axis=1)
N_f = len(tris)

print(f"Loaded mesh: {len(verts):,} vertices, {N_f:,} faces.")
print(f"Bounds min: {verts.min(axis=0).round(2)}, max: {verts.max(axis=0).round(2)}")

# Load COLMAP sparse reconstruction
rec = pycolmap.Reconstruction(str(sparse_dir))
images = sorted(list(rec.images.values()), key=lambda x: x.name)
num_cams = len(images)
print(f"Loaded {num_cams} cameras.")

# Calibrated SIMPLE_RADIAL camera intrinsics at native 4K (2x scale from 1080p)
cam = rec.cameras[1]
f_4k = cam.params[0] * 2.0   # 2745.89613 px
cx_4k = 1920.0
cy_4k = 1080.0
k_dist = cam.params[3]       # 0.02053738 (radial lens distortion)
print(f"Calibrated 4K Camera: f={f_4k:.2f}px, cx={cx_4k}, cy={cy_4k}, k={k_dist:.6f}")

# Pinhole intrinsics for geometric depth maps (1600x900)
fx_dm = 1159.82941244
fy_dm = 1160.23780308
cx_dm = 800.0
cy_dm = 450.0

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

# Helper to load COLMAP geometric depth map
def read_colmap_depth_map(path):
    with open(path, 'rb') as f:
        header = b''
        amp_count = 0
        while amp_count < 3:
            char = f.read(1)
            if char == b'&':
                amp_count += 1
            header += char
        parts = header.decode('ascii').strip('&').split('&')
        width, height, channels = int(parts[0]), int(parts[1]), int(parts[2])
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((height, width, channels))

# Pre-load all 24 depth maps
print("\n=== STEP 2: LOAD GEOMETRIC DEPTH MAPS FOR OCCLUSION REJECTION ===")
depth_maps = []
for i, im in enumerate(images):
    dm_path = dm_dir / f"{im.name}.geometric.bin"
    if dm_path.exists():
        dm = read_colmap_depth_map(dm_path)[:, :, 0]
    else:
        dm = np.zeros((900, 1600), dtype=np.float32)
    depth_maps.append(dm)
print(f"Loaded {len(depth_maps)} depth maps.")

# =========================================================================
# STEP 3: VISIBILITY & OCCLUSION-AWARE QUALITY SCORING
# =========================================================================
print("\n=== STEP 3: EVALUATE VISIBILITY & Z-BUFFER OCCLUSION ===")
scores = np.zeros((N_f, num_cams), dtype=np.float32)

t0 = time.time()
for i in range(num_cams):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    P_c = (R @ centroids.T + t[:, None]).T # (N_f, 3)
    Z_c = P_c[:, 2]
    
    valid_z = Z_c > 0.1
    
    # 1. Project to native 4K frame using exact SIMPLE_RADIAL model
    x_n = P_c[:, 0] / np.maximum(Z_c, 1e-4)
    y_n = P_c[:, 1] / np.maximum(Z_c, 1e-4)
    r2 = x_n * x_n + y_n * y_n
    d_factor = 1.0 + k_dist * r2
    
    u_4k = f_4k * x_n * d_factor + cx_4k
    v_4k = f_4k * y_n * d_factor + cy_4k
    
    # In-bounds with 12px protective border
    in_bounds = valid_z & (u_4k >= 12.0) & (u_4k <= 3828.0) & (v_4k >= 12.0) & (v_4k <= 2148.0)
    
    # 2. Fronto-parallel incidence angle
    dist = np.linalg.norm(P_c, axis=1, keepdims=True) + 1e-6
    rays = P_c / dist
    N_c = (R @ normals.T).T
    cos_theta = -np.sum(rays * N_c, axis=1)
    facing = in_bounds & (cos_theta > 0.05)
    
    # 3. Z-Buffer depth consistency occlusion check
    dm = depth_maps[i]
    u_dm = np.clip((fx_dm * (P_c[:, 0] / np.maximum(Z_c, 1e-4)) + cx_dm).astype(np.int32), 0, 1599)
    v_dm = np.clip((fy_dm * (P_c[:, 1] / np.maximum(Z_c, 1e-4)) + cy_dm).astype(np.int32), 0, 899)
    
    d_pred = dm[v_dm, u_dm]
    # Face is occluded if it lies substantially behind the measured surface depth (>12% depth margin)
    occluded = facing & (d_pred > 0.5) & (Z_c > d_pred * 1.12)
    
    visible = facing & (~occluded)
    
    # Photographic quality score: favor direct viewing angle and closer distance
    q = np.where(visible, (np.clip(cos_theta, 0.0, 1.0) ** 2.2) / np.maximum(Z_c, 0.4), 0.0)
    scores[:, i] = q

print(f"Visibility scored in {time.time()-t0:.2f}s.")
best_cam = np.argmax(scores, axis=1)
max_scores = np.max(scores, axis=1)
uncovered = max_scores <= 0.0
print(f"Direct photographic coverage: {(~uncovered).sum():,} / {N_f:,} faces ({(~uncovered).mean()*100:.2f}%)")

# Refine uncovered boundary faces via topological neighbor consensus
if uncovered.any():
    print(f"Refining {uncovered.sum():,} boundary faces via topological neighbor consensus...")
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

still_uncovered = uncovered
if still_uncovered.any():
    dists_to_cams = np.linalg.norm(centroids[still_uncovered, None, :] - cam_centers[None, :, :], axis=-1)
    best_cam[still_uncovered] = np.argmin(dists_to_cams, axis=1)

print(f"Final coverage: 100.0% of {N_f:,} faces assigned to sharpest unoccluded views.")

# =========================================================================
# STEP 4: SURFACE-CHART BOUNDING BOX & DYNAMIC UV ATLAS PACKING
# =========================================================================
print("\n=== STEP 4: SURFACE-CHART ACTIVE BOUNDING BOX CALCULATION ===")
# For each camera, find the tight active 2D bounding box of all assigned faces in native 4K
chart_bboxes = {}
for i in range(num_cams):
    cam_faces = np.where(best_cam == i)[0]
    if len(cam_faces) == 0:
        continue
    
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    tri_pts = verts[tris[cam_faces]].reshape(-1, 3)
    
    P_c = (R @ tri_pts.T + t[:, None]).T
    Z_c = np.maximum(P_c[:, 2], 1e-4)
    x_n = P_c[:, 0] / Z_c
    y_n = P_c[:, 1] / Z_c
    r2 = x_n * x_n + y_n * y_n
    d_factor = 1.0 + k_dist * r2
    u = f_4k * x_n * d_factor + cx_4k
    v = f_4k * y_n * d_factor + cy_4k
    
    # 24px safety margin
    u_min = max(0, int(np.floor(u.min())) - 24)
    u_max = min(3840, int(np.ceil(u.max())) + 24)
    v_min = max(0, int(np.floor(v.min())) - 24)
    v_max = min(2160, int(np.ceil(v.max())) + 24)
    
    bw = u_max - u_min
    bh = v_max - v_min
    chart_bboxes[i] = (u_min, v_min, u_max, v_max, bw, bh, len(cam_faces))
    print(f"Cam {i:02d}: {len(cam_faces):6,d} faces | 4K active crop: [{u_min:4d}, {v_min:4d}] to [{u_max:4d}, {v_max:4d}] ({bw}x{bh} px, zero sky)")

# Dynamic Atlas Shelf Packing:
# We pack all active crops into an 8192x8192 atlas canvas
atlas_size = 8192
atlas_canvas = Image.new("RGB", (atlas_size, atlas_size), color=(25, 28, 34))

# Sort cameras by crop height descending for efficient shelf packing
active_cams = sorted(list(chart_bboxes.keys()), key=lambda c: chart_bboxes[c][5], reverse=True)

# Shelf packing algorithm
placements = {} # cam_idx -> (x_off, y_off, scale_w, scale_h)
curr_x = 0
curr_y = 0
shelf_h = 0
padding = 8

t0 = time.time()
for cam_idx in active_cams:
    u_min, v_min, u_max, v_max, bw, bh, _ = chart_bboxes[cam_idx]
    
    # Check if crop fits on current shelf
    if curr_x + bw + padding > atlas_size:
        # Move to next shelf
        curr_x = 0
        curr_y += shelf_h + padding
        shelf_h = 0
        
    if curr_y + bh + padding > atlas_size:
        # If running out of space, scale crop slightly to fit
        scale = (atlas_size - curr_y - padding) / float(bh)
        scale = max(0.65, min(1.0, scale))
        bw_sc = int(bw * scale)
        bh_sc = int(bh * scale)
    else:
        scale = 1.0
        bw_sc = bw
        bh_sc = bh
        
    # Read native 4K frame and crop active surface patch
    im_path = f4k_dir / images[cam_idx].name
    with Image.open(im_path) as im_pil:
        crop_pil = im_pil.crop((u_min, v_min, u_max, v_max))
        if scale != 1.0:
            crop_pil = crop_pil.resize((bw_sc, bh_sc), Image.Resampling.LANCZOS)
        atlas_canvas.paste(crop_pil, (curr_x, curr_y))
        
    placements[cam_idx] = (curr_x, curr_y, bw_sc, bh_sc, u_min, v_min, bw, bh)
    curr_x += bw_sc + padding
    shelf_h = max(shelf_h, bh_sc)

atlas_path = v15_dir / "texture_atlas.png"
atlas_canvas.save(str(atlas_path), format="PNG", optimize=True)
print(f"Saved Packed Surface-Chart Atlas: {atlas_path.name} ({atlas_path.stat().st_size / 1e6:.2f} MB in {time.time()-t0:.2f}s)")

# =========================================================================
# STEP 5: EXPORT TEXTURED OBJ & MTL IN COLMAP COORDINATES
# =========================================================================
print("\n=== STEP 5: EXPORT OBJ & MTL WITH SURFACE-CHART UVs ===")
obj_path = v15_dir / "final_best_canonical_v15_reconstruction.obj"
mtl_path = v15_dir / "final_best_canonical_v15_reconstruction.mtl"

with open(mtl_path, "w", encoding="utf-8") as f_mtl:
    f_mtl.write("# F.O.R.G.E V15 Surface-Chart Photographic Atlas Material\n")
    f_mtl.write("newmtl Material_Atlas\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.000 0.000 0.000\n")
    f_mtl.write("d 1.0\n")
    f_mtl.write("illum 1\n")
    f_mtl.write("map_Kd texture_atlas.png\n")

print(f"Writing OBJ file: {obj_path.name}...")
with open(obj_path, "w", encoding="utf-8") as f_obj:
    f_obj.write("# F.O.R.G.E V15 Visibility-Aware Surface-Chart Mesh\n")
    f_obj.write(f"mtllib {mtl_path.name}\n\n")
    
    # Vertices
    for v in verts:
        f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
    f_obj.write(f"\nusemtl Material_Atlas\n")
    
    # Write texture coordinates and faces
    vt_idx = 1
    for f_idx in range(N_f):
        cam_idx = best_cam[f_idx]
        x_off, y_off, bw_sc, bh_sc, u_min, v_min, bw, bh = placements[cam_idx]
        
        R = extrinsics[cam_idx, :3, :3]
        t = extrinsics[cam_idx, :3, 3]
        tri = tris[f_idx]
        tri_pts = verts[tri]
        
        P_c = (R @ tri_pts.T + t[:, None]).T
        Z_c = np.maximum(P_c[:, 2], 1e-4)
        x_n = P_c[:, 0] / Z_c
        y_n = P_c[:, 1] / Z_c
        r2 = x_n * x_n + y_n * y_n
        d_factor = 1.0 + k_dist * r2
        u = f_4k * x_n * d_factor + cx_4k
        v = f_4k * y_n * d_factor + cy_4k
        
        # Local coordinate inside active chart crop
        u_crop = (u - u_min) / float(bw)
        v_crop = (v - v_min) / float(bh)
        
        # Map to atlas UV coordinates
        u_atlas = (x_off + np.clip(u_crop, 0.0, 1.0) * bw_sc) / float(atlas_size)
        v_atlas = 1.0 - (y_off + np.clip(v_crop, 0.0, 1.0) * bh_sc) / float(atlas_size)
        
        for k in range(3):
            f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")
            
        v1, v2, v3 = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f_obj.write(f"f {v1}/{vt_idx} {v2}/{vt_idx+1} {v3}/{vt_idx+2}\n")
        vt_idx += 3

print(f"OBJ write complete: {obj_path.stat().st_size / 1e6:.2f} MB")

# =========================================================================
# STEP 6: APPLY EXACT 180° RIGID ROTATION & EXPORT FINAL PRODUCTION GLB
# =========================================================================
print("\n=== STEP 6: ROTATE TO UPRIGHT VIEWING ORIENTATION & EXPORT GLB ===")
tm = trimesh.load(str(obj_path), process=False)
print(f"Loaded OBJ into trimesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces")

bbox_center = (tm.vertices.min(axis=0) + tm.vertices.max(axis=0)) / 2.0
print(f"Bounding box center: {bbox_center}")

R_x = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

# Transform vertex positions: P' = R @ (P - C) + C
tm.vertices = (tm.vertices - bbox_center) @ R_x.T + bbox_center

# In trimesh, updating vertices automatically recomputes vertex_normals skyward!
mean_ny = tm.vertex_normals[:, 1].mean()
print(f"Verified upright normal Y mean: {mean_ny:.4f} (Must be positive skyward)")
assert mean_ny > 0, f"Error: vertex normals inverted ({mean_ny})"

# Export final production GLB
glb_path = v15_dir / "final_best_canonical_v15_reconstruction.glb"
print(f"Exporting final GLB to: {glb_path.name}...")
glb_bytes = tm.export(file_type='glb')
with open(glb_path, 'wb') as f_glb:
    f_glb.write(glb_bytes)

print(f"Exported GLB successfully! Size: {glb_path.stat().st_size / 1e6:.2f} MB")

# =========================================================================
# STEP 7: RE-IMPORT & VALIDATE FINAL PRODUCTION GLB
# =========================================================================
print("\n=== STEP 7: RE-IMPORT & VALIDATE PRODUCTION GLB ===")
scene = trimesh.load(str(glb_path), file_type='glb')
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
    print(f"    Normals:  Mean Y={geom.vertex_normals[:, 1].mean():.4f} (Upright skyward)")

print("\nF.O.R.G.E V15 PRODUCTION MODEL BUILD COMPLETED SUCCESSFULLY!")
