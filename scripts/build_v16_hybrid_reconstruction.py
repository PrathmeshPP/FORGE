import sys, os, time
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
import pycolmap

v14_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v14')
v15_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v15')
v16_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v16')
v16_dir.mkdir(parents=True, exist_ok=True)

sparse_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10/pilot_sfm/sparse/0')
dense_mvs_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12/dense_mvs')
dm_dir = dense_mvs_dir / 'stereo/depth_maps'
f4k_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v13/frames_4k')
fused_pcd_path = dense_mvs_dir / 'fused.ply'

mesh_v14_path = v14_dir / 'final_best_canonical_v14_mesh.ply'
mesh_v15_path = v15_dir / 'final_best_canonical_v15_mesh.ply'

print("=========================================================================")
print("=== F.O.R.G.E V16: BUILD HYBRID RECONSTRUCTION & PHOTOGRAPHIC TEXTURE ===")
print("=========================================================================")

# -------------------------------------------------------------------------
# STEP 1: LOAD ASSETS & CONSTRUCT CANDIDATE B HYBRID GEOMETRY
# -------------------------------------------------------------------------
print("\n=== STEP 1: LOAD ASSETS & CONSTRUCT CANDIDATE B HYBRID GEOMETRY ===")
m14 = trimesh.load(str(mesh_v14_path), process=False)
m15 = trimesh.load(str(mesh_v15_path), process=False)
pcd = trimesh.load(str(fused_pcd_path))

print(f"Loaded V14 Recognisable Core: {len(m14.vertices):,} vertices, {len(m14.faces):,} faces")
print(f"Loaded V15 Extended Mesh:    {len(m15.vertices):,} vertices, {len(m15.faces):,} faces")
print(f"Loaded Fused Point Cloud:    {len(pcd.vertices):,} points")

b14_min = m14.bounds[0]
b14_max = m14.bounds[1]
tree_pcd = cKDTree(np.asarray(pcd.vertices))

# Analyze V15 faces to extract strictly supported, non-spiky extension
v15 = np.asarray(m15.vertices)
f15 = np.asarray(m15.faces)
cens15 = v15[f15].mean(axis=1)

e1 = np.linalg.norm(v15[f15[:, 0]] - v15[f15[:, 1]], axis=1)
e2 = np.linalg.norm(v15[f15[:, 1]] - v15[f15[:, 2]], axis=1)
e3 = np.linalg.norm(v15[f15[:, 2]] - v15[f15[:, 0]], axis=1)
max_e = np.maximum(np.maximum(e1, e2), e3)
dists15, _ = tree_pcd.query(cens15, k=1)
norms15 = m15.face_normals

# Rigorous multi-view and geometric support masks
# Northern Road Corridor (Z > 28.27m):
north_mask = (cens15[:, 2] > b14_max[2]) & (cens15[:, 2] <= 42.0) & \
             (cens15[:, 0] >= -5.0) & (cens15[:, 0] <= 30.0) & \
             (cens15[:, 1] >= -5.0) & (cens15[:, 1] <= 2.0) & \
             (dists15 <= 0.25) & (max_e <= 0.30) & (norms15[:, 1] < 0.1)

# Eastern Courtyard / Landscape (X > 26.17m):
east_mask = (cens15[:, 0] > b14_max[0]) & (cens15[:, 0] <= 33.0) & \
            (cens15[:, 2] >= 2.0) & (cens15[:, 2] <= b14_max[2]) & \
            (cens15[:, 1] >= -5.0) & (cens15[:, 1] <= 2.0) & \
            (dists15 <= 0.25) & (max_e <= 0.30) & (norms15[:, 1] < 0.1)

ext_faces = f15[north_mask | east_mask]
m_ext = trimesh.Trimesh(vertices=v15, faces=ext_faces, process=True)
print(f"Extracted Supported Extension: {len(m_ext.vertices):,} vertices, {len(m_ext.faces):,} faces")

# Combine V14 and extension
nV14 = len(m14.vertices)
all_v = np.vstack([m14.vertices, m_ext.vertices])
all_f = [m14.faces, m_ext.faces + nV14]

# Interface Seam Stitching:
# Find boundary vertices of V14 near interface
bnd_edges_14 = m14.edges[trimesh.grouping.group_rows(m14.edges_sorted, require_count=1)]
bnd_v14 = np.unique(bnd_edges_14)
pts_bnd_14 = m14.vertices[bnd_v14]
mask_if_14 = (pts_bnd_14[:, 2] > b14_max[2] - 0.5) | (pts_bnd_14[:, 0] > b14_max[0] - 0.5)
if_idx_14 = bnd_v14[mask_if_14]

# Find boundary vertices of extension near interface
bnd_edges_ext = m_ext.edges[trimesh.grouping.group_rows(m_ext.edges_sorted, require_count=1)]
bnd_ext = np.unique(bnd_edges_ext)
pts_bnd_ext = m_ext.vertices[bnd_ext]
mask_if_ext = (pts_bnd_ext[:, 2] < b14_max[2] + 0.6) | (pts_bnd_ext[:, 0] < b14_max[0] + 0.6)
if_idx_ext = bnd_ext[mask_if_ext]

tree_if_ext = cKDTree(m_ext.vertices[if_idx_ext])
stitch_faces = []
for i_14 in if_idx_14:
    pt = m14.vertices[i_14]
    d, nearest_local = tree_if_ext.query(pt, k=2)
    if d[0] < 0.8:
        i_ext_0 = if_idx_ext[nearest_local[0]] + nV14
        i_ext_1 = if_idx_ext[nearest_local[1]] + nV14
        stitch_faces.append([i_14, i_ext_0, i_ext_1])

if len(stitch_faces) > 0:
    all_f.append(np.array(stitch_faces))
print(f"Stitched Seam: Added {len(stitch_faces)} transitional connection triangles.")

final_faces = np.vstack(all_f)
m_v16 = trimesh.Trimesh(vertices=all_v, faces=final_faces, process=True)

# Save V16 neutral mesh
mesh_v16_path = v16_dir / 'final_best_canonical_v16_mesh.ply'
m_v16.export(str(mesh_v16_path))
print(f"Saved Candidate B Mesh: {mesh_v16_path.name}")
print(f"  Vertices: {len(m_v16.vertices):,}, Faces: {len(m_v16.faces):,}")
print(f"  Bounds Min: {m_v16.bounds[0].round(2)}, Max: {m_v16.bounds[1].round(2)}")
print(f"  Extents:    X={m_v16.extents[0]:.2f}m, Y={m_v16.extents[1]:.2f}m, Z={m_v16.extents[2]:.2f}m")

# Save supporting point cloud
pcd_v16_path = v16_dir / 'final_best_canonical_v16_point_cloud.ply'
pcd.export(str(pcd_v16_path))
print(f"Saved Supporting Point Cloud: {pcd_v16_path.name} ({len(pcd.vertices):,} points)")

# -------------------------------------------------------------------------
# STEP 2: LOAD CAMERAS & GEOMETRIC DEPTH MAPS
# -------------------------------------------------------------------------
print("\n=== STEP 2: LOAD CAMERAS & GEOMETRIC DEPTH MAPS ===")
rec = pycolmap.Reconstruction(str(sparse_dir))
images = sorted(list(rec.images.values()), key=lambda x: x.name)
num_cams = len(images)

cam = rec.cameras[1]
f_4k = cam.params[0] * 2.0   # 2745.89613 px
cx_4k = 1920.0
cy_4k = 1080.0
k_dist = cam.params[3]       # 0.02053738
print(f"Loaded {num_cams} cameras. Calibrated 4K Camera: f={f_4k:.2f}px, cx={cx_4k}, cy={cy_4k}, k={k_dist:.6f}")

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

depth_maps = []
for i, im in enumerate(images):
    dm_path = dm_dir / f"{im.name}.geometric.bin"
    if dm_path.exists():
        dm = read_colmap_depth_map(dm_path)[:, :, 0]
    else:
        dm = np.zeros((900, 1600), dtype=np.float32)
    depth_maps.append(dm)
print(f"Loaded {len(depth_maps)} geometric depth maps for Z-buffer occlusion gating.")

# -------------------------------------------------------------------------
# STEP 3: VISIBILITY & OCCLUSION-AWARE PROJECTION SCORING
# -------------------------------------------------------------------------
print("\n=== STEP 3: VISIBILITY & OCCLUSION-AWARE PROJECTION SCORING ===")
N_f = len(m_v16.faces)
verts = np.asarray(m_v16.vertices)
tris = np.asarray(m_v16.faces)
normals = np.asarray(m_v16.face_normals)
centroids = verts[tris].mean(axis=1)

scores = np.zeros((N_f, num_cams), dtype=np.float32)
t0 = time.time()

for i in range(num_cams):
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    C = cam_centers[i]
    dm = depth_maps[i]
    
    view_vecs = C - centroids
    dist = np.linalg.norm(view_vecs, axis=1)
    view_dir = view_vecs / (dist[:, None] + 1e-8)
    
    cos_theta = np.sum(normals * view_dir, axis=1)
    valid_facing = cos_theta > 0.05
    
    P_c = (R @ centroids.T).T + t
    Z_c = P_c[:, 2]
    valid_front = Z_c > 0.5
    
    # SIMPLE_RADIAL projection to 4K
    x_n = P_c[:, 0] / Z_c
    y_n = P_c[:, 1] / Z_c
    r2 = x_n**2 + y_n**2
    radial = 1.0 + k_dist * r2
    x_d = x_n * radial
    y_d = y_n * radial
    
    u_4k = f_4k * x_d + cx_4k
    v_4k = f_4k * y_d + cy_4k
    valid_in_4k = (u_4k >= 10) & (u_4k <= 3830) & (v_4k >= 10) & (v_4k <= 2150)
    
    # Depth map projection
    u_dm = fx_dm * x_n + cx_dm
    v_dm = fy_dm * y_n + cy_dm
    valid_in_dm = (u_dm >= 0) & (u_dm < 1600) & (v_dm >= 0) & (v_dm < 900)
    
    u_idx = np.clip(np.round(u_dm).astype(int), 0, 1599)
    v_idx = np.clip(np.round(v_dm).astype(int), 0, 899)
    d_pred = dm[v_idx, u_idx]
    
    # True Z-buffer occlusion gate:
    not_occluded = (d_pred <= 0.0) | (Z_c <= d_pred * 1.12)
    
    active_mask = valid_facing & valid_front & valid_in_4k & valid_in_dm & not_occluded
    scores[active_mask, i] = cos_theta[active_mask] / (dist[active_mask] + 1e-4)

print(f"Scored {N_f:,} faces across 24 cameras in {time.time()-t0:.2f}s.")

best_cam_idx = np.argmax(scores, axis=1)
max_score = np.max(scores, axis=1)
no_view_mask = max_score == 0

if np.any(no_view_mask):
    c_fallback = centroids[no_view_mask]
    dists_to_cams = np.linalg.norm(c_fallback[:, None, :] - cam_centers[None, :, :], axis=2)
    best_cam_idx[no_view_mask] = np.argmin(dists_to_cams, axis=1)

# -------------------------------------------------------------------------
# STEP 4: COMPUTE ACTIVE SURFACE-CHART BOUNDING BOXES & 4x6 GRID PACKING
# -------------------------------------------------------------------------
print("\n=== STEP 4: COMPUTE ACTIVE SURFACE-CHART BOUNDING BOXES & PACK ATLAS ===")
# 4 columns x 6 rows = 24 camera charts
atlas_size = 8192
n_cols = 4
n_rows = 6
slot_w = atlas_size // n_cols  # 2048 px
slot_h = atlas_size // n_rows  # 1365 px

cam_layout = {}
print(f"Atlas Grid: {n_cols} cols x {n_rows} rows | Slot size: {slot_w}x{slot_h} px")

atlas_img = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)

for i in range(num_cams):
    f_indices = np.where(best_cam_idx == i)[0]
    if len(f_indices) == 0:
        continue
    
    t_assigned = tris[f_indices]
    v_assigned_idx = np.unique(t_assigned)
    pts = verts[v_assigned_idx]
    
    R = extrinsics[i, :3, :3]
    t = extrinsics[i, :3, 3]
    P_c = (R @ pts.T).T + t
    Z_c = np.maximum(P_c[:, 2], 0.1)
    
    x_n = P_c[:, 0] / Z_c
    y_n = P_c[:, 1] / Z_c
    r2 = x_n**2 + y_n**2
    radial = 1.0 + k_dist * r2
    u = f_4k * (x_n * radial) + cx_4k
    v = f_4k * (y_n * radial) + cy_4k
    
    # Active 4K surface bounding box (excluding sky above v_min)
    u_min = max(0, int(np.floor(np.percentile(u, 0.5))) - 20)
    u_max = min(3840, int(np.ceil(np.percentile(u, 99.5))) + 20)
    v_min = max(0, int(np.floor(np.percentile(v, 0.5))) - 20)
    v_max = min(2160, int(np.ceil(np.percentile(v, 99.5))) + 20)
    
    bw = max(100, u_max - u_min)
    bh = max(100, v_max - v_min)
    
    col = i % n_cols
    row = i // n_cols
    slot_x = col * slot_w
    slot_y = row * slot_h
    
    # Scale active patch to fit slot while preserving aspect ratio
    scale_w = slot_w / float(bw)
    scale_h = slot_h / float(bh)
    scale = min(scale_w, scale_h)
    
    tw = int(bw * scale)
    th = int(bh * scale)
    ox = slot_x + (slot_w - tw) // 2
    oy = slot_y + (slot_h - th) // 2
    
    cam_layout[i] = {
        'x': ox, 'y': oy,
        'w': tw, 'h': th,
        'scale_x': tw / float(bw),
        'scale_y': th / float(bh),
        'crop': (u_min, v_min, u_max, v_max),
        'bw': bw, 'bh': bh
    }
    
    # Load 4K frame and paste active crop
    im_name = images[i].name
    frame_path = f4k_dir / f"{Path(im_name).stem}.png"
    if not frame_path.exists():
        cand = list(f4k_dir.glob(f"*{Path(im_name).stem}*"))
        if len(cand) > 0: frame_path = cand[0]
        else: frame_path = list(f4k_dir.glob("*.png"))[i]
        
    src_bgr = cv2.imread(str(frame_path))
    crop = src_bgr[v_min:v_max, u_min:u_max]
    crop_res = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
    atlas_img[oy:oy+th, ox:ox+tw] = crop_res
    print(f"  Cam {i:2d} ({im_name}): Crop [{u_min:4d}, {v_min:4d}, {bw}x{bh}px] -> Slot [{ox:4d}, {oy:4d}, {tw}x{th}px] | {len(f_indices):,} faces")

atlas_path = v16_dir / 'texture_atlas.png'
cv2.imwrite(str(atlas_path), atlas_img)
print(f"Saved Packed 8192x8192 Atlas: {atlas_path.name} ({atlas_path.stat().st_size / 1e6:.2f} MB)")

# -------------------------------------------------------------------------
# STEP 5: WRITE EXPLICIT UV INTERCHANGE OBJ
# -------------------------------------------------------------------------
print("\n=== STEP 5: WRITE EXPLICIT UV INTERCHANGE OBJ ===")
obj_path = v16_dir / 'final_best_canonical_v16_reconstruction.obj'
mtl_path = v16_dir / 'final_best_canonical_v16_reconstruction.mtl'

with open(mtl_path, 'w') as f_mtl:
    f_mtl.write("newmtl material_0\n")
    f_mtl.write("Ka 1.000 1.000 1.000\n")
    f_mtl.write("Kd 1.000 1.000 1.000\n")
    f_mtl.write("Ks 0.050 0.050 0.050\n")
    f_mtl.write("d 1.0\n")
    f_mtl.write("illum 2\n")
    f_mtl.write("map_Kd texture_atlas.png\n")

with open(obj_path, 'w') as f_obj:
    f_obj.write("# F.O.R.G.E V16 Canonical Hybrid Model\n")
    f_obj.write(f"mtllib {mtl_path.name}\n")
    
    for v in verts:
        f_obj.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
    f_obj.write("usemtl material_0\n")
    vt_idx = 1
    
    for fi in range(N_f):
        c_idx = best_cam_idx[fi]
        lay = cam_layout[c_idx]
        tri = tris[fi]
        P_tri = verts[tri]
        
        R = extrinsics[c_idx, :3, :3]
        t = extrinsics[c_idx, :3, 3]
        P_c = (R @ P_tri.T).T + t
        Z_c = np.maximum(P_c[:, 2], 0.1)
        
        x_n = P_c[:, 0] / Z_c
        y_n = P_c[:, 1] / Z_c
        r2 = x_n**2 + y_n**2
        radial = 1.0 + k_dist * r2
        u = f_4k * (x_n * radial) + cx_4k
        v = f_4k * (y_n * radial) + cy_4k
        
        u0, v0, u1, v1 = lay['crop']
        u_crop = np.clip(u - u0, 0, lay['bw'])
        v_crop = np.clip(v - v0, 0, lay['bh'])
        
        px_atlas = lay['x'] + u_crop * lay['scale_x']
        py_atlas = lay['y'] + v_crop * lay['scale_y']
        
        u_atlas = px_atlas / float(atlas_size)
        v_atlas = 1.0 - py_atlas / float(atlas_size)
        
        u_atlas = np.clip(u_atlas, 0.0, 1.0)
        v_atlas = np.clip(v_atlas, 0.0, 1.0)
        
        for k in range(3):
            f_obj.write(f"vt {u_atlas[k]:.6f} {v_atlas[k]:.6f}\n")
            
        v1_i, v2_i, v3_i = tri[0] + 1, tri[1] + 1, tri[2] + 1
        f_obj.write(f"f {v1_i}/{vt_idx} {v2_i}/{vt_idx+1} {v3_i}/{vt_idx+2}\n")
        vt_idx += 3

print(f"OBJ write complete: {obj_path.stat().st_size / 1e6:.2f} MB")

# -------------------------------------------------------------------------
# STEP 6: APPLY RIGID 180-DEGREE ROTATION AROUND CANONICAL CENTER & EXPORT GLB
# -------------------------------------------------------------------------
print("\n=== STEP 6: APPLY RIGID DISPLAY ROTATION AROUND CANONICAL CENTER & EXPORT GLB ===")
tm = trimesh.load(str(obj_path), process=False)
print(f"Loaded OBJ into trimesh: {len(tm.vertices):,} vertices, {len(tm.faces):,} faces")

# CANONICAL rotation center (exact building core center from V14)
c_canon = np.array([7.18673944, -0.8999939, 12.86114335], dtype=np.float64)
print(f"Using Canonical Rotation Center: {c_canon}")

R_x = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=np.float64)

# Apply rigid transformation
tm.vertices = (tm.vertices - c_canon) @ R_x.T + c_canon
mean_ny = tm.vertex_normals[:, 1].mean()
print(f"Verified Upright Normal Y Mean: {mean_ny:.4f} (Skyward = Positive)")
assert mean_ny > 0, f"Error: normals inverted ({mean_ny})"

glb_path = v16_dir / 'final_best_canonical_v16_reconstruction.glb'
print(f"Exporting final production GLB to: {glb_path}...")
glb_bytes = tm.export(file_type='glb')
with open(glb_path, 'wb') as f_glb:
    f_glb.write(glb_bytes)
print(f"Exported GLB successfully! Size: {glb_path.stat().st_size / 1e6:.2f} MB")

# -------------------------------------------------------------------------
# STEP 7: INDEPENDENT RE-IMPORT & VALIDATION
# -------------------------------------------------------------------------
print("\n=== STEP 7: INDEPENDENT RE-IMPORT & VALIDATION ===")
scene = trimesh.load(str(glb_path), file_type='glb')
for g_name, geom in scene.geometry.items():
    print(f"  Geometry: {g_name}")
    print(f"    Vertices: {len(geom.vertices):,}")
    print(f"    Faces:    {len(geom.faces):,}")
    print(f"    Bounds:   Min={geom.bounds[0].round(2)}, Max={geom.bounds[1].round(2)}")
    print(f"    Span:     X={geom.bounds[1][0]-geom.bounds[0][0]:.2f}m, Y={geom.bounds[1][1]-geom.bounds[0][1]:.2f}m, Z={geom.bounds[1][2]-geom.bounds[0][2]:.2f}m")
    print(f"    Normals:  Mean Y={geom.vertex_normals[:, 1].mean():.4f} (Upright skyward)")

print("\nF.O.R.G.E V16 HYBRID RECONSTRUCTION BUILD COMPLETE!")
