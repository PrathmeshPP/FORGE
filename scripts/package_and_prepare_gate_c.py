"""
Package Gate C models and point clouds:
1. Generate TSDF-fused point clouds (tsdf_pcd_*.ply) using Open3D ScalableTSDFVolume.extract_point_cloud()
2. Export GLB meshes with vertex colors for all 3 treatments:
   - BASELINE — DEFECTIVE
   - CANDIDATE A — CORRECTED SIM(3)
   - CANDIDATE B — RIGID SE(3)
3. Register into viewer/public/model-manifest.json and viewer/src/data/model-manifest.json
"""

import sys
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXP_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261031_142303"
INVEST_DIR = PROJECT_ROOT / "outputs/indoor/distortion_investigation_20261031"
GATE_C_DIR = INVEST_DIR / "gate_c_extended"
CACHE_DIR = GATE_C_DIR / "vggt_cache"

report = json.loads((EXP_DIR / "frame_curation_report.json").read_text())
frame_records = report["frames"][:36]
image_paths = [r["file_path"] for r in frame_records]

cached_raw = np.load(CACHE_DIR / "raw_windows_0_to_10.npz")
summary = json.loads((GATE_C_DIR / "gate_c_extended_summary.json").read_text())

candidates = [
    {
        "key": "baseline_defective",
        "display_name": "BASELINE — DEFECTIVE",
        "label": "Defective Clamped Scale (Unrecomputed Translation, 2.54m Drift)",
        "mesh_ply": GATE_C_DIR / "tsdf_mesh_baseline_defective.ply",
        "mesh_glb": GATE_C_DIR / "tsdf_mesh_baseline_defective.glb",
        "tsdf_pcd_ply": GATE_C_DIR / "tsdf_pcd_baseline_defective.ply",
        "pre_pcd_ply": GATE_C_DIR / "pre_tsdf_pcd_baseline_defective.ply",
    },
    {
        "key": "candidate_A_bounded_sim3",
        "display_name": "CANDIDATE A — CORRECTED SIM(3)",
        "label": "Corrected Bounded Sim(3) + Mandatory Translation Recomputation",
        "mesh_ply": GATE_C_DIR / "tsdf_mesh_candidate_A_bounded_sim3.ply",
        "mesh_glb": GATE_C_DIR / "tsdf_mesh_candidate_A_bounded_sim3.glb",
        "tsdf_pcd_ply": GATE_C_DIR / "tsdf_pcd_candidate_A_bounded_sim3.ply",
        "pre_pcd_ply": GATE_C_DIR / "pre_tsdf_pcd_candidate_A_bounded_sim3.ply",
    },
    {
        "key": "candidate_B_rigid_se3",
        "display_name": "CANDIDATE B — RIGID SE(3)",
        "label": "Rigid SE(3) s=1.0 + Mandatory Translation Recomputation",
        "mesh_ply": GATE_C_DIR / "tsdf_mesh_candidate_B_rigid_se3.ply",
        "mesh_glb": GATE_C_DIR / "tsdf_mesh_candidate_B_rigid_se3.glb",
        "tsdf_pcd_ply": GATE_C_DIR / "tsdf_pcd_candidate_B_rigid_se3.ply",
        "pre_pcd_ply": GATE_C_DIR / "pre_tsdf_pcd_candidate_B_rigid_se3.ply",
    }
]

# 1. Convert all meshes to GLB with vertex colors
print("[*] Exporting GLB meshes with vertex colors...")
for cand in candidates:
    o3d_m = o3d.io.read_triangle_mesh(str(cand["mesh_ply"]))
    verts = np.asarray(o3d_m.vertices)
    tris = np.asarray(o3d_m.triangles)
    if o3d_m.has_vertex_colors():
        vcols = (np.asarray(o3d_m.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    else:
        vcols = np.ones((len(verts), 4), dtype=np.uint8) * 180
        
    tri_m = trimesh.Trimesh(vertices=verts, faces=tris, vertex_colors=vcols)
    tri_m.export(str(cand["mesh_glb"]))
    print(f"  [+] {cand['display_name']}: GLB exported -> {cand['mesh_glb'].name} ({cand['mesh_glb'].stat().st_size / 1024 / 1024:.2f} MB)")

# 2. Extract TSDF point cloud from Open3D volume
print("\n[*] Rebuilding TSDF volumes to extract pure TSDF surface point clouds...")
windows = [list(range(i * 3, i * 3 + 6)) for i in range(11)]
CONF_THRESH = 2.5205
H_d, W_d = cached_raw["depth_0"].shape[1], cached_raw["depth_0"].shape[2]

from scripts.run_gate_c_extended_alignment import solve_alignment

for cand in candidates:
    mode_type = {
        "baseline_defective": "defective_baseline",
        "candidate_A_bounded_sim3": "bounded_sim3",
        "candidate_B_rigid_se3": "rigid_se3"
    }[cand["key"]]
    w_cam_weight = 0.0 if mode_type == "defective_baseline" else 10.0
    
    global_frames = {}
    w0_ext = cached_raw["ext_0"]
    w0_intr = cached_raw["intr_0"]
    w0_depth = cached_raw["depth_0"]
    w0_conf = cached_raw["conf_0"]
    for lp, gp in enumerate(windows[0]):
        R = w0_ext[lp, :3, :3]
        t = w0_ext[lp, :3, 3]
        c = -R.T @ t
        global_frames[gp] = {
            "ext": w0_ext[lp].copy(),
            "intr": w0_intr[lp].copy(),
            "depth": w0_depth[lp].copy(),
            "conf": w0_conf[lp].copy(),
            "center": c,
            "scale": 1.0
        }
        
    from src.reconstruction.point_cloud import depth_to_points
    for w_idx in range(1, len(windows)):
        win = windows[w_idx]
        ext_w = cached_raw[f"ext_{w_idx}"]
        intr_w = cached_raw[f"intr_{w_idx}"]
        depth_w = cached_raw[f"depth_{w_idx}"]
        conf_w = cached_raw[f"conf_{w_idx}"]
        
        shared_local = [lp for lp, gp in enumerate(win) if gp in global_frames]
        shared_global = [gp for lp, gp in enumerate(win) if gp in global_frames]
        
        sub_y = np.linspace(int(H_d * 0.15), int(H_d * 0.85), 20, dtype=int)
        sub_x = np.linspace(int(W_d * 0.15), int(W_d * 0.85), 20, dtype=int)
        
        pts_loc = []
        pts_glob = []
        cams_loc = []
        cams_glob = []
        
        for lp, gp in zip(shared_local, shared_global):
            ge = global_frames[gp]
            pg = depth_to_points(ge["depth"], ge["ext"], ge["intr"])
            pl = depth_to_points(depth_w[lp], ext_w[lp], intr_w[lp])
            c_loc = conf_w[lp]
            c_glob = ge["conf"]
            mask = (c_loc[sub_y][:, sub_x] >= CONF_THRESH) & (c_glob[sub_y][:, sub_x] >= CONF_THRESH)
            pl_sub = pl[sub_y][:, sub_x][mask]
            pg_sub = pg[sub_y][:, sub_x][mask]
            if len(pl_sub) > 10:
                pts_loc.append(pl_sub)
                pts_glob.append(pg_sub)
            R_l = ext_w[lp, :3, :3]
            t_l = ext_w[lp, :3, 3]
            cams_loc.append(-R_l.T @ t_l)
            cams_glob.append(ge["center"])
            
        X_all = np.vstack(pts_loc)
        Y_all = np.vstack(pts_glob)
        C_src = np.array(cams_loc)
        C_tgt = np.array(cams_glob)
        
        np.random.seed(42 + w_idx)
        N_pts = len(X_all)
        perm = np.random.permutation(N_pts)
        n_train = int(N_pts * 0.8)
        X_train, Y_train = X_all[perm[:n_train]], Y_all[perm[:n_train]]
        
        s_w, R_w, t_w, _, _ = solve_alignment(
            X_train, Y_train, C_src, C_tgt,
            mode_type=mode_type,
            w_pts=1.0, w_cam=w_cam_weight
        )
        
        for lp, gp in enumerate(win):
            if gp not in global_frames:
                R_local = ext_w[lp, :3, :3]
                t_local = ext_w[lp, :3, 3]
                c_local = -R_local.T @ t_local
                c_global = s_w * (R_w @ c_local) + t_w
                R_global = R_local @ R_w.T
                t_global = -R_global @ c_global
                ext_global = np.hstack([R_global, t_global.reshape(3, 1)])
                depth_global = s_w * depth_w[lp]
                global_frames[gp] = {
                    "ext": ext_global.astype(np.float32),
                    "intr": intr_w[lp].astype(np.float32),
                    "depth": depth_global.astype(np.float32),
                    "conf": conf_w[lp].copy(),
                    "center": c_global,
                    "scale": s_w
                }
                
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.015, sdf_trunc=0.045, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for i in range(36):
        gf = global_frames[i]
        d = gf["depth"].copy()
        mask_inv = (d < 0.20) | (d > 5.0) | (gf["conf"] < CONF_THRESH)
        d[mask_inv] = 0.0
        im_bgr = cv2.imread(image_paths[i])
        im_rgb = cv2.cvtColor(cv2.resize(im_bgr, (W_d, H_d)), cv2.COLOR_BGR2RGB)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(im_rgb), o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        K = gf["intr"]
        intr_obj = o3d.camera.PinholeCameraIntrinsic(W_d, H_d, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :4] = gf["ext"]
        vol.integrate(rgbd, intr_obj, ext_4x4)
        
    tsdf_pcd = vol.extract_point_cloud()
    o3d.io.write_point_cloud(str(cand["tsdf_pcd_ply"]), tsdf_pcd)
    print(f"  [+] {cand['display_name']}: TSDF Point Cloud saved -> {cand['tsdf_pcd_ply'].name} ({len(tsdf_pcd.points):,} points, {cand['tsdf_pcd_ply'].stat().st_size / 1024 / 1024:.2f} MB)")

print("\n[+] All models, GLBs, and TSDF point clouds successfully exported!")

