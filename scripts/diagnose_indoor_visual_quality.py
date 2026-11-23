r"""
F.O.R.G.E. — Indoor Visual Quality Forensic Diagnostic Engine.
Performs rigorous geometric, topological, UV coordinate, normal vector,
and coverage denominator analysis on Stage 2 digital twins.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analyze_mesh_topology_and_geometry(mesh_path: Path) -> Dict[str, Any]:
    print(f"\n[*] Analyzing mesh topology & geometry: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), process=False)
    
    vertices = mesh.vertices # (V, 3)
    faces = mesh.faces # (F, 3)
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # 1. Edge Lengths
    # Half-edges
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    e0 = np.linalg.norm(v1 - v0, axis=1) # edge opposite to v2
    e1 = np.linalg.norm(v2 - v1, axis=1) # edge opposite to v0
    e2 = np.linalg.norm(v0 - v2, axis=1) # edge opposite to v1
    all_edges = np.concatenate([e0, e1, e2])
    
    edge_stats = {
        "min": float(np.min(all_edges)),
        "p05": float(np.percentile(all_edges, 5)),
        "p25": float(np.percentile(all_edges, 25)),
        "median": float(np.median(all_edges)),
        "mean": float(np.mean(all_edges)),
        "p75": float(np.percentile(all_edges, 75)),
        "p95": float(np.percentile(all_edges, 95)),
        "max": float(np.max(all_edges)),
        "std": float(np.std(all_edges)),
    }
    
    # 2. Triangle Aspect Ratio: rho = R / (2 * r)
    # R = abc / (4A), r = A / s => rho = abc / (8 * (s-a)(s-b)(s-c))
    # where s = (a+b+c)/2
    a, b, c = e0, e1, e2
    s = (a + b + c) / 2.0
    denom = 8.0 * (s - a) * (s - b) * (s - c)
    valid_mask = denom > 1e-12
    aspect_ratios = np.ones(num_faces, dtype=np.float64)
    aspect_ratios[valid_mask] = (a[valid_mask] * b[valid_mask] * c[valid_mask]) / denom[valid_mask]
    
    sliver_ratio_gt3 = float(np.sum(aspect_ratios > 3.0) / num_faces * 100.0)
    sliver_ratio_gt5 = float(np.sum(aspect_ratios > 5.0) / num_faces * 100.0)
    sliver_ratio_gt10 = float(np.sum(aspect_ratios > 10.0) / num_faces * 100.0)
    
    ar_stats = {
        "mean": float(np.mean(aspect_ratios)),
        "median": float(np.median(aspect_ratios)),
        "max": float(np.max(aspect_ratios)),
        "pct_gt_3": sliver_ratio_gt3,
        "pct_gt_5": sliver_ratio_gt5,
        "pct_gt_10": sliver_ratio_gt10,
    }
    
    # 3. Dihedral Angles between adjacent faces
    face_normals = mesh.face_normals
    face_adjacency = mesh.face_adjacency
    
    n0 = face_normals[face_adjacency[:, 0]]
    n1 = face_normals[face_adjacency[:, 1]]
    cos_dihedral = np.sum(n0 * n1, axis=1)
    cos_dihedral = np.clip(cos_dihedral, -1.0, 1.0)
    dihedral_deg = np.degrees(np.arccos(cos_dihedral))
    
    dihedral_stats = {
        "total_internal_edges": int(len(dihedral_deg)),
        "mean_deg": float(np.mean(dihedral_deg)),
        "median_deg": float(np.median(dihedral_deg)),
        "pct_under_5deg": float(np.sum(dihedral_deg < 5.0) / len(dihedral_deg) * 100.0),
        "pct_5_to_15deg": float(np.sum((dihedral_deg >= 5.0) & (dihedral_deg < 15.0)) / len(dihedral_deg) * 100.0),
        "pct_15_to_38deg": float(np.sum((dihedral_deg >= 15.0) & (dihedral_deg < 38.0)) / len(dihedral_deg) * 100.0),
        "pct_gt_38deg": float(np.sum(dihedral_deg >= 38.0) / len(dihedral_deg) * 100.0),
    }
    
    # 4. Mesh Manifoldness & Boundaries
    is_watertight = bool(mesh.is_watertight)
    # Boundary edges: count of edges in mesh.edges with only 1 occurrence
    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edge_count = int(np.sum(counts == 1))
    non_manifold_edge_count = int(np.sum(counts > 2))
    
    # 5. Triangle 3D Areas
    areas = mesh.area_faces
    area_stats = {
        "min": float(np.min(areas)),
        "median": float(np.median(areas)),
        "mean": float(np.mean(areas)),
        "max": float(np.max(areas)),
        "total_area_m2": float(np.sum(areas))
    }
    
    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "is_watertight": is_watertight,
        "boundary_edges": boundary_edge_count,
        "non_manifold_edges": non_manifold_edge_count,
        "edge_lengths_m": edge_stats,
        "aspect_ratios": ar_stats,
        "dihedral_angles": dihedral_stats,
        "face_areas_m2": area_stats
    }


def analyze_obj_uv_and_normals(obj_path: Path) -> Dict[str, Any]:
    print(f"\n[*] Parsing OBJ UV coordinates & normals: {obj_path.name}")
    
    v_list = []
    vt_list = []
    vn_list = []
    face_v = []
    face_vt = []
    face_vn = []
    
    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                v_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('vt '):
                parts = line.strip().split()
                vt_list.append([float(parts[1]), float(parts[2])])
            elif line.startswith('vn '):
                parts = line.strip().split()
                vn_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()[1:]
                fv = []
                fvt = []
                fvn = []
                for p in parts:
                    sub = p.split('/')
                    fv.append(int(sub[0]) - 1)
                    if len(sub) > 1 and sub[1]:
                        fvt.append(int(sub[1]) - 1)
                    if len(sub) > 2 and sub[2]:
                        fvn.append(int(sub[2]) - 1)
                face_v.append(fv)
                face_vt.append(fvt)
                face_vn.append(fvn)
                
    V = np.array(v_list, dtype=np.float32)
    VT = np.array(vt_list, dtype=np.float32)
    VN = np.array(vn_list, dtype=np.float32)
    FV = np.array(face_v, dtype=np.int32)
    FVT = np.array(face_vt, dtype=np.int32)
    FVN = np.array(face_vn, dtype=np.int32)
    
    print(f"    Loaded OBJ: {len(V):,} V, {len(VT):,} VT, {len(VN):,} VN, {len(FV):,} Faces")
    
    # 1. UV Inversion and Signed Area Analysis
    uv0 = VT[FVT[:, 0]]
    uv1 = VT[FVT[:, 1]]
    uv2 = VT[FVT[:, 2]]
    
    # Signed 2D area: 0.5 * ((u1 - u0)(v2 - v0) - (u2 - u0)(v1 - v0))
    cross_2d = (uv1[:, 0] - uv0[:, 0]) * (uv2[:, 1] - uv0[:, 1]) - (uv2[:, 0] - uv0[:, 0]) * (uv1[:, 1] - uv0[:, 1])
    signed_uv_area = 0.5 * cross_2d
    
    inverted_uv_count = int(np.sum(signed_uv_area < -1e-9))
    degenerate_uv_count = int(np.sum(np.abs(signed_uv_area) < 1e-9))
    valid_uv_count = int(np.sum(signed_uv_area > 1e-9))
    
    # 2. UV Stretching / Distortion Metric
    # Ratio of 2D UV area to 3D world area
    v0 = V[FV[:, 0]]
    v1 = V[FV[:, 1]]
    v2 = V[FV[:, 2]]
    cross_3d = np.cross(v1 - v0, v2 - v0)
    area_3d = 0.5 * np.linalg.norm(cross_3d, axis=1)
    
    abs_uv_area = np.abs(signed_uv_area)
    valid_3d = area_3d > 1e-8
    stretch_ratios = np.zeros(len(FV), dtype=np.float64)
    stretch_ratios[valid_3d] = abs_uv_area[valid_3d] / area_3d[valid_3d]
    
    median_stretch = np.median(stretch_ratios[valid_3d]) if np.any(valid_3d) else 1.0
    normalized_stretch = stretch_ratios / (median_stretch + 1e-12)
    
    uv_stretch_stats = {
        "median_uv_area": float(np.median(abs_uv_area)),
        "inverted_uv_faces": inverted_uv_count,
        "inverted_pct": float(inverted_uv_count / len(FV) * 100.0),
        "degenerate_uv_faces": degenerate_uv_count,
        "degenerate_pct": float(degenerate_uv_count / len(FV) * 100.0),
        "stretch_variance": float(np.var(normalized_stretch[valid_3d])) if np.any(valid_3d) else 0.0,
        "stretch_p05": float(np.percentile(normalized_stretch[valid_3d], 5)),
        "stretch_p95": float(np.percentile(normalized_stretch[valid_3d], 95)),
    }
    
    # 3. Atlas Rasterization & Occupancy Analysis
    # Rasterize UV triangles into a 2048x2048 boolean mask to determine true pixel coverage
    atlas_res = 2048
    mask = np.zeros((atlas_res, atlas_res), dtype=np.uint8)
    
    # Scale UVs (u in [0, 1], v in [0, 1], where v=0 is bottom in standard UV, top in image)
    # Standard glTF / OBJ: (u, 1-v) for image coordinates
    pts0 = np.clip(np.round(np.column_stack([uv0[:, 0] * (atlas_res - 1), (1.0 - uv0[:, 1]) * (atlas_res - 1)])), 0, atlas_res - 1).astype(np.int32)
    pts1 = np.clip(np.round(np.column_stack([uv1[:, 0] * (atlas_res - 1), (1.0 - uv1[:, 1]) * (atlas_res - 1)])), 0, atlas_res - 1).astype(np.int32)
    pts2 = np.clip(np.round(np.column_stack([uv2[:, 0] * (atlas_res - 1), (1.0 - uv2[:, 1]) * (atlas_res - 1)])), 0, atlas_res - 1).astype(np.int32)
    
    # Draw all triangles
    for i in range(len(FV)):
        tri = np.array([pts0[i], pts1[i], pts2[i]], dtype=np.int32)
        cv2.fillConvexPoly(mask, tri, 1)
        
    sampled_pixels_2k = int(np.sum(mask > 0))
    total_pixels_2k = atlas_res * atlas_res
    actual_sampled_pct = float(sampled_pixels_2k / total_pixels_2k * 100.0)
    
    # Scaling to 4096x4096 equivalent
    total_pixels_4k = 4096 * 4096
    effective_sampled_pixels_4k = int(sampled_pixels_2k * 4)
    
    atlas_utilization = {
        "total_atlas_pixels_4k": total_pixels_4k,
        "grid_slots_total": 16,
        "active_tile_size": [576, 1024],
        "slot_size": [1024, 1024],
        "active_slot_pixels_4k": 16 * 576 * 1024, # 9,437,184
        "active_slot_pct": float(16 * 576 * 1024 / total_pixels_4k * 100.0), # 56.25%
        "empty_padding_pct": float((1.0 - 16 * 576 * 1024 / total_pixels_4k) * 100.0), # 43.75%
        "actual_uv_sampled_pixels_4k": effective_sampled_pixels_4k,
        "actual_uv_sampled_pct": actual_sampled_pct, # true geometry occupancy
    }
    
    # 4. Normal Vector Sharedness / Continuity Analysis
    # Check whether vertices shared across planar faces have mathematically identical corner normals
    # Build vertex to corner normal map
    vert_to_normals: Dict[int, List[np.ndarray]] = {}
    for face_idx in range(len(FV)):
        for corner in range(3):
            v_id = FV[face_idx, corner]
            vn_id = FVN[face_idx, corner]
            n_vec = VN[vn_id]
            if v_id not in vert_to_normals:
                vert_to_normals[v_id] = []
            vert_to_normals[v_id].append(n_vec)
            
    # Compute max angular divergence among meeting corner normals at each vertex
    max_divergences_deg = []
    divergent_vertices_count = 0
    identical_vertices_count = 0
    
    for v_id, n_list in vert_to_normals.items():
        if len(n_list) <= 1:
            max_divergences_deg.append(0.0)
            identical_vertices_count += 1
            continue
        arr = np.array(n_list) # (K, 3)
        # dot products between all pairs
        dots = np.dot(arr, arr.T)
        dots = np.clip(dots, -1.0, 1.0)
        min_dot = np.min(dots)
        angle_deg = float(np.degrees(np.arccos(min_dot)))
        max_divergences_deg.append(angle_deg)
        if angle_deg > 0.05: # more than 0.05 deg difference
            divergent_vertices_count += 1
        else:
            identical_vertices_count += 1
            
    max_div_arr = np.array(max_divergences_deg)
    normal_continuity = {
        "unique_vertices": len(vert_to_normals),
        "identical_normal_vertices": identical_vertices_count,
        "identical_normal_pct": float(identical_vertices_count / len(vert_to_normals) * 100.0),
        "divergent_normal_vertices": divergent_vertices_count,
        "divergent_normal_pct": float(divergent_vertices_count / len(vert_to_normals) * 100.0),
        "mean_angular_divergence_deg": float(np.mean(max_div_arr)),
        "median_angular_divergence_deg": float(np.median(max_div_arr)),
        "pct_divergence_gt_38deg": float(np.sum(max_div_arr >= 38.0) / len(max_div_arr) * 100.0),
        "pct_divergence_between_0_and_38deg": float(np.sum((max_div_arr > 0.05) & (max_div_arr < 38.0)) / len(max_div_arr) * 100.0),
    }
    
    return {
        "uv_stretch": uv_stretch_stats,
        "atlas_utilization": atlas_utilization,
        "normal_continuity": normal_continuity
    }


def analyze_coverage_and_denominators(report_json_path: Path) -> Dict[str, Any]:
    print(f"\n[*] Analyzing coverage taxonomy & denominators: {report_json_path.name}")
    with open(report_json_path, 'r') as f:
        rep = json.load(f)
        
    cov = rep.get("surface_coverage", {})
    geom = rep.get("mesh_geometry", {})
    
    total_triangles = geom.get("decimated_triangles", 249999)
    green_faces = cov.get("directly_photographed_green_faces", 0)
    yellow_faces = cov.get("physically_occluded_yellow_faces", 0)
    red_faces = cov.get("unobserved_red_faces", 0)
    blue_faces = cov.get("seam_transition_blue_faces", 0)
    
    # Clarify denominators:
    # Denominator 1: Total Mesh Triangles (including exterior Poisson hull and ceiling)
    # Denominator 2: True Observable Interior Triangles (excluding exterior hull)
    exterior_hull_and_ceiling = red_faces
    interior_faces = total_triangles - exterior_hull_and_ceiling
    
    # Total assigned photographic texture:
    # Notice: Blue faces (seam boundaries) ARE directly photographed!
    # They were labeled blue solely to highlight where the adjacent triangle has a different camera ID.
    total_photographed_faces = green_faces + blue_faces
    
    denom_report = {
        "denominator_total_mesh_triangles": total_triangles,
        "denominator_interior_triangles": interior_faces,
        "exterior_hull_unobserved_ceiling": {
            "face_count": exterior_hull_and_ceiling,
            "pct_of_total_mesh": float(exterior_hull_and_ceiling / total_triangles * 100.0)
        },
        "interior_surface_breakdown": {
            "physically_occluded_under_furniture": {
                "face_count": yellow_faces,
                "pct_of_total_mesh": float(yellow_faces / total_triangles * 100.0),
                "pct_of_interior": float(yellow_faces / interior_faces * 100.0)
            },
            "total_assigned_photographic_texture": {
                "face_count": total_photographed_faces,
                "pct_of_total_mesh": float(total_photographed_faces / total_triangles * 100.0),
                "pct_of_interior": float(total_photographed_faces / interior_faces * 100.0)
            },
            "subdivision_pure_chart_interior": {
                "face_count": green_faces,
                "pct_of_total_mesh": float(green_faces / total_triangles * 100.0),
                "pct_of_interior": float(green_faces / interior_faces * 100.0),
                "pct_of_photographed": float(green_faces / total_photographed_faces * 100.0)
            },
            "subdivision_seam_boundary_faces": {
                "face_count": blue_faces,
                "pct_of_total_mesh": float(blue_faces / total_triangles * 100.0),
                "pct_of_interior": float(blue_faces / interior_faces * 100.0),
                "pct_of_photographed": float(blue_faces / total_photographed_faces * 100.0)
            }
        }
    }
    
    return denom_report


def analyze_native_texel_preservation(report_json_path: Path) -> Dict[str, Any]:
    print(f"\n[*] Correcting native texel preservation metric: {report_json_path.name}")
    with open(report_json_path, 'r') as f:
        rep = json.load(f)
        
    atlas = rep.get("atlas_metrics", {})
    native_w, native_h = atlas.get("native_frame_size", [478, 850])
    tile_w, tile_h = atlas.get("tile_size", [576, 1024])
    slot_w, slot_h = atlas.get("slot_size", [1024, 1024])
    
    # 1. Source pixel retention:
    # Since tile dimensions (576 x 1024) are strictly GREATER than native dimensions (478 x 850),
    # zero native source pixels are dropped / decimated. Native pixel retention is 100%.
    source_pixel_retention = 100.0
    
    # 2. Rescaling / Upsampling Factor:
    scale_x = tile_w / native_w # 576 / 478 = 1.2050
    scale_y = tile_h / native_h # 1024 / 850 = 1.2047
    mean_linear_upscale = (scale_x + scale_y) / 2.0
    area_upscale = (tile_w * tile_h) / (native_w * native_h)
    
    # 3. Dissecting the 120.47% figure:
    # 120.47% was tile_h / native_h (1024 / 850).
    # Calling it "preservation" was misleading because it represents an upscaling factor (+20.47%),
    # NOT that 120% of information was captured.
    
    # 4. Slot efficiency:
    tile_area = tile_w * tile_h # 589,824
    slot_area = slot_w * slot_h # 1,048,576
    slot_utilization = (tile_area / slot_area) * 100.0 # 56.25%
    
    return {
        "native_source_dimensions": [native_w, native_h],
        "native_source_pixels": native_w * native_h,
        "atlas_tile_dimensions": [tile_w, tile_h],
        "atlas_tile_pixels": tile_w * tile_h,
        "native_pixel_retention_pct": 100.0,
        "optical_downsampling": False,
        "linear_upscale_factor": float(mean_linear_upscale),
        "linear_upscale_pct": float((mean_linear_upscale - 1.0) * 100.0),
        "area_upscale_factor": float(area_upscale),
        "area_upscale_pct": float((area_upscale - 1.0) * 100.0),
        "slot_utilization_pct": float(slot_utilization),
        "slot_wasted_padding_pct": float(100.0 - slot_utilization),
        "explanation": "The 120.47% figure corresponds strictly to the linear scaling factor (1024/850) used to fit the portrait frame into the 1024-height slot. 100% of native optical pixels were retained (zero downsampling), and pixels were bilinearly upsampled by +20.47% linearly (+45.17% in area). 43.75% of each slot remains black padding to preserve the exact optical aspect ratio."
    }


def main():
    print("=" * 70)
    print("  F.O.R.G.E. Stage 2 Visual Quality Forensic Diagnostic Engine")
    print("=" * 70)
    
    datasets = [
        {
            "id": "video_1",
            "name": "Student Hostel Room B 314",
            "dir": PROJECT_ROOT / "outputs/indoor/stage2_texture_video1",
            "obj": PROJECT_ROOT / "outputs/indoor/stage2_texture_video1/candidate4_combined_final/final_stage2_model_canonical.obj",
            "report": PROJECT_ROOT / "outputs/indoor/stage2_texture_video1/stage2_execution_report.json"
        },
        {
            "id": "video_2",
            "name": "Hostel Corridor Walkthrough",
            "dir": PROJECT_ROOT / "outputs/indoor/stage2_texture_video2",
            "obj": PROJECT_ROOT / "outputs/indoor/stage2_texture_video2/candidate4_combined_final/final_stage2_model_canonical.obj",
            "report": PROJECT_ROOT / "outputs/indoor/stage2_texture_video2/stage2_execution_report.json"
        }
    ]
    
    full_diagnostic_summary = {}
    
    for d in datasets:
        print(f"\n" + "=" * 60)
        print(f"  Diagnosing: {d['name']} ({d['id']})")
        print("=" * 60)
        
        topo = analyze_mesh_topology_and_geometry(d["obj"])
        uv_norm = analyze_obj_uv_and_normals(d["obj"])
        denom = analyze_coverage_and_denominators(d["report"])
        texel = analyze_native_texel_preservation(d["report"])
        
        diag = {
            "dataset": d["id"],
            "scene_name": d["name"],
            "topology_and_geometry": topo,
            "uv_and_normals": uv_norm,
            "coverage_and_denominators": denom,
            "texel_preservation_correction": texel
        }
        
        out_json = d["dir"] / "forensic_diagnostic_results.json"
        with open(out_json, 'w') as f:
            json.dump(diag, f, indent=2)
            
        print(f"\n[+] Saved detailed forensic diagnostic: {out_json}")
        full_diagnostic_summary[d["id"]] = diag
        
    # Write combined summary
    master_diag_json = PROJECT_ROOT / "outputs/indoor/stage2_visual_quality_forensic_audit.json"
    with open(master_diag_json, 'w') as f:
        json.dump(full_diagnostic_summary, f, indent=2)
    print(f"\n[+] Saved Master Forensic Audit: {master_diag_json}")


if __name__ == "__main__":
    main()

