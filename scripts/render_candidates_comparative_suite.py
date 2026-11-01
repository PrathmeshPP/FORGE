#!/usr/bin/env python3
"""
F.O.R.G.E. - Multi-Candidate Comparative Rendering & Visual Verification Suite
=============================================================================
Renders Baseline, Candidate A, Candidate B, and Candidate C from strictly
identical camera viewpoints:
1. Front (Walkthrough entrance looking forward down central aisle)
2. Back (Looking backward toward entrance door)
3. Left Cot (Bed 1 & Study Desk close-up)
4. Right Cot (Bed 2 & Metal Wardrobes close-up)
5. Top (Top-down room plan overview)
6. Oblique (Corner perspective showing 3D enclosure)
7. Eye Level (Human 1.42m height inside aisle)

Assembles:
- Comparative multi-view contact sheets
- Master 4-way evaluation panel
- Quantitative sharpness, continuity, and triangle metric comparison table
"""

from __future__ import annotations
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import shutil
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EXP_DIR = PROJECT_ROOT / "outputs/indoor/photographic_texture_and_geometry_quality_recovery_nov1"
COMP_DIR = EXP_DIR / "comparisons"
COMP_DIR.mkdir(parents=True, exist_ok=True)

# Standardized Camera Viewpoints (in canonical Three.js space: Y up, floor at 0.0)
VIEWPOINTS = {
    "front_aisle": {
        "eye": [0.0, 1.42, -2.20],
        "target": [0.0, 1.30, 0.50],
        "up": [0.0, 1.0, 0.0],
        "label": "Front View (Central Walkway)"
    },
    "back_window": {
        "eye": [0.0, 1.42, 2.00],
        "target": [0.0, 1.30, -1.50],
        "up": [0.0, 1.0, 0.0],
        "label": "Back View (Curtains to Door)"
    },
    "left_cot_desk": {
        "eye": [-0.60, 1.35, -0.20],
        "target": [-1.65, 1.05, 0.05],
        "up": [0.0, 1.0, 0.0],
        "label": "Left Cot & Desk Close-up"
    },
    "right_cot_wardrobe": {
        "eye": [0.60, 1.35, -0.20],
        "target": [1.65, 1.05, 0.05],
        "up": [0.0, 1.0, 0.0],
        "label": "Right Cot & Wardrobe Close-up"
    },
    "top_overview": {
        "eye": [0.0, 4.20, 0.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "label": "Top-Down Room Plan"
    },
    "oblique_corner": {
        "eye": [1.80, 2.30, -2.60],
        "target": [-0.40, 0.90, 0.20],
        "up": [0.0, 1.0, 0.0],
        "label": "Oblique Diagonal Enclosure"
    },
    "eye_level_aisle": {
        "eye": [0.0, 1.42, -0.80],
        "target": [0.0, 1.40, 1.00],
        "up": [0.0, 1.0, 0.0],
        "label": "First-Person Walkthrough (1.42m)"
    }
}

MODELS = {
    "baseline": {
        "name": "Current Baseline (250k Tris, Naive 4K Grid)",
        "ply": PROJECT_ROOT / "outputs/indoor/texture_experiment_video_20261101_142811/room_photographic_model.ply",
        "color": (239, 68, 68) # Red
    },
    "candidate_a": {
        "name": "Candidate A (8K Super-Sampled UV Atlas)",
        "ply": EXP_DIR / "candidate_a_texture_projection/candidate_a_photographic_room.ply",
        "color": (245, 158, 11) # Amber
    },
    "candidate_b": {
        "name": "Candidate B (Ray-Oriented Normals + 450k Mesh)",
        "ply": EXP_DIR / "candidate_b_geometry_and_texture/candidate_b_photographic_room.ply",
        "color": (59, 130, 246) # Blue
    },
    "candidate_c": {
        "name": "Candidate C (Object-Aware + Multi-View Blended)",
        "ply": EXP_DIR / "candidate_c_combined_object_aware/candidate_c_photographic_room.ply",
        "color": (34, 197, 94) # Green
    }
}

def render_mesh_view(mesh: o3d.geometry.TriangleMesh, vp: dict, width: int = 1280, height: int = 720) -> np.ndarray:
    """Offscreen rendering of mesh with Open3D Visualizer."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.09, 0.10, 0.12])
    opt.light_on = True
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_lookat(vp["target"])
    ctr.set_up(vp["up"])
    ctr.set_front([vp["eye"][0] - vp["target"][0], vp["eye"][1] - vp["target"][1], vp["eye"][2] - vp["target"][2]])
    # Approximate camera zoom / distance
    dist = np.linalg.norm(np.array(vp["eye"]) - np.array(vp["target"]))
    ctr.set_zoom(0.70)

    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    im_np = (np.asarray(img) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)

def measure_sharpness_laplacian(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def run_comparative_suite():
    print("[*] Starting Multi-Candidate Comparative Rendering Suite...")

    # Load all models
    loaded_meshes = {}
    for key, info in MODELS.items():
        if info["ply"].exists():
            print(f"  Loading {key}: {info['ply'].name}...")
            m = o3d.io.read_triangle_mesh(str(info["ply"]))
            m.compute_vertex_normals()
            loaded_meshes[key] = m
        else:
            print(f"  [!] Missing {key}: {info['ply']}")

    renders = {}
    sharpness_scores = {}

    for vp_key, vp in VIEWPOINTS.items():
        print(f"\n[*] Rendering viewpoint: {vp['label']}...")
        vp_renders = {}
        for m_key, mesh in loaded_meshes.items():
            img = render_mesh_view(mesh, vp)
            vp_renders[m_key] = img
            sharpness = measure_sharpness_laplacian(img)
            sharpness_scores.setdefault(m_key, []).append(sharpness)
            out_img_path = COMP_DIR / f"{m_key}_{vp_key}.png"
            cv2.imwrite(str(out_img_path), img)
        renders[vp_key] = vp_renders

    # Generate 4-Way Comparison Panels for key views: front_aisle, left_cot_desk, right_cot_wardrobe
    print("\n[*] Assembling 4-Way Comparative Panels...")
    panel_views = ["front_aisle", "left_cot_desk", "right_cot_wardrobe", "oblique_corner"]

    for pv in panel_views:
        if pv not in renders:
            continue
        curr_vp_renders = renders[pv]
        if len(curr_vp_renders) < 4:
            continue

        # 2x2 grid (1280x720 each -> 2560x1440 panel)
        row1 = np.hstack([curr_vp_renders["baseline"], curr_vp_renders["candidate_a"]])
        row2 = np.hstack([curr_vp_renders["candidate_b"], curr_vp_renders["candidate_c"]])
        grid = np.vstack([row1, row2])

        # Add Header Banner
        banner_h = 100
        banner = np.full((banner_h, grid.shape[1], 3), (18, 20, 24), dtype=np.uint8)
        cv2.putText(banner, f"F.O.R.G.E. - 4-WAY COMPARATIVE AUDIT: {VIEWPOINTS[pv]['label'].upper()}", (40, 60), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2)
        panel = np.vstack([banner, grid])

        # Add labels on quadrants
        q_labels = [
            ("BASELINE (Current Nov 1)", 50, banner_h + 50, (0, 0, 255)),
            ("CANDIDATE A (8K Super-Sampled UV Atlas)", 1280 + 50, banner_h + 50, (0, 165, 255)),
            ("CANDIDATE B (Ray-Oriented Normals + 450k Mesh)", 50, banner_h + 720 + 50, (255, 150, 50)),
            ("CANDIDATE C (Object-Aware + Multi-View Blended)", 1280 + 50, banner_h + 720 + 50, (50, 220, 50))
        ]
        for text, x, y, col in q_labels:
            cv2.rectangle(panel, (x - 10, y - 30), (x + len(text) * 22, y + 10), (15, 18, 22), -1)
            cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, col, 2)

        panel_path = COMP_DIR / f"comparative_panel_{pv}.png"
        cv2.imwrite(str(panel_path), panel)
        print(f"  [+] Saved panel: {panel_path.name}")

    # Copy primary comparative panel to Artifact Directory
    artifact_dir = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953")
    master_panel_src = COMP_DIR / "comparative_panel_front_aisle.png"
    if master_panel_src.exists():
        shutil.copyfile(str(master_panel_src), str(artifact_dir / "master_room_recovery_comparison_panel.png"))

    # Print Summary Table
    print("\n" + "=" * 70)
    print("F.O.R.G.E. - EXPERIMENTAL CANDIDATES EVALUATION TABLE")
    print("=" * 70)
    print(f"{'Candidate':<20} | {'Vertices':<10} | {'Faces':<10} | {'Mean Sharpness':<16} | {'GLB Size (MB)':<14}")
    print("-" * 78)
    for m_key, info in MODELS.items():
        if m_key in loaded_meshes:
            m = loaded_meshes[m_key]
            mean_s = np.mean(sharpness_scores.get(m_key, [0]))
            glb_path = info["ply"].parent / f"{info['ply'].stem}.glb"
            glb_sz = round(glb_path.stat().st_size / (1024**2), 2) if glb_path.exists() else 0.0
            print(f"{m_key:<20} | {len(m.vertices):<10,} | {len(m.triangles):<10,} | {mean_s:<16.2f} | {glb_sz:<14.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_comparative_suite()

