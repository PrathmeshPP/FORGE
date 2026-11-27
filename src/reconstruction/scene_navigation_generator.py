"""
F.O.R.G.E. — Scene-Specific Walkthrough Metadata & Route Validation Engine.

Generates scene-specific navigation bounds, floor and ceiling planes, walkable surfaces,
obstacle collision bounding boxes, spawn location, and performs route validation
for interactive first-person free-roam walkthroughs without hardcoding room parameters.

Eligibility Outcomes:
- WALKTHROUGH_READY: Continuous floor evidence, valid obstacle clearance, stable single-level navigation.
- PARTIAL_MODEL: Reconstructed geometry present, but floor coverage is partial or uncertain. Orbit mode recommended.
- RECAPTURE_NEEDED: Insufficient visual/geometric coverage for reliable navigation. Actionable advice generated.
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import open3d as o3d


@dataclass
class RouteValidationResult:
    """Detailed route validation metrics along reconstructed camera trajectory."""
    total_waypoints: int
    path_length_m: float
    floor_continuity_ratio: float  # fraction of waypoints with verified floor underneath [0.0 - 1.0]
    obstacle_clearance_ratio: float  # fraction of waypoints with clearance >= player_radius
    max_step_height_m: float  # maximum vertical jump between consecutive waypoints
    elevation_variance_m: float
    is_single_level: bool
    floor_gaps_detected: int
    wall_penetrations_detected: int
    eligibility_status: str  # "WALKTHROUGH_READY" | "PARTIAL_MODEL" | "RECAPTURE_NEEDED"
    actionable_notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneNavigationMetadata:
    """Complete scene-specific navigation and collision definition."""
    scene_id: str
    reconstructed_model_url: str
    bounds_xyz: List[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    floor_y: float
    ceiling_y: float
    room_height_m: float
    eye_height: float
    spawn_position: List[float]  # [x, y, z]
    spawn_yaw_deg: float
    walkable_bounds_2d: List[float]  # [x_min, z_min, x_max, z_max]
    collision_obstacles: List[Dict[str, Any]]
    camera_trajectory: List[Dict[str, Any]]
    route_validation: RouteValidationResult
    controls_guide: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["route_validation"] = self.route_validation.to_dict()
        return d


def derive_scene_navigation(
    mesh: o3d.geometry.TriangleMesh,
    extrinsics: np.ndarray,
    detected_planes: Optional[Dict[str, Any]] = None,
    scene_id: str = "custom_indoor_scene",
    model_glb_url: str = "",
    player_radius: float = 0.08,
    frame_manifest: Optional[List[Dict[str, Any]]] = None,
    floor_rotation: Optional[np.ndarray] = None,
    canonical_center: Optional[np.ndarray] = None
) -> SceneNavigationMetadata:
    """Derives scene-specific navigation bounds, floor plane, obstacles, and validates walkthrough route."""
    verts = np.asarray(mesh.vertices)
    if len(verts) == 0:
        raise ValueError("Mesh contains no vertices.")

    # 1. Scene spatial bounds
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bounds_xyz = [
        round(float(bbox_min[0]), 3),
        round(float(bbox_min[1]), 3),
        round(float(bbox_min[2]), 3),
        round(float(bbox_max[0]), 3),
        round(float(bbox_max[1]), 3),
        round(float(bbox_max[2]), 3),
    ]

    # 2. Camera trajectory extraction & alignment to canonical deliverable coordinate system
    N = len(extrinsics)
    trajectory: List[Dict[str, Any]] = []
    cam_centers = []
    T_canon = np.array([1.0, -1.0, -1.0], dtype=np.float64)

    for i in range(N):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        c = -R.T @ t
        fwd = R.T @ np.array([0.0, 0.0, 1.0])

        if floor_rotation is not None:
            c = floor_rotation @ c
            fwd = floor_rotation @ fwd

        if canonical_center is not None:
            c = (c * T_canon) - canonical_center
            fwd = fwd * T_canon

        cam_centers.append(c)

        frame_url = ""
        filename = f"frame_{i:02d}.png"
        ts = float(i) * 0.5
        if frame_manifest and i < len(frame_manifest):
            ts = frame_manifest[i].get("timestamp_sec", ts)
            filename = Path(frame_manifest[i].get("file_path", filename)).name
            if model_glb_url and "/" in model_glb_url and model_glb_url.startswith("/"):
                base_url_dir = model_glb_url.rsplit("/", 1)[0]
                frame_url = f"{base_url_dir}/frames/{filename}"
            else:
                frame_url = f"/frames/{filename}"

        trajectory.append({
            "index": i,
            "position": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)],
            "forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4), round(float(fwd[2]), 4)],
            "timestamp_sec": round(ts, 2),
            "frame_url": frame_url,
            "filename": filename
        })

    cam_centers_arr = np.array(cam_centers) if cam_centers else np.zeros((1, 3))

    # 3. Trajectory-Grounded Floor & Ceiling plane derivation
    # Rather than using arbitrary bbox_min[1] which is corrupted by outlier boundary points,
    # sample mesh vertices directly underneath the camera trajectory walking path.
    under_floors = []
    for c in cam_centers:
        dxz2 = (verts[:, 0] - c[0]) ** 2 + (verts[:, 2] - c[2]) ** 2
        below = dxz2 < (0.35 ** 2)
        if np.any(below):
            y_below = verts[below, 1]
            y_under = y_below[y_below < c[1] - 0.10]
            if len(y_under) > 10:
                under_floors.append(float(np.percentile(y_under, 5)))

    if len(under_floors) >= 3:
        floor_y = round(float(np.median(under_floors)), 3)
    else:
        floor_y = round(float(np.percentile(verts[:, 1], 5)), 3)

    ceiling_cand = float(np.percentile(verts[:, 1], 98))
    ceiling_y = round(float(max(floor_y + 0.35, ceiling_cand)), 3)

    room_height = max(0.35, ceiling_y - floor_y)
    # Scale-faithful camera eye level: median camera trajectory Y clamped cleanly inside [floor, ceiling]
    med_cam_y = float(np.median(cam_centers_arr[:, 1])) if len(cam_centers_arr) > 0 else (floor_y + room_height * 0.45)
    eye_height = round(float(np.clip(med_cam_y, floor_y + room_height * 0.25, floor_y + room_height * 0.65)), 3)

    # 4. Obstacle & Walkable Floor clustering
    # Identify non-floor structural geometry (e.g. furniture, beds, wardrobes)
    # Exclude outer walls (bounded by room perimeter) and human walking corridor
    non_floor_mask = (verts[:, 1] > floor_y + 0.06) & (verts[:, 1] < ceiling_y - 0.08)
    wall_margin = 0.10
    inner_mask = (
        (verts[:, 0] > bbox_min[0] + wall_margin) & (verts[:, 0] < bbox_max[0] - wall_margin) &
        (verts[:, 2] > bbox_min[2] + wall_margin) & (verts[:, 2] < bbox_max[2] - wall_margin)
    )

    # Exclude the human walking corridor (camera trajectory with player radius clearance)
    corridor_clearance = player_radius + 0.08
    if len(cam_centers) > 0:
        traj_dist_sq = np.min([((verts[:, 0] - c[0]) ** 2 + (verts[:, 2] - c[2]) ** 2) for c in cam_centers], axis=0)
        corridor_mask = traj_dist_sq > (corridor_clearance ** 2)
    else:
        corridor_mask = np.ones(len(verts), dtype=bool)

    cand_pts = verts[non_floor_mask & inner_mask & corridor_mask]
    obstacles: List[Dict[str, Any]] = []

    if len(cand_pts) > 0:
        nbins = 6
        bx = np.linspace(bbox_min[0] + wall_margin, bbox_max[0] - wall_margin, nbins + 1)
        bz = np.linspace(bbox_min[2] + wall_margin, bbox_max[2] - wall_margin, nbins + 1)

        for ix in range(nbins):
            for iz in range(nbins):
                m = (
                    (cand_pts[:, 0] >= bx[ix]) & (cand_pts[:, 0] < bx[ix + 1]) &
                    (cand_pts[:, 2] >= bz[iz]) & (cand_pts[:, 2] < bz[iz + 1])
                )
                pts_sub = cand_pts[m]
                if len(pts_sub) >= 300:  # significant structural mass
                    sub_min = pts_sub.min(axis=0)
                    sub_max = pts_sub.max(axis=0)
                    b_cand = [
                        round(float(sub_min[0]), 3),
                        round(float(sub_min[2]), 3),
                        round(float(sub_max[0]), 3),
                        round(float(sub_max[2]), 3),
                    ]
                    # Verify obstacle cluster does not bridge across the human walking trajectory corridor
                    crosses_path = any(
                        (b_cand[0] <= c[0] <= b_cand[2]) and (b_cand[1] <= c[2] <= b_cand[3])
                        for c in cam_centers
                    )
                    if not crosses_path:
                        obs_id = f"furniture_{ix}_{iz}"
                        obstacles.append({
                            "id": obs_id,
                            "name": f"Furnishing Block ({obs_id})",
                            "type": "obstacle",
                            "bounds": b_cand,
                            "height_range": [round(float(sub_min[1]), 3), round(float(sub_max[1]), 3)],
                            "label": f"Obstacle ({sub_max[1]-sub_min[1]:.2f}m)"
                        })

    # 5. Spawn point selection
    # Choose position along trajectory that is within open central space
    spawn_pos = [0.0, eye_height, 0.0]
    spawn_yaw = 180.0

    if len(cam_centers) > 0:
        # Check candidate points along first 25% of trajectory for clearance
        best_idx = 0
        best_cand_pos = cam_centers[0]
        best_clearance = -1.0
        for ci_idx, c_pos in enumerate(cam_centers[:max(1, len(cam_centers) // 4)]):
            min_dist_to_obs = 100.0
            for obs in obstacles:
                b = obs["bounds"]
                # 2D distance to box
                dx = max(b[0] - c_pos[0], 0, c_pos[0] - b[2])
                dz = max(b[1] - c_pos[2], 0, c_pos[2] - b[3])
                dist = math.sqrt(dx * dx + dz * dz)
                if dist < min_dist_to_obs:
                    min_dist_to_obs = dist

            if min_dist_to_obs > best_clearance:
                best_clearance = min_dist_to_obs
                best_cand_pos = c_pos
                best_idx = ci_idx

        spawn_pos = [round(float(best_cand_pos[0]), 3), eye_height, round(float(best_cand_pos[2]), 3)]
        fwd_vec = np.array(trajectory[best_idx]["forward"], dtype=np.float64)
        if math.hypot(fwd_vec[0], fwd_vec[2]) > 0.1:
            spawn_yaw = round(float(np.degrees(math.atan2(fwd_vec[0], fwd_vec[2]))), 1)
        elif len(cam_centers) > 1:
            diff = cam_centers[1] - cam_centers[0]
            spawn_yaw = round(float(np.degrees(math.atan2(diff[0], diff[2]))), 1)

    # 6. Route Validation
    # Test walking steps along camera trajectory
    valid_floor_count = 0
    clearance_ok_count = 0
    max_step = 0.0
    floor_gaps = 0
    wall_pens = 0
    notes: List[str] = []

    vert_y = verts[:, 1]
    floor_search_radius = 0.28  # realistic stride radius for indoor footstep support
    for idx, c in enumerate(cam_centers):
        # 1. Floor underneath test: check vertices within stride radius in X-Z near floor_y
        d_xz_sq = (verts[:, 0] - c[0]) ** 2 + (verts[:, 2] - c[2]) ** 2
        nearby = d_xz_sq < (floor_search_radius ** 2)
        if np.any(nearby):
            y_near = vert_y[nearby]
            # Check if there is a surface near floor_y
            floor_near = np.any(np.abs(y_near - floor_y) < 0.20)
            if floor_near:
                valid_floor_count += 1
            else:
                floor_gaps += 1
        else:
            floor_gaps += 1

        # 2. Obstacle clearance test
        in_obstacle = False
        for obs in obstacles:
            b = obs["bounds"]
            if (b[0] - player_radius <= c[0] <= b[2] + player_radius) and \
               (b[1] - player_radius <= c[2] <= b[3] + player_radius):
                in_obstacle = True
                break
        if not in_obstacle:
            clearance_ok_count += 1

        # 3. Step height test
        if idx > 0:
            dy = abs(c[1] - cam_centers[idx - 1][1])
            if dy > max_step:
                max_step = dy

        # 4. Bounding box confinement test (allowing 15cm margin for Poisson trimmed mesh boundaries)
        wall_margin_tol = 0.15
        if c[0] < bbox_min[0] - wall_margin_tol or c[0] > bbox_max[0] + wall_margin_tol or \
           c[2] < bbox_min[2] - wall_margin_tol or c[2] > bbox_max[2] + wall_margin_tol:
            wall_pens += 1

    floor_ratio = float(valid_floor_count / max(1, N))
    clearance_ratio = float(clearance_ok_count / max(1, N))
    elev_var = float(np.std(cam_centers_arr[:, 1])) if N > 1 else 0.0
    is_single = elev_var < 0.18
    path_len = float(np.sum(np.linalg.norm(cam_centers_arr[1:] - cam_centers_arr[:-1], axis=1))) if N > 1 else 0.0

    # Eligibility classification
    if path_len < 1.0 and N >= 20:
        eligibility = "RECAPTURE_NEEDED"
        notes.append(f"Stationary camera pan detected: Total translation path length is only {path_len:.2f}m (< 1.0m threshold).")
        notes.append("Multi-view parallax requires genuine camera translation across the room.")
        notes.append("Recapture recommendation: Walk along the room perimeter or corridor rather than standing in one spot.")
    elif floor_ratio >= 0.75 and clearance_ratio >= 0.75 and max_step <= 0.15 and is_single and wall_pens == 0:
        eligibility = "WALKTHROUGH_READY"
        notes.append("Continuous floor evidence verified along trajectory.")
        notes.append("Safe obstacle clearance verified.")
        notes.append("Single-level elevation confirmed; first-person free-roam enabled.")
    elif floor_ratio >= 0.35 and clearance_ratio >= 0.40:
        eligibility = "PARTIAL_MODEL"
        notes.append(f"Partial floor evidence ({floor_ratio*100:.1f}% continuous coverage).")
        notes.append("Walkthrough restricted to verified sections; orbit inspection mode recommended.")
    else:
        eligibility = "RECAPTURE_NEEDED"
        if floor_ratio < 0.35:
            notes.append(f"Insufficient floor coverage ({floor_ratio*100:.1f}%) for reliable navigation.")
        if clearance_ratio < 0.40:
            notes.append(f"Severe obstacle occlusion / path blockage ({clearance_ratio*100:.1f}% clearance).")
        notes.append("Record the environment again: hold camera at steady height and record floor/doorways continuously.")

    route_res = RouteValidationResult(
        total_waypoints=N,
        path_length_m=round(path_len, 3),
        floor_continuity_ratio=round(floor_ratio, 4),
        obstacle_clearance_ratio=round(clearance_ratio, 4),
        max_step_height_m=round(float(max_step), 4),
        elevation_variance_m=round(elev_var, 4),
        is_single_level=is_single,
        floor_gaps_detected=floor_gaps,
        wall_penetrations_detected=wall_pens,
        eligibility_status=eligibility,
        actionable_notes=notes
    )

    controls = {
        "W": "Forward translation",
        "S": "Backward translation",
        "A": "Left translation",
        "D": "Right translation",
        "U": "Elevate vertically upward",
        "P": "Lower vertically downward",
        "Shift": "Sprint (1.9x speed)",
        "Esc": "Exit walkthrough to orbit inspection",
        "Mouse": "Look / aim (pitch and yaw)"
    }

    walkable_bounds = [
        round(float(bbox_min[0] + 0.05), 3),
        round(float(bbox_min[2] + 0.05), 3),
        round(float(bbox_max[0] - 0.05), 3),
        round(float(bbox_max[2] - 0.05), 3),
    ]

    return SceneNavigationMetadata(
        scene_id=scene_id,
        reconstructed_model_url=model_glb_url,
        bounds_xyz=bounds_xyz,
        floor_y=round(floor_y, 3),
        ceiling_y=round(ceiling_y, 3),
        room_height_m=round(room_height, 3),
        eye_height=eye_height,
        spawn_position=spawn_pos,
        spawn_yaw_deg=spawn_yaw,
        walkable_bounds_2d=walkable_bounds,
        collision_obstacles=obstacles,
        camera_trajectory=trajectory,
        route_validation=route_res,
        controls_guide=controls
    )
