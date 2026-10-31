"""
F.O.R.G.E. - Indoor Walkthrough Navigation, Floor Plan & Collision Precomputation.
Generates:
1. Canonical camera trajectory with synchronized frame URLs & timestamps.
2. 2D top-down floor plan grid & obstacle vector polygons (bed, desk, wardrobe, walls, doors).
3. Collision boundaries & walkable navigation bounds.
4. Pre-rendered 2D floor plan bitmap for crisp mini-map rendering.
"""

from pathlib import Path
import json
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

EXP_BASE = Path("outputs/indoor/texture_experiment_video1_reproduced_video_170234")
OUT_DIR = Path("outputs/indoor/photographic_texture_quality_recovery_video1")
VIEWER_PUBLIC = Path("viewer/public/data")
VIEWER_SRC = Path("viewer/src/data")

for d in [VIEWER_PUBLIC, VIEWER_SRC]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Camera Poses & Canonical Trajectory
cam_npz = np.load(EXP_BASE / "camera_geometry/camera_poses.npz")
extr = cam_npz["extrinsics"]

with open(EXP_BASE / "plane_detection.json") as fp:
    plane = json.load(fp)

n = np.array(plane["detected_planes"][0]["normal"], dtype=np.float64)
target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
v = np.cross(n, target)
s = np.linalg.norm(v)
c = np.dot(n, target)
vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R_floor = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2 + 1e-8))

centers_raw = np.array([-extr[i, :3, :3].T @ extr[i, :3, 3] for i in range(len(extr))])
centers_leveled = (R_floor @ centers_raw.T).T

mesh = o3d.io.read_triangle_mesh(str(OUT_DIR / "meshes/mesh_decimated_leveled.ply"))
verts = np.asarray(mesh.vertices)
T_canon = np.array([1.0, -1.0, -1.0])
center = (verts * T_canon).mean(axis=0)

centers_canonical = (centers_leveled * T_canon) - center

with open(EXP_BASE / "frame_sampling_report.json") as fr:
    sampling = json.load(fr)
keyframes = sampling.get("selected_keyframes", [])
frame_paths = sorted(list((EXP_BASE / "frames").glob("*.png")))

trajectory = []
for i in range(len(extr)):
    pos = centers_canonical[i]
    R_eff = extr[i, :3, :3] @ R_floor.T
    fwd = (R_eff.T @ np.array([0, 0, 1.0])) * T_canon
    kf_info = keyframes[i] if i < len(keyframes) else {}
    rel_path = f"/outputs/indoor/texture_experiment_video1_reproduced_video_170234/frames/{frame_paths[i].name}"

    trajectory.append({
        "index": i,
        "position": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
        "forward": [round(float(fwd[0]), 4), round(float(fwd[1]), 4), round(float(fwd[2]), 4)],
        "timestamp_sec": kf_info.get("timestamp_sec", round(i * 0.64, 2)),
        "frame_url": rel_path,
        "filename": frame_paths[i].name
    })

# 2. Geometry Slicing & Floor Plan
tris = np.asarray(mesh.triangles)
verts_canon = (verts * T_canon) - center
tri_v = verts_canon[tris]
centroids = tri_v.mean(axis=1)

v0 = tri_v[:, 0]
v1 = tri_v[:, 1]
v2 = tri_v[:, 2]
cross = np.cross(v1 - v0, v2 - v0)
norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-12
normals = cross / norms

# Floor slice
floor_mask = (normals[:, 1] > 0.6) & (centroids[:, 1] <= -0.42)
floor_pts = centroids[floor_mask]

# Obstacle slice (Y in [-0.40, 0.25])
obst_mask = (centroids[:, 1] > -0.40) & (centroids[:, 1] <= 0.25)
obst_pts = centroids[obst_mask]

# 2D Grid Setup
grid_res = 0.04 # 4 cm per pixel
x_min, x_max = -1.30, 1.20
z_min, z_max = -2.30, 1.50
W_g = int(np.ceil((x_max - x_min) / grid_res))
H_g = int(np.ceil((z_max - z_min) / grid_res))

floor_img = np.zeros((H_g, W_g), dtype=np.uint8)
obst_img = np.zeros((H_g, W_g), dtype=np.uint8)

fx = np.clip(((floor_pts[:, 0] - x_min) / grid_res).astype(int), 0, W_g - 1)
fz = np.clip(((floor_pts[:, 2] - z_min) / grid_res).astype(int), 0, H_g - 1)
floor_img[fz, fx] = 255

bx = np.clip(((obst_pts[:, 0] - x_min) / grid_res).astype(int), 0, W_g - 1)
bz = np.clip(((obst_pts[:, 2] - z_min) / grid_res).astype(int), 0, H_g - 1)
obst_img[bz, bx] = 255

# Morphological closing to fill small scan gaps in floor
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
floor_closed = cv2.morphologyEx(floor_img, cv2.MORPH_CLOSE, kernel_close)

# Dilate obstacles for player collision margin (radius = 0.18m = ~5 pixels)
kernel_player = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
obst_dilated = cv2.dilate(obst_img, kernel_player)

# Walkable floor
walkable = (floor_closed > 0) & (obst_dilated == 0)

# Unseen / unscanned space (outside the room footprint)
# Find outer bounding contour of scanned space
contours, _ = cv2.findContours(floor_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
scanned_mask = np.zeros((H_g, W_g), dtype=np.uint8)
if contours:
    cv2.drawContours(scanned_mask, contours, -1, 255, -1)
unseen = scanned_mask == 0

# 3. Key Architectural Regions & Obstacle Bounding Boxes (2D)
# Defined in canonical space: [x_min, z_min, x_max, z_max]
regions = [
    {
        "id": "bed",
        "name": "Hostel Bed & Mattress",
        "type": "obstacle",
        "bounds": [-1.15, -0.95, -0.12, 0.45],
        "height_range": [-0.50, 0.05],
        "label": "BED"
    },
    {
        "id": "desk_shelf",
        "name": "Study Desk, Laptop & Shelves",
        "type": "obstacle",
        "bounds": [-0.85, 0.20, 0.15, 1.15],
        "height_range": [-0.50, 0.35],
        "label": "DESK"
    },
    {
        "id": "wardrobe",
        "name": "Wooden Wardrobe & Cupboard",
        "type": "obstacle",
        "bounds": [0.45, -1.80, 1.10, -0.60],
        "height_range": [-0.50, 0.65],
        "label": "WARDROBE"
    },
    {
        "id": "door_threshold",
        "name": "Room Entrance & Doorway",
        "type": "doorway",
        "bounds": [0.05, -2.10, 0.50, -1.50],
        "height_range": [-0.50, 0.65],
        "label": "DOORWAY"
    },
    {
        "id": "walkable_corridor",
        "name": "Walkable Interior Aisle",
        "type": "walkway",
        "bounds": [-0.08, -1.95, 0.52, 0.55],
        "height_range": [-0.50, 0.65],
        "label": "WALKWAY"
    }
]

# 4. Render Pre-computed Floor Plan Bitmap (for high-contrast mini-map)
map_canvas = Image.new("RGBA", (W_g * 4, H_g * 4), color=(15, 23, 42, 240))
map_draw = ImageDraw.Draw(map_canvas)

# Scale factor for drawing
S = 4

# Draw unseen hatched background
for z in range(0, H_g * S, 12):
    map_draw.line([(0, z), (W_g * S, z + W_g * S)], fill=(30, 41, 59, 100), width=1)

# Draw floor area
floor_resized = cv2.resize(floor_closed, (W_g * S, H_g * S), interpolation=cv2.INTER_NEAREST)
for y in range(0, H_g * S, 2):
    for x in range(0, W_g * S, 2):
        if floor_resized[y, x] > 0:
            map_draw.point((x, y), fill=(24, 38, 64, 255))

# Draw walkable corridor
walk_resized = cv2.resize((walkable.astype(np.uint8) * 255), (W_g * S, H_g * S), interpolation=cv2.INTER_NEAREST)
for y in range(0, H_g * S, 2):
    for x in range(0, W_g * S, 2):
        if walk_resized[y, x] > 0:
            map_draw.point((x, y), fill=(34, 197, 94, 60))

# Draw obstacles in red/orange outlines
obst_resized = cv2.resize(obst_img, (W_g * S, H_g * S), interpolation=cv2.INTER_NEAREST)
for y in range(0, H_g * S, 2):
    for x in range(0, W_g * S, 2):
        if obst_resized[y, x] > 0:
            map_draw.point((x, y), fill=(239, 68, 68, 180))

# Draw camera trajectory route
traj_pixels = []
for pt in trajectory:
    px = int(((pt["position"][0] - x_min) / grid_res) * S)
    pz = int(((pt["position"][2] - z_min) / grid_res) * S)
    traj_pixels.append((px, pz))

for i in range(len(traj_pixels) - 1):
    map_draw.line([traj_pixels[i], traj_pixels[i + 1]], fill=(56, 189, 248, 220), width=2)

for i, pt in enumerate(traj_pixels):
    r = 2 if i % 5 != 0 else 4
    col = (255, 255, 255, 255) if i == 0 else ((56, 189, 248, 200) if i % 5 != 0 else (250, 204, 21, 255))
    map_draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=col)

# Label regions
for reg in regions:
    b = reg["bounds"]
    x1 = int(((b[0] - x_min) / grid_res) * S)
    z1 = int(((b[1] - z_min) / grid_res) * S)
    x2 = int(((b[2] - x_min) / grid_res) * S)
    z2 = int(((b[3] - z_min) / grid_res) * S)
    
    col = (56, 189, 248) if reg["type"] == "walkway" else ((34, 197, 94) if reg["type"] == "doorway" else (248, 113, 113))
    map_draw.rectangle([x1, z1, x2, z2], outline=col, width=1)
    map_draw.text((x1 + 4, z1 + 4), reg["label"], fill=col)

map_png_path = OUT_DIR / "floor_plan_minimap.png"
map_canvas.save(map_png_path)
print(f"[+] Saved Floor Plan Mini-Map: {map_png_path}")

# Copy to public & artifacts
map_canvas.save(VIEWER_PUBLIC / "floor_plan_minimap.png")
map_canvas.save(Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953/floor_plan_minimap.png"))

# 5. Assemble Navigation Metadata JSON
metadata = {
    "version": "1.0.0",
    "scene_id": "indoor_candidate_b_sharpest",
    "coordinate_system": "canonical (X: right, Y: up, Z: forward)",
    "floor_elevation": {
        "mean_y": -0.4867,
        "min_y": -0.55,
        "max_y": -0.42,
        "ceiling_y": 0.6346,
        "room_height_m": 1.1213
    },
    "player_physics": {
        "spawn_position": [0.1198, 0.0939, -1.2724],
        "spawn_heading_rad": 3.1415,
        "eye_height": 0.58,
        "player_radius": 0.18,
        "walk_speed": 1.15,
        "sprint_speed": 2.10,
        "gravity": 9.8,
        "collision_sliding": True,
        "step_height_max": 0.08
    },
    "spatial_bounds": {
        "x_min": round(float(x_min), 3),
        "x_max": round(float(x_max), 3),
        "z_min": round(float(z_min), 3),
        "z_max": round(float(z_max), 3),
        "grid_resolution": grid_res,
        "grid_dimensions": [W_g, H_g],
        "room_width_m": round(float(x_max - x_min), 2),
        "room_length_m": round(float(z_max - z_min), 2)
    },
    "architectural_regions": regions,
    "camera_trajectory": trajectory,
    "total_trajectory_points": len(trajectory)
}

for out_json in [
    OUT_DIR / "indoor_walkthrough_metadata.json",
    VIEWER_PUBLIC / "indoor_walkthrough_metadata.json",
    VIEWER_SRC / "indoor_walkthrough_metadata.json"
]:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved walkthrough metadata: {out_json}")

print("[+] Indoor walkthrough precomputation finished successfully!")

