import json
import numpy as np
from pathlib import Path

META_PATH = Path("viewer/src/data/indoor_walkthrough_metadata.json")
POSES_PATH = Path("outputs/indoor/texture_experiment_video_20261101_142811/camera_geometry/camera_poses.npz")
FRAMES_DIR = Path("outputs/indoor/texture_experiment_video_20261101_142811/frames")

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Build architectural_regions for MiniMap and collision sliding
regions = []

# Doorway
regions.append({
    "id": "doorway_entrance",
    "name": "Entrance Doorway",
    "type": "doorway",
    "bounds": [-0.60, -3.20, 0.60, -2.40],
    "height_range": [0.0, 2.10],
    "label": "Entrance"
})

# Walkway
regions.append({
    "id": "central_aisle",
    "name": "Central Aisle Walkway",
    "type": "walkway",
    "bounds": [-0.95, -2.40, 0.95, 2.60],
    "height_range": [0.0, 2.85],
    "label": "Walkway"
})

# Add obstacles from collision_boxes (beds, desks, wardrobes, cooler, chairs)
for b in meta.get("collision_boxes", []):
    bid = b["id"]
    if bid.startswith("wall_"):
        continue
    x1, y1, z1 = b["min"]
    x2, y2, z2 = b["max"]
    label = b.get("name") or bid
    regions.append({
        "id": bid,
        "name": label,
        "type": "obstacle",
        "bounds": [round(x1, 2), round(z1, 2), round(x2, 2), round(z2, 2)],
        "height_range": [round(y1, 2), round(y2, 2)],
        "label": label
    })

meta["architectural_regions"] = regions

# Add camera_trajectory
if POSES_PATH.exists():
    data = np.load(POSES_PATH)
    ext = data["extrinsics"]
    frame_files = sorted([f.name for f in FRAMES_DIR.glob("*.png")]) if FRAMES_DIR.exists() else []
    
    trajectory = []
    for i in range(len(ext)):
        R = ext[i, :, :3]
        t = ext[i, :, 3]
        c = -R.T @ t # raw center
        # Canonical transform T = diag(1, -1, -1)
        c_canon = [c[0], -c[1], -c[2]]
        
        # Camera forward in canonical space
        fwd_raw = R.T[:, 2]
        fwd_canon = [fwd_raw[0], -fwd_raw[1], -fwd_raw[2]]
        
        fname = frame_files[i] if i < len(frame_files) else f"frame_{i:03d}.png"
        trajectory.append({
            "index": i,
            "position": [round(float(c_canon[0]), 4), round(float(c_canon[1]), 4), round(float(c_canon[2]), 4)],
            "forward": [round(float(fwd_canon[0]), 4), round(float(fwd_canon[1]), 4), round(float(fwd_canon[2]), 4)],
            "timestamp_sec": round(i * 0.5, 2),
            "frame_url": f"/outputs/indoor/texture_experiment_video_20261101_142811/frames/{fname}",
            "filename": fname
        })
    meta["camera_trajectory"] = trajectory

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"Updated {META_PATH} with {len(regions)} regions and {len(meta.get('camera_trajectory', []))} trajectory points.")

