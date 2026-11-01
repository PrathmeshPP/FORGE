import json
from pathlib import Path

NEW_SCENE = {
  "id": "indoor_nov1_poisson",
  "scene_name": "indoor_nov1_poisson",
  "display_name": "HOSTEL ROOM B 314 — POISSON MESH (NOVEMBER 1)",
  "subtitle": "Screened Poisson Surface Mesh (937,663 Vertices, 1,876,315 Triangles, Direct Vertex Colors)",
  "category": "Indoor Architecture",
  "acceptance_status": "PRODUCTION_PASS",
  "acceptance_label": "UNMODIFIED POISSON MESH DIGITAL TWIN",
  "description": "Exact unmodified Screened Poisson 3D surface mesh from WhatsApp Video 2026-11-01 at 14.28.11.mp4. Loaded directly from poisson_mesh.ply with zero geometric modification, zero decimations, and authentic vertex colors and normals. Navigable with first-person walkthrough controls, collision sliding, and 40-frame trajectory tracking.",
  "vertex_count": 937663,
  "triangle_count": 1876315,
  "default_representation": "photographic_ply",
  "coordinate_rotation": [3.141592653589793, 0.0, 0.0],
  "representations": {
    "photographic_ply": {
      "type": "ply",
      "label": "Poisson Mesh PLY (937k Vertices, 1.88M Triangles)",
      "ply_url": "/outputs/indoor/texture_experiment_video_20261101_142811/meshes/poisson_mesh.ply",
      "file_size_bytes": 72213235,
      "sha256": "80df04264c6addf8ac5df56bfc4169d2b9322b61126e64a6303678b4c2bbd3e5",
      "available": True
    }
  },
  "thumbnail_url": "/models/indoor_nov1_poisson/preview_thumbnail.png",
  "previews": {
    "eye_level": "/models/indoor_nov1_poisson/preview_eye_level.png",
    "top": "/models/indoor_nov1_poisson/preview_top.png"
  },
  "metadata_url": "/models/indoor_nov1_poisson/metadata.json",
  "initial_camera": {
    "position": [0.0, 1.42, -2.2],
    "target": [0.0, 1.30, 0.5]
  },
  "camera_presets": [
    {
      "id": "entrance",
      "label": "Entrance View",
      "description": "Standing at room entrance facing inward along central aisle",
      "position": [0.0, 1.42, -2.2],
      "target": [0.0, 1.30, 0.5]
    },
    {
      "id": "left_cot",
      "label": "Left Cot & Desk",
      "description": "Perspective focused on left cot and study desk",
      "position": [-0.8, 1.35, -0.2],
      "target": [-1.65, 1.10, 0.1]
    },
    {
      "id": "right_cot",
      "label": "Right Cot & Wardrobe",
      "description": "Perspective focused on right cot and metal wardrobes",
      "position": [0.8, 1.35, -0.2],
      "target": [1.65, 1.10, 0.1]
    },
    {
      "id": "overview",
      "label": "Room Overview",
      "description": "High-angle overview showing room bounds and ceiling",
      "position": [0.0, 3.4, -3.2],
      "target": [0.0, 1.2, 0.0]
    }
  ],
  "reconstruction_run_id": "texture_experiment_video_20261101_142811",
  "source_video_provenance": {
    "path": "C:\\Users\\25beevdt025\\Downloads\\WhatsApp Video 2026-11-01 at 14.28.11.mp4",
    "duration_sec": 73.13,
    "resolution": [848, 480],
    "fps": 25.0,
    "sha256": "1c02de602ff34bd63d7332f662dfb6fb73de5676a05ca07c2704f1fbe6c6ce35"
  }
}

manifest_paths = [
    Path("viewer/public/model-manifest.json"),
    Path("viewer/src/data/model-manifest.json")
]

for p in manifest_paths:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Remove existing if present to avoid duplicates
    data["scenes"] = [s for s in data["scenes"] if s.get("id") != "indoor_nov1_poisson"]
    
    # Insert at beginning
    data["scenes"].insert(0, NEW_SCENE)
    
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Successfully updated {p} (total scenes: {len(data['scenes'])})")

