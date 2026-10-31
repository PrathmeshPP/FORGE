"""
Generates Collision Geometry and Spatial Occupancy Floor-Plan Representations
for Hostel Room B 314 under Rigid Metric Geometry.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
import trimesh

EXP_DIR = Path("outputs/indoor/geometry_quality_improvement_video1")
DIAG_DIR = EXP_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def generate_spatial_representations():
    print("[*] Generating Collision Geometry and Spatial Occupancy Representation...")
    
    # 1. Load Candidate D mesh
    mesh_path = EXP_DIR / "candidate_d_rigid_poisson_hq.glb"
    scene = trimesh.load(str(mesh_path), force='scene')
    for geom in scene.geometry.values():
        if isinstance(geom, trimesh.Trimesh):
            mesh = geom
            break
            
    verts = mesh.vertices
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    
    floor_y = -0.51
    ceiling_y = 1.88
    
    # 2. Collision Bounds (Strict non-penetrating bounding volumes)
    # Define exact bounding boxes based on measured coordinates:
    # A. Floor collider
    # B. Ceiling collider
    # C. Left Wall (X min)
    # D. Right Wall (X max)
    # E. Back Wall (Z max)
    # F. Entry / Front Wall (Z min)
    # G. Bed collider
    # H. Desk collider
    # I. Wardrobe collider
    
    collision_boxes = [
        {
            "name": "Floor_Plane",
            "type": "floor",
            "box": {
                "center": [float((bounds_min[0] + bounds_max[0])/2.0), float(floor_y - 0.05), float((bounds_min[2] + bounds_max[2])/2.0)],
                "size": [float(bounds_max[0] - bounds_min[0] + 0.2), 0.10, float(bounds_max[2] - bounds_min[2] + 0.2)]
            }
        },
        {
            "name": "Ceiling_Plane",
            "type": "ceiling",
            "box": {
                "center": [float((bounds_min[0] + bounds_max[0])/2.0), float(ceiling_y + 0.05), float((bounds_min[2] + bounds_max[2])/2.0)],
                "size": [float(bounds_max[0] - bounds_min[0] + 0.2), 0.10, float(bounds_max[2] - bounds_min[2] + 0.2)]
            }
        },
        {
            "name": "Left_Wall",
            "type": "wall",
            "box": {
                "center": [float(bounds_min[0] + 0.05), float((floor_y + ceiling_y)/2.0), float((bounds_min[2] + bounds_max[2])/2.0)],
                "size": [0.10, float(ceiling_y - floor_y), float(bounds_max[2] - bounds_min[2])]
            }
        },
        {
            "name": "Right_Wall",
            "type": "wall",
            "box": {
                "center": [float(bounds_max[0] - 0.05), float((floor_y + ceiling_y)/2.0), float((bounds_min[2] + bounds_max[2])/2.0)],
                "size": [0.10, float(ceiling_y - floor_y), float(bounds_max[2] - bounds_min[2])]
            }
        },
        {
            "name": "Back_Window_Wall",
            "type": "wall",
            "box": {
                "center": [float((bounds_min[0] + bounds_max[0])/2.0), float((floor_y + ceiling_y)/2.0), float(bounds_max[2] - 0.05)],
                "size": [float(bounds_max[0] - bounds_min[0]), float(ceiling_y - floor_y), 0.10]
            }
        },
        {
            "name": "Front_Door_Threshold",
            "type": "wall",
            "box": {
                "center": [float((bounds_min[0] + bounds_max[0])/2.0), float((floor_y + ceiling_y)/2.0), float(bounds_min[2] + 0.05)],
                "size": [float(bounds_max[0] - bounds_min[0]), float(ceiling_y - floor_y), 0.10]
            }
        },
        {
            "name": "Bed_Solid_Volume",
            "type": "furniture",
            "box": {
                "center": [0.95, float(floor_y + 0.35), 0.30],
                "size": [0.90, 0.70, 2.05]
            }
        },
        {
            "name": "Desk_Solid_Volume",
            "type": "furniture",
            "box": {
                "center": [0.00, float(floor_y + 0.45), 1.25],
                "size": [1.65, 0.90, 0.75]
            }
        },
        {
            "name": "Wardrobe_Solid_Volume",
            "type": "furniture",
            "box": {
                "center": [-1.05, float(floor_y + 1.00), -0.85],
                "size": [0.85, 2.00, 1.10]
            }
        }
    ]
    
    col_path = EXP_DIR / "collision_bounds.json"
    with open(col_path, "w") as f:
        json.dump({"room": "Hostel Room B 314", "colliders": collision_boxes}, f, indent=2)
    print(f"[+] Saved collision bounds: {col_path}")
    
    # 3. Spatial Occupancy 2D / 2.5D Floor Plan
    # Grid resolution: 2.0 cm per pixel
    res = 0.02 # 2 cm
    x_min, x_max = -2.0, 2.5
    z_min, z_max = -5.5, 2.0
    
    w_pixels = int(np.ceil((x_max - x_min) / res))
    h_pixels = int(np.ceil((z_max - z_min) / res))
    
    # 0 = Unseen / Outside (Slate Gray)
    # 1 = Walkable Free Floor (Light Emerald / Mint Green)
    # 2 = Occupied Solid Obstacle (Coral / Crimson Red)
    # 3 = Camera Capture Trajectory (Bright Cyan)
    grid = np.zeros((h_pixels, w_pixels), dtype=np.uint8)
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(EXP_DIR / "point_clouds" / "point_cloud_rigid_clean.ply"))
    pts = np.asarray(pcd.points)
    
    # Floor points: Y near floor
    p_floor = pts[(pts[:, 1] >= floor_y - 0.10) & (pts[:, 1] <= floor_y + 0.15)]
    # Obstacle points: Y between floor + 0.15 and ceiling - 0.20
    p_obs = pts[(pts[:, 1] > floor_y + 0.15) & (pts[:, 1] < ceiling_y - 0.20)]
    
    # Map floor points to grid
    ix_floor = np.clip(((p_floor[:, 0] - x_min) / res).astype(int), 0, w_pixels - 1)
    iz_floor = np.clip(((p_floor[:, 2] - z_min) / res).astype(int), 0, h_pixels - 1)
    grid[iz_floor, ix_floor] = 1 # Walkable floor
    
    # Dilation for contiguous walkable floor
    from scipy.ndimage import binary_dilation, binary_closing
    walkable_mask = binary_closing(grid == 1, iterations=3)
    grid[walkable_mask] = 1
    
    # Map obstacles
    ix_obs = np.clip(((p_obs[:, 0] - x_min) / res).astype(int), 0, w_pixels - 1)
    iz_obs = np.clip(((p_obs[:, 2] - z_min) / res).astype(int), 0, h_pixels - 1)
    grid[iz_obs, ix_obs] = 2 # Obstacle
    
    # Obstacle dilation slightly for safety buffer
    obs_mask = binary_dilation(grid == 2, iterations=1)
    grid[obs_mask] = 2
    
    # Render rich RGB floor plan image
    img = np.zeros((h_pixels, w_pixels, 3), dtype=np.uint8)
    # Unseen: Dark background [15, 23, 42]
    img[grid == 0] = [15, 23, 42]
    # Walkable Free Floor: [34, 197, 94] (Emerald Green)
    img[grid == 1] = [34, 197, 94]
    # Obstacles: [239, 68, 68] (Coral Red)
    img[grid == 2] = [239, 68, 68]
    
    # Render camera trajectory
    poses_npz = np.load(str(EXP_DIR / "camera_geometry" / "camera_poses_rigid.npz"))
    extrinsics = poses_npz["extrinsics"]
    cam_centers = []
    for E in extrinsics:
        R = E[:3, :3]
        t = E[:3, 3]
        C = -R.T @ t
        cam_centers.append(C)
    cam_centers = np.array(cam_centers)
    
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    traj_pts = []
    for c in cam_centers:
        cx = int((c[0] - x_min) / res)
        cz = int((c[2] - z_min) / res)
        traj_pts.append((cx, cz))
        
    for i in range(len(traj_pts) - 1):
        draw.line([traj_pts[i], traj_pts[i+1]], fill=(56, 189, 248), width=3)
    for p in traj_pts:
        draw.ellipse([p[0]-3, p[1]-3, p[0]+3, p[1]+3], fill=(255, 255, 255), outline=(14, 165, 233))
        
    # Rotate 180 or flip so Z points upwards
    pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Add annotated labels & legend
    canvas = Image.new("RGB", (pil_img.width + 320, pil_img.height + 80), color=(11, 15, 25))
    canvas.paste(pil_img, (40, 40))
    d = ImageDraw.Draw(canvas)
    
    # Header
    d.text((40, 12), "F.O.R.G.E. SPATIAL OCCUPANCY & METRIC FLOOR PLAN (Hostel Room B 314)", fill=(56, 189, 248))
    
    # Legend on right side
    lx = pil_img.width + 60
    ly = 60
    d.text((lx, ly), "SPATIAL CLASSIFICATION", fill=(255, 255, 255))
    
    # Walkable
    d.rectangle([lx, ly + 35, lx + 24, ly + 55], fill=(34, 197, 94))
    d.text((lx + 34, ly + 37), "Walkable Free Floor", fill=(203, 213, 225))
    
    # Obstacles
    d.rectangle([lx, ly + 75, lx + 24, ly + 95], fill=(239, 68, 68))
    d.text((lx + 34, ly + 77), "Solid Obstacle (Bed/Desk/Wardrobe)", fill=(203, 213, 225))
    
    # Unseen
    d.rectangle([lx, ly + 115, lx + 24, ly + 135], fill=(15, 23, 42), outline=(100, 116, 139))
    d.text((lx + 34, ly + 117), "Unseen / Outside Perimeter", fill=(203, 213, 225))
    
    # Trajectory
    d.line([lx, ly + 165, lx + 24, ly + 165], fill=(56, 189, 248), width=3)
    d.ellipse([lx + 9, ly + 162, lx + 15, ly + 168], fill=(255, 255, 255))
    d.text((lx + 34, ly + 157), "Camera Capture Trajectory", fill=(203, 213, 225))
    
    # Room Metric Specs
    d.text((lx, ly + 210), "METRIC DIMENSIONS", fill=(255, 255, 255))
    d.text((lx, ly + 235), f"Room Width (X): {bounds_max[0] - bounds_min[0]:.2f} m", fill=(148, 163, 184))
    d.text((lx, ly + 260), f"Room Depth (Z): {bounds_max[2] - bounds_min[2]:.2f} m", fill=(148, 163, 184))
    d.text((lx, ly + 285), f"Ceiling Height (Y): {ceiling_y - floor_y:.2f} m", fill=(148, 163, 184))
    d.text((lx, ly + 310), f"Bed Footprint: 2.05m x 0.90m", fill=(148, 163, 184))
    d.text((lx, ly + 335), f"Desk Footprint: 1.65m x 0.75m", fill=(148, 163, 184))
    d.text((lx, ly + 360), f"Grid Resolution: 2.0 cm/px", fill=(148, 163, 184))
    
    out_img_path = EXP_DIR / "floor_plan_occupancy.png"
    canvas.save(out_img_path)
    print(f"[+] Saved spatial occupancy floor plan: {out_img_path}")
    
    # Copy to artifacts directory
    art_path = Path("C:/Users/25beevdt025/.gemini/antigravity/brain/536a497a-fba6-496d-b4a9-b4f0cf756953/floor_plan_occupancy.png")
    canvas.save(art_path)
    print(f"[+] Exported artifact: {art_path}")

if __name__ == "__main__":
    generate_spatial_representations()
