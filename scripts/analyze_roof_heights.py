#!/usr/bin/env python3
"""
Inspect the real building roof coordinates and features
"""
import trimesh
import numpy as np
from pathlib import Path

V8_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\new_video_reconstructions\whatsapp_video_2026_10_24\best_reconstruction_campaign\final_best_canonical_v8")
mesh = trimesh.load(str(V8_DIR / "final_best_canonical_v8_mesh.ply"), process=False)

components = mesh.split(only_watertight=False)

print(f"Total components: {len(components)}")
# Look at Component #0 (the main building & ground)
c0 = components[0]
print(f"Component #0 bounds:")
print(f"  X: [{c0.bounds[0,0]:.3f}, {c0.bounds[1,0]:.3f}]")
print(f"  Y: [{c0.bounds[0,1]:.3f}, {c0.bounds[1,1]:.3f}]")
print(f"  Z: [{c0.bounds[0,2]:.3f}, {c0.bounds[1,2]:.3f}]")

# What is at the top of Component #0?
y0 = c0.vertices[:, 1]
high_verts_c0 = c0.vertices[y0 > 0.15]
print(f"Comp #0 vertices with Y > 0.15: {len(high_verts_c0):,} verts, Y max={np.max(y0):.3f}")
print(f"  X range for high verts: [{high_verts_c0[:,0].min():.3f}, {high_verts_c0[:,0].max():.3f}]")
print(f"  Z range for high verts: [{high_verts_c0[:,2].min():.3f}, {high_verts_c0[:,2].max():.3f}]")

# Check all components with bounds[1, 1] > 0.15
for i, c in enumerate(components):
    if c.bounds[1, 1] > 0.15:
        print(f"\nComp #{i}: {len(c.vertices)} verts, {len(c.faces)} faces")
        print(f"  Center: {np.mean(c.vertices, axis=0)}")
        print(f"  Bounds: X [{c.bounds[0,0]:.3f}, {c.bounds[1,0]:.3f}], Y [{c.bounds[0,1]:.3f}, {c.bounds[1,1]:.3f}], Z [{c.bounds[0,2]:.3f}, {c.bounds[1,2]:.3f}]")
        # Is it connected to the roof or floating above?
        # Minimum Y of this component vs maximum Y of c0 in the same X, Z column
        x_mask = (c0.vertices[:, 0] >= c.bounds[0, 0]) & (c0.vertices[:, 0] <= c.bounds[1, 0])
        z_mask = (c0.vertices[:, 2] >= c.bounds[0, 2]) & (c0.vertices[:, 2] <= c.bounds[1, 2])
        col_mask = x_mask & z_mask
        if np.any(col_mask):
            c0_col_max_y = np.max(c0.vertices[col_mask, 1])
            gap = c.bounds[0, 1] - c0_col_max_y
            print(f"  c0 maximum Y directly below Comp #{i}: {c0_col_max_y:.3f} m. Gap = {gap:.3f} m")
        else:
            print(f"  No c0 vertices directly below Comp #{i}!")
