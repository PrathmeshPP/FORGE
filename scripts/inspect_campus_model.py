#!/usr/bin/env python3
"""
Inspect Campus Candidate A reference GLB
"""
import trimesh
from pathlib import Path

CAMPUS_GLB = Path(r"C:\Users\25beevdt025\Desktop\FORGE\outputs\opencv5_experiments\phase2d\campus_0049\candidate_a\campus_0049_candidate_a_reconstruction.glb")

print(f"Loading {CAMPUS_GLB}...")
scene = trimesh.load(str(CAMPUS_GLB))
if isinstance(scene, trimesh.Scene):
    print(f"Scene geometries: {len(scene.geometry)}")
    for k, v in scene.geometry.items():
        print(f"  Geom '{k}': {len(v.vertices)} verts, {len(v.faces)} faces, visual: {type(v.visual)}")
        if hasattr(v.visual, 'material'):
            mat = v.visual.material
            print(f"    Material: {type(mat).__name__}")
            if hasattr(mat, 'image') and mat.image is not None:
                print(f"    Image: {mat.image.size}, mode: {mat.image.mode}")
            if hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                print(f"    baseColorTexture: {mat.baseColorTexture.size}")
else:
    print(f"Mesh: {len(scene.vertices)} verts, {len(scene.faces)} faces")
