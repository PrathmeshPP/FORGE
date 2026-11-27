"""
Automated Tests — F.O.R.G.E. V2 Milestones 6 & 7:
  - Test #18: Hidden completion preserves doorways/KNOWN_FREE_SPACE & corridor clearance.
  - Test #19: Observed and inferred surfaces remain cleanly separated with distinct provenance/material.
  - Test #20 & #21: Texture self-occlusion visibility, zero OOR UVs, and GLB reload integrity.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence
from src.reconstruction.hidden_surface_completion import (
    HiddenCompletionConfig,
    run_optional_hidden_surface_completion,
)
from src.reconstruction.planar_refinement import PlanarRegionRecord
from src.reconstruction.visibility_texture_baker import VisibilityAwareTextureBaker


class TestCompletionAndTextureBaker(unittest.TestCase):

    def test_18_and_19_hidden_completion_freespace_and_provenance(self):
        """
        Wall has TWO empty gaps:
          1) Occluded gap behind furniture (x in [-0.45, -0.15]) where NO free-space rays pass -> MUST be inferred.
          2) Doorway passage (x in [+0.25, +0.55]) where camera sees through at z=4.5m -> KNOWN_FREE_SPACE -> MUST NOT be filled!
        """
        xs, ys = np.meshgrid(np.linspace(-0.9, 0.9, 46), np.linspace(-0.5, 0.5, 26))
        zs = np.full_like(xs, 2.0)
        pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]).astype(np.float64)

        # Carve out both rectangular gaps from observed wall points
        gap_occl = (pts[:, 0] > -0.45) & (pts[:, 0] < -0.15) & (pts[:, 1] > -0.25) & (pts[:, 1] < 0.25)
        gap_door = (pts[:, 0] > 0.25) & (pts[:, 0] < 0.55) & (pts[:, 1] > -0.25) & (pts[:, 1] < 0.35)
        obs_pts = pts[~(gap_occl | gap_door)]

        # Create coarse observed mesh covering observed wall points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_pts)
        pcd.normals = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 0.0, -1.0]]), (len(obs_pts), 1)))
        mesh_obs = o3d.geometry.TriangleMesh()
        # Add tiny triangles centered on observed wall points
        verts, tris = [], []
        for idx, p in enumerate(obs_pts):
            b = idx * 3
            verts.extend([p, p + [0.03, 0, 0], p + [0, 0.03, 0]])
            tris.append([b, b + 1, b + 2])
        mesh_obs.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
        mesh_obs.triangles = o3d.utility.Vector3iVector(np.asarray(tris))

        # Camera at z=0.2 looks at scene: doorway sees d=4.5m background, occluded zone sees foreground bed at d=1.2m
        H, W = 64, 96
        K = np.array([[70.0, 0.0, 48.0], [0.0, 70.0, 32.0], [0.0, 0.0, 1.0]])
        ext = extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.2]))[None, :, :]
        intr = K[None, :, :]
        dmap = np.full((1, H, W, 1), 1.8, dtype=np.float32)
        cmap = np.full((1, H, W), 3.0, dtype=np.float32)
        # Rays through doorway (u=58..72) penetrate straight through z=2.0m up to z=4.5m!
        dmap[0, 18:48, 56:74, 0] = 4.5
        # Rays toward occluded wall behind bed stop at foreground d=1.0m!
        dmap[0, 20:44, 28:44, 0] = 1.0

        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.05)
        fse.build_from_depth_maps(dmap, cmap, ext, intr, max_depth=5.0, pixel_stride=1)

        wall_reg = PlanarRegionRecord(
            region_id=0, semantic_label="wall",
            point_indices=np.arange(len(obs_pts)),
            normal=np.array([0.0, 0.0, -1.0]), d_coeff=2.0,
            centroid=np.array([0.0, 0.0, 2.0]),
            bbox_min=obs_pts.min(axis=0), bbox_max=obs_pts.max(axis=0),
            mad_before_m=0.001, mad_after_m=0.001,
            crossview_support_ratio=1.0, supporting_views_count=1,
            free_space_conflict_ratio=0.0, accepted=True,
        )

        with tempfile.TemporaryDirectory() as td:
            res = run_optional_hidden_surface_completion(
                observed_mesh=mesh_obs, planar_regions=[wall_reg],
                observed_points=obs_pts, cam_centers=[np.array([0.0, 0.0, 0.2])],
                free_space_evidence=fse, out_dir=Path(td),
                config=HiddenCompletionConfig(enabled=True, grid_step_m=0.08),
            )
            self.assertGreater(res["report"]["rejected_freespace_barrier"], 0)
            self.assertGreater(res["report"]["accepted_cells_count"], 0)
            inf_v = np.asarray(res["inferred_mesh"].vertices)
            # Inferred vertices MUST NOT intrude into doorway x in [+0.25, +0.55]
            in_doorway = np.any((inf_v[:, 0] > 0.22) & (inf_v[:, 0] < 0.58))
            self.assertFalse(in_doorway, "Doorway KNOWN_FREE_SPACE must NEVER be filled by completion!")

    def test_20_and_21_visibility_texture_zbuffer_and_glb_reload(self):
        """Foreground wall must block camera from wrongly texturing occluded back panel; GLB must reload cleanly."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Create two distinct planar quads facing -Z: Front quad at z=1.2 (smaller, centered), Back quad at z=2.2 (larger)
            verts = np.array([
                [-0.25, -0.25, 1.2], [0.25, -0.25, 1.2], [0.25, 0.25, 1.2], [-0.25, 0.25, 1.2],  # Front
                [-0.90, -0.60, 2.2], [0.90, -0.60, 2.2], [0.90, 0.60, 2.2], [-0.90, 0.60, 2.2],  # Back
            ], dtype=np.float64)
            tris = np.array([[0, 2, 1], [0, 3, 2], [4, 6, 5], [4, 7, 6]], dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(tris)

            # Side quad to give unambiguous multi-face export + inferred surface test
            H, W = 120, 160
            img1_p = td_p / "cam0.png"
            img2_p = td_p / "cam1.png"
            Image.new("RGB", (W, H), (220, 40, 40)).save(img1_p)
            Image.new("RGB", (W, H), (40, 120, 220)).save(img2_p)

            K_inf = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
            # Cam 0 at center z=0 (blocked at center by Front quad); Cam 1 at x=+1.2, yawed to see Back quad cleanly!
            R_side = np.array([[0.88, 0.0, 0.47], [0.0, 1.0, 0.0], [-0.47, 0.0, 0.88]])
            ext = np.stack([
                extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.0])),
                extrinsic_from_center_and_rotation(R_side, np.array([1.2, 0.0, 0.0])),
            ])
            intr = np.tile(K_inf[None, :, :], (2, 1, 1))
            dmaps = np.full((2, H, W, 1), 2.2, dtype=np.float32)

            baker = VisibilityAwareTextureBaker(mesh, ext, intr, [img1_p, img2_p], depth_maps=dmaps, min_cos_theta=0.15)
            best_cam, scores, tstats = baker.compute_visibility_and_best_cameras()
            # Front quad (faces 0,1) seen well from Cam 0; Back quad faces behind front quad have zero/low score from Cam 0 due to BVH self-occlusion!
            self.assertGreater(tstats["zbuffer_self_occluded_tests"], 0)

            atlas_p = td_p / "atlas.png"
            rects, _ = baker.synthesize_atlas(best_cam, atlas_p, atlas_size=1024)
            # Also pass an inferred quad to test Test #19 separate provenance material!
            inf_m = o3d.geometry.TriangleMesh()
            inf_m.vertices = o3d.utility.Vector3dVector(np.array([[-0.2, 0.5, 2.2], [0.2, 0.5, 2.2], [0.0, 0.7, 2.2]]))
            inf_m.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2]]))

            deliv, tqa = baker.export_canonical_deliverables(
                td_p, "test_room", best_cam, rects, atlas_p, atlas_size=1024, inferred_mesh=inf_m
            )
            self.assertEqual(tqa["out_of_range_uv_count"], 0)
            self.assertTrue(tqa["glb_reload_valid"])
            self.assertIn("Material_Inferred_Extrapolated", Path(deliv["completed_mtl"]).read_text())
            self.assertNotIn("Material_Inferred_Extrapolated", Path(deliv["mtl"]).read_text())

    def test_21b_cross_camera_seam_exposure_consistency(self):
        """Seam-coupled cameras with +20% brightness offset receive bounded equalizing gain in [0.85, 1.18]."""
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Large continuous wall quad grid so triangles meet along shared edges
            verts = np.array([
                [-0.8, -0.5, 2.0], [0.0, -0.5, 2.0], [0.8, -0.5, 2.0],
                [-0.8,  0.5, 2.0], [0.0,  0.5, 2.0], [0.8,  0.5, 2.0],
            ], dtype=np.float64)
            tris = np.array([[0, 3, 1], [1, 3, 4], [1, 4, 2], [2, 4, 5]], dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(tris)

            H, W = 120, 160
            img0_p = td_p / "c0.png"
            img1_p = td_p / "c1.png"
            Image.new("RGB", (W, H), (140, 140, 140)).save(img0_p)
            Image.new("RGB", (W, H), (175, 175, 175)).save(img1_p)  # +25% brighter exposure

            K_inf = np.array([[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]])
            ext = np.stack([
                extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.2, 0.0, 0.0])),
                extrinsic_from_center_and_rotation(np.eye(3), np.array([0.2, 0.0, 0.0])),
            ])
            intr = np.tile(K_inf[None, :, :], (2, 1, 1))
            dmaps = np.full((2, H, W, 1), 2.0, dtype=np.float32)

            baker = VisibilityAwareTextureBaker(mesh, ext, intr, [img0_p, img1_p], depth_maps=dmaps)
            best_cam = np.array([0, 0, 1, 1], dtype=np.int32)
            gains, estats = baker._estimate_exposure_gains(best_cam)
            self.assertTrue(estats["applied"], f"Expected exposure equalization across shared edge, got {estats}")
            self.assertGreater(gains[0], gains[1])
            self.assertGreaterEqual(gains[1], 0.85)
            self.assertLessEqual(gains[0], 1.18)


if __name__ == "__main__":
    unittest.main()
