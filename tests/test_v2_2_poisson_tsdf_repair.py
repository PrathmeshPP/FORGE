"""
Automated Tests — F.O.R.G.E. V2.2 Conditional Local Poisson, B1 Planar CDT,
TSDF Depth-Sync/Cluster Preservation, Boundary Stitching, and Per-Patch + Global Mesh QA.
Covers all 24 deterministic V2.2 test scenarios.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.conditional_gap_repair import (
    align_patch_winding_to_mesh,
    assemble_stitched_patch_into_mesh,
    build_boundary_pinned_planar_patch,
    build_polygon_clipped_stitched_poisson_patch,
    verify_patch_local_qa,
)
from src.reconstruction.depth_consistency_filter import FreeSpaceVoxelEvidence
from src.reconstruction.mesh_branches_and_qa import (
    MeshBranchQAConfig,
    _clean_mesh_topology_with_stats,
    analyze_tsdf_boundary_holes,
    build_canonical_scene_geometry,
    build_global_tsdf_branch,
    run_conditional_localized_repair,
    run_global_tsdf_and_conditional_repair,
)
from src.reconstruction.planar_refinement import project_canonical_points_to_depth_maps
from src.reconstruction.visibility_texture_baker import VisibilityAwareTextureBaker


def _make_grid_mesh_with_hole(
    nx: int = 21,
    ny: int = 21,
    span_x: float = 0.8,
    span_y: float = 0.8,
    hole_box: tuple[float, float, float, float] | None = (-0.12, 0.12, -0.12, 0.12),
    z_fn=None,
) -> o3d.geometry.TriangleMesh:
    """Builds a clean triangulated wall mesh on [-span_x, span_x] x [-span_y, span_y] with optional hole cutout."""
    xs = np.linspace(-span_x, span_x, nx)
    ys = np.linspace(-span_y, span_y, ny)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, 2.0) if z_fn is None else z_fn(xx, yy)
    verts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    tris = []
    for r in range(ny - 1):
        for c in range(nx - 1):
            i0 = r * nx + c
            i1 = i0 + 1
            i2 = i0 + nx
            i3 = i2 + 1
            for tri in ([i0, i1, i3], [i0, i3, i2]):
                cent = verts[tri].mean(axis=0)
                if hole_box is not None:
                    hx0, hx1, hy0, hy1 = hole_box
                    if hx0 <= cent[0] <= hx1 and hy0 <= cent[1] <= hy1:
                        continue
                tris.append(tri)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _make_dummy_scene_geom(
    pts: np.ndarray,
    free_space: FreeSpaceVoxelEvidence | None = None,
    voxel_size: float = 0.02,
) -> object:
    H, W = 64, 64
    K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ext = np.stack([
        extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.15, 0.0, 0.0])),
        extrinsic_from_center_and_rotation(np.eye(3), np.array([0.15, 0.0, 0.0])),
    ])
    intr = np.tile(K[None, :, :], (2, 1, 1))
    dmaps = np.full((2, H, W, 1), 2.0, dtype=np.float32)
    wmaps = np.full((2, H, W), 2.0, dtype=np.float32)
    cols = np.full((len(pts), 3), 200, dtype=np.uint8)
    return build_canonical_scene_geometry(
        points=pts,
        colors=cols,
        cam_centers=[np.array([0.0, 0.0, 0.0])],
        extrinsics=ext,
        intrinsics=intr,
        synchronized_depth_maps=dmaps,
        validated_weight_maps=wmaps,
        voxel_size=voxel_size,
        free_space_evidence=free_space,
    )


class TestV22PoissonAndTSDFRepair(unittest.TestCase):

    def test_01_ordered_ccw_boundary_loop_extraction(self):
        mesh = _make_grid_mesh_with_hole(nx=19, ny=19, hole_box=(-0.15, 0.15, -0.15, 0.15))
        holes = analyze_tsdf_boundary_holes(mesh, voxel_size=0.02)
        interior = [h for h in holes if h["diameter_m"] < 0.7]
        self.assertGreaterEqual(len(interior), 1)
        h = interior[0]
        self.assertGreater(h["area_m2"], 0.05)
        self.assertIn("loop_plane_mad_m", h)
        self.assertLess(h["loop_plane_mad_m"], 0.002)
        # Verify CCW ordering in (u_axis, v_axis)
        verts = np.asarray(mesh.vertices)
        pts_l = verts[h["vertex_indices"]]
        px = (pts_l - h["centroid"]) @ h["u_axis"]
        py = (pts_l - h["centroid"]) @ h["v_axis"]
        signed_a = 0.5 * float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))
        self.assertGreater(signed_a, 0.0)

    def test_02_branching_degree4_boundary_untangling(self):
        # Two square quad rings touching at a single shared pinched corner vertex -> degree-4 junction
        v = np.array([
            [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0],  # 0..3 left hole outer
            [0.3, 0.3, 2.0], [0.7, 0.3, 2.0], [0.7, 0.7, 2.0], [0.3, 0.7, 2.0],  # 4..7 hole 1
            [-1.0, -1.0, 2.0], [0.0, -1.0, 2.0], [-1.0, 0.0, 2.0],               # outer lower left
        ], dtype=np.float64)
        # Build a clean figure-8 boundary graph by constructing two separate open 4-loops meeting at vertex 0
        # 4 outer corners surround 2 diamonds touching at center vertex 0
        v8 = np.array([
            [0.0, 0.0, 2.0],   # 0: shared pinch vertex!
            [-0.2, 0.1, 2.0],  # 1
            [-0.4, 0.0, 2.0],  # 2
            [-0.2, -0.1, 2.0], # 3
            [0.2, 0.1, 2.0],   # 4
            [0.4, 0.0, 2.0],   # 5
            [0.2, -0.1, 2.0],  # 6
            [-0.6, 0.4, 2.0], [0.0, 0.4, 2.0], [0.6, 0.4, 2.0],
            [0.6, -0.4, 2.0], [0.0, -0.4, 2.0], [-0.6, -0.4, 2.0],
        ], dtype=np.float64)
        t8 = np.array([
            [7, 8, 1], [8, 0, 1], [7, 1, 2], [7, 2, 12], [2, 3, 12], [3, 0, 11], [3, 11, 12],
            [8, 9, 4], [8, 4, 0], [4, 9, 5], [9, 10, 5], [5, 10, 6], [6, 10, 11], [6, 11, 0],
        ], dtype=np.int32)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(v8)
        m.triangles = o3d.utility.Vector3iVector(t8)
        m.compute_vertex_normals()
        holes = analyze_tsdf_boundary_holes(m, voxel_size=0.05)
        small_loops = [h for h in holes if h["num_vertices"] <= 6]
        self.assertGreaterEqual(len(small_loops), 2)

    def test_03_category_a_tiny_pinhole_boundary_pinned_no_poisson(self):
        mesh = _make_grid_mesh_with_hole(nx=31, ny=31, span_x=0.6, span_y=0.6, hole_box=(-0.015, 0.015, -0.015, 0.015))
        xw, yw = np.meshgrid(np.linspace(-0.6, 0.6, 40), np.linspace(-0.6, 0.6, 40))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.02)
        cfg = MeshBranchQAConfig(voxel_size=0.02, tsdf_voxel_length=0.02, small_hole_max_voxel_diam=5.0, case_b_min_area_m2=0.008)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), cfg)
            self.assertFalse(res["localized_poisson_executed"])
            self.assertFalse(res["poisson_triggered"])
            self.assertGreaterEqual(res["small_holes_repaired"], 1)

    def test_04_dead_zone_eliminated_medium_planar_gap(self):
        # Medium ~9cm x 9cm gap (area ~0.0064m2) that fell in old Case A (<0.008m2) vs B dead zone!
        mesh = _make_grid_mesh_with_hole(nx=29, ny=29, span_x=0.7, span_y=0.7, hole_box=(-0.045, 0.045, -0.045, 0.045))
        xw, yw = np.meshgrid(np.linspace(-0.7, 0.7, 50), np.linspace(-0.7, 0.7, 50))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.012)
        # Keep config with legacy high case_b_min_area_m2=0.010 to verify dead zone is STILL closed by B1!
        cfg = MeshBranchQAConfig(
            voxel_size=0.012, tsdf_voxel_length=0.012, small_hole_max_voxel_diam=5.0,
            case_a_max_loop_verts=12, case_b_min_area_m2=0.010,
        )
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), cfg)
            cats = [g["category"] for g in res["gap_classifications"]]
            self.assertIn("CATEGORY_B1_PLANAR_CONSTRAINED", cats)
            self.assertGreaterEqual(res["b1_patches_accepted"], 1)
            self.assertFalse(res["localized_poisson_executed"])

    def test_05_b1_constrained_planar_reduces_open_edges_shares_rim(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.15, 0.15, -0.15, 0.15))
        xw, yw = np.meshgrid(np.linspace(-0.8, 0.8, 55), np.linspace(-0.8, 0.8, 55))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)
        cfg = MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015)
        e0 = np.sort(np.asarray(mesh.triangles)[:, [0, 1]], axis=1)
        e1 = np.sort(np.asarray(mesh.triangles)[:, [1, 2]], axis=1)
        e2 = np.sort(np.asarray(mesh.triangles)[:, [2, 0]], axis=1)
        _, c_before = np.unique(np.vstack([e0, e1, e2]), axis=0, return_counts=True)
        open_before = int(np.count_nonzero(c_before == 1))

        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), cfg)
            self.assertGreaterEqual(res["b1_patches_accepted"], 1)
            rep = res["repaired_mesh"]
            self.assertIsNotNone(rep)
            rt = np.asarray(rep.triangles)
            _, c_after = np.unique(np.vstack([np.sort(rt[:, [0, 1]], 1), np.sort(rt[:, [1, 2]], 1), np.sort(rt[:, [2, 0]], 1)]), axis=0, return_counts=True)
            open_after = int(np.count_nonzero(c_after == 1))
            self.assertLess(open_after, open_before)
            self.assertEqual(int(np.count_nonzero(c_after > 2)), 0)

    def test_06_concave_l_shaped_b1_preserves_boundary_edges(self):
        # Strictly concave L-shaped hole cutout inside grid
        nx, ny = 21, 21
        xs = np.linspace(-0.8, 0.8, nx)
        ys = np.linspace(-0.8, 0.8, ny)
        xx, yy = np.meshgrid(xs, ys)
        verts = np.column_stack([xx.ravel(), yy.ravel(), np.full_like(xx, 2.0).ravel()])
        tris = []
        for r in range(ny - 1):
            for c in range(nx - 1):
                i0 = r * nx + c
                for tri in ([i0, i0 + 1, i0 + nx + 1], [i0, i0 + nx + 1, i0 + nx]):
                    cx, cy = verts[tri].mean(axis=0)[:2]
                    in_l = ((-0.20 <= cx <= 0.20) and (-0.20 <= cy <= 0.00)) or ((-0.20 <= cx <= 0.00) and (0.00 <= cy <= 0.25))
                    if not in_l:
                        tris.append(tri)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        holes = analyze_tsdf_boundary_holes(mesh, voxel_size=0.02)
        inner = min(holes, key=lambda r: r["area_m2"])
        p_pts, p_tris, gids = build_boundary_pinned_planar_patch(
            inner, np.asarray(mesh.vertices), verts, np.zeros(0, dtype=np.int64),
            voxel_size=0.02, max_height_delta_m=0.02, allow_interior_subdivision=True,
        )
        self.assertGreater(len(p_tris), len(gids) - 3)
        # Check every boundary edge (i, (i+1)%L) exists inside p_tris!
        pe = np.vstack([np.sort(p_tris[:, [0, 1]], 1), np.sort(p_tris[:, [1, 2]], 1), np.sort(p_tris[:, [2, 0]], 1)])
        ue, ec = np.unique(pe, axis=0, return_counts=True)
        b_edges = {(int(a), int(b)) for (a, b), cnt in zip(ue, ec) if cnt == 1}
        L = int(np.count_nonzero(gids >= 0))
        expected = {(min(i, (i + 1) % L), max(i, (i + 1) % L)) for i in range(L)}
        self.assertEqual(b_edges, expected)

    def test_07_b1_harmonic_lift_blends_interior_pins_rim(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.15, 0.15, -0.15, 0.15))
        holes = analyze_tsdf_boundary_holes(mesh, voxel_size=0.02)
        inner = min(holes, key=lambda r: r["area_m2"])
        # Interior support slightly displaced +8mm along z inside hole
        gx, gy = np.meshgrid(np.linspace(-0.10, 0.10, 6), np.linspace(-0.10, 0.10, 6))
        sup_pts = np.column_stack([gx.ravel(), gy.ravel(), np.full_like(gx, 2.008).ravel()])
        p_pts, _, gids = build_boundary_pinned_planar_patch(
            inner, np.asarray(mesh.vertices), sup_pts, np.arange(len(sup_pts)),
            voxel_size=0.02, max_height_delta_m=0.015, allow_interior_subdivision=True,
        )
        rim_gids = gids[gids >= 0]
        L = len(rim_gids)
        mverts = np.asarray(mesh.vertices)
        self.assertAlmostEqual(float(np.max(np.linalg.norm(p_pts[:L] - mverts[rim_gids], axis=1))), 0.0, places=8)
        if len(p_pts) > L:
            self.assertLess(abs(float(np.mean(p_pts[L:, 2])) - 2.008), 0.008)

    def test_08_and_09_curved_b2_triggers_clipped_stitched_poisson(self):
        # Strongly curved 3D surface z = 2.0 + 0.14*sin(5*x) + 0.09*cos(4.5*y) -> triggers Case B2 Poisson!
        z_curve = lambda x, y: 2.0 + 0.14 * np.sin(5.0 * x) + 0.09 * np.cos(4.5 * y)
        mesh = _make_grid_mesh_with_hole(nx=27, ny=27, span_x=0.8, span_y=0.8, hole_box=(-0.18, 0.18, -0.18, 0.18), z_fn=z_curve)
        xw, yw = np.meshgrid(np.linspace(-0.8, 0.8, 70), np.linspace(-0.8, 0.8, 70))
        zw = z_curve(xw, yw)
        pts = np.column_stack([xw.ravel(), yw.ravel(), zw.ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)
        cfg = MeshBranchQAConfig(
            voxel_size=0.015, tsdf_voxel_length=0.015, poisson_depth=6,
            case_b_min_support_pts=40, b1_planar_mad_max_m=0.004,
            case_b_max_normal_std_deg=42.0,
        )
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            res = run_conditional_localized_repair(mesh, geom, td_p, cfg)
            self.assertTrue(res["localized_poisson_executed"], f"Expected Case B2 Poisson, got {res['note']}")
            self.assertTrue(res["poisson_triggered"])
            self.assertGreaterEqual(res["b2_poisson_patches_accepted"], 1)
            for p_rec in res["case_b_patches"]:
                self.assertEqual(p_rec["per_patch_qa"]["patch_components"], 1)
                self.assertLessEqual(p_rec["per_patch_qa"]["open_edges_delta"], 0)
                self.assertGreater(p_rec["per_patch_qa"]["shared_rim_vertices"], 4)

    def test_10_adjacent_perpendicular_wall_ignored_zero_double_shell(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.14, 0.14, -0.14, 0.14))
        xw, yw = np.meshgrid(np.linspace(-0.8, 0.8, 50), np.linspace(-0.8, 0.8, 50))
        pts_front = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()])
        # Perpendicular side wall nearby (x=0.20, z in [1.7..2.3]) -> must NOT contaminate support normals/patch!
        yw2, zw2 = np.meshgrid(np.linspace(-0.5, 0.5, 30), np.linspace(1.7, 2.3, 30))
        pts_perp = np.column_stack([np.full_like(yw2, 0.20).ravel(), yw2.ravel(), zw2.ravel()])
        pts = np.vstack([pts_front, pts_perp]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)
        cfg = MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), cfg)
            self.assertGreaterEqual(res["b1_patches_accepted"], 1)
            for p in res["case_b_patches"]:
                self.assertLessEqual(p["per_patch_qa"]["local_dbl_delta"], 0.01)

    def test_11_case_c_doorway_freespace_never_sealed(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.15, 0.15, -0.15, 0.15))
        xw, yw = np.meshgrid(np.linspace(-0.8, 0.8, 50), np.linspace(-0.8, 0.8, 50))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        H, W = 64, 64
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([extrinsic_from_center_and_rotation(np.eye(3), np.zeros(3))])
        intr = K[None, :, :]
        dmap = np.full((1, H, W, 1), 2.0, dtype=np.float32)
        dmap[0, 22:42, 22:42, 0] = 4.5  # open hallway straight through hole
        cmap = np.full((1, H, W), 3.0, dtype=np.float32)
        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
        fse.build_from_depth_maps(dmap, cmap, ext, intr, max_depth=5.0, pixel_stride=1)
        geom = _make_dummy_scene_geom(pts, free_space=fse, voxel_size=0.015)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015))
            self.assertEqual(res["qualifying_gaps_count"], 0)
            self.assertGreaterEqual(len(res["case_c_preserved_gaps"]), 1)

    def test_12_case_c_open_perimeter_outer_rim_preserved(self):
        # Single outer perimeter loop has no surrounding mesh across all 8 azimuthal sectors -> enclosed_sectors < 6!
        mesh = _make_grid_mesh_with_hole(nx=15, ny=15, span_x=0.35, span_y=0.35, hole_box=None)
        xw, yw = np.meshgrid(np.linspace(-0.35, 0.35, 30), np.linspace(-0.35, 0.35, 30))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015, large_hole_max_diam_m=2.0))
            self.assertEqual(res["qualifying_gaps_count"], 0)
            cats = [g["category"] for g in res["gap_classifications"]]
            self.assertIn("CATEGORY_C_OPEN_BOUNDARY_RIM", cats)

    def test_13_case_c_oversized_opening_preserved(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.48, 0.48, -0.48, 0.48))
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.02)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), MeshBranchQAConfig(voxel_size=0.02, tsdf_voxel_length=0.02, large_hole_max_diam_m=0.80))
            cats = [g["category"] for g in res["gap_classifications"]]
            self.assertIn("CATEGORY_C_OVERSIZED_OPENING", cats)
            self.assertEqual(res["qualifying_gaps_count"], 0)

    def test_14_case_c_unsupported_void_preserved(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.15, 0.15, -0.15, 0.15))
        # Empty canonical point cloud near the hole -> zero observation support!
        pts = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [5.0, 5.1, 5.0]], dtype=np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)
        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015))
            cats = [g["category"] for g in res["gap_classifications"]]
            self.assertIn("CATEGORY_C_INSUFFICIENT_SUPPORT", cats)
            self.assertEqual(res["qualifying_gaps_count"], 0)

    def test_15_per_patch_qa_rejects_hovering_double_shell(self):
        # Solid wall (NO hole cutout at z=2.0), plus adversarial patch floating 4cm in front at z=1.96!
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=None)
        mv = np.asarray(mesh.vertices)
        mt = np.asarray(mesh.triangles)
        fake_hole = {
            "hole_id": 99,
            "vertex_indices": [0, 1, 2, 3],
            "centroid": np.array([0.0, 0.0, 2.0]),
            "normal": np.array([0.0, 0.0, -1.0]),
            "u_axis": np.array([1.0, 0.0, 0.0]),
            "v_axis": np.array([0.0, -1.0, 0.0]),
            "diameter_m": 0.30,
        }
        # Floating 4-triangle quad hovering 4cm in front of existing solid wall
        p_pts = np.array([
            [-0.12, -0.12, 1.96], [0.12, -0.12, 1.96], [0.12, 0.12, 1.96], [-0.12, 0.12, 1.96], [0.0, 0.0, 1.96]
        ], dtype=np.float64)
        p_tris = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int32)
        stitched_gids = np.full(len(p_pts), -1, dtype=np.int64)
        pts = mv.astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)

        ok, reason, _, _ = verify_patch_local_qa(
            fake_hole, mv, mt, p_pts, p_tris, stitched_gids,
            None, geom.synchronized_depth_maps, geom.extrinsics, geom.intrinsics,
            mv, np.full((len(mv), 3), 200.0), cKDTree(mv),
            voxel_size=0.015, min_depth=0.2, max_depth=5.5,
            max_freespace_ratio=0.16, max_normal_deg=30.0, max_dbl_shell_delta=0.008,
        )
        self.assertFalse(ok)
        self.assertIn("DOUBLE_SHELL", reason)

    def test_16_per_patch_qa_rejects_confetti_non_manifold_and_open_edge_bloat(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.14, 0.14, -0.14, 0.14))
        holes = analyze_tsdf_boundary_holes(mesh, voxel_size=0.015)
        inner = min(holes, key=lambda r: r["area_m2"])
        mv = np.asarray(mesh.vertices)
        mt = np.asarray(mesh.triangles)
        pts = mv.astype(np.float32)
        geom = _make_dummy_scene_geom(pts, voxel_size=0.015)

        # Case A: Two disconnected triangles (confetti!) -> rejected!
        p_conf = np.array([
            [-0.08, -0.08, 2.0], [-0.04, -0.08, 2.0], [-0.06, -0.04, 2.0],
            [0.04, 0.04, 2.0], [0.08, 0.04, 2.0], [0.06, 0.08, 2.0],
        ], dtype=np.float64)
        t_conf = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        ok1, r1, _, _ = verify_patch_local_qa(
            inner, mv, mt, p_conf, t_conf, np.full(6, -1, dtype=np.int64),
            None, geom.synchronized_depth_maps, geom.extrinsics, geom.intrinsics,
            mv, np.full((len(mv), 3), 200.0), cKDTree(mv),
            0.015, 0.2, 5.5, 0.16, 30.0, 0.008,
        )
        self.assertFalse(ok1)
        self.assertIn("FRAGMENTED", r1)

        # Case B: Single floating unconnected triangle increasing open boundary edges by +3 -> rejected!
        ok2, r2, _, _ = verify_patch_local_qa(
            inner, mv, mt, p_conf[:3], t_conf[:1], np.full(3, -1, dtype=np.int64),
            None, geom.synchronized_depth_maps, geom.extrinsics, geom.intrinsics,
            mv, np.full((len(mv), 3), 200.0), cKDTree(mv),
            0.015, 0.2, 5.5, 0.16, 30.0, 0.008,
        )
        self.assertFalse(ok2)
        self.assertIn("INCREASED_OPEN_EDGES", r2)

    def test_17_per_patch_independent_accept_reject_isolated(self):
        # Wall with TWO holes: left hole valid interior gap; right hole pierced by KNOWN_FREE_SPACE doorway rays!
        nx, ny = 31, 21
        xs = np.linspace(-1.0, 1.0, nx)
        ys = np.linspace(-0.7, 0.7, ny)
        xx, yy = np.meshgrid(xs, ys)
        verts = np.column_stack([xx.ravel(), yy.ravel(), np.full_like(xx, 2.0).ravel()])
        tris = []
        for r in range(ny - 1):
            for c in range(nx - 1):
                i0 = r * nx + c
                for tri in ([i0, i0 + 1, i0 + nx + 1], [i0, i0 + nx + 1, i0 + nx]):
                    cx, cy = verts[tri].mean(axis=0)[:2]
                    in_left = (-0.52 <= cx <= -0.22) and (-0.18 <= cy <= 0.18)
                    in_right = (0.22 <= cx <= 0.52) and (-0.18 <= cy <= 0.18)
                    if not (in_left or in_right):
                        tris.append(tri)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        H, W = 64, 96
        K = np.array([[60.0, 0.0, 48.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = np.stack([extrinsic_from_center_and_rotation(np.eye(3), np.zeros(3))])
        intr = K[None, :, :]
        dmap = np.full((1, H, W, 1), 2.0, dtype=np.float32)
        # Rays through right hole (+0.22..+0.52) travel unobstructed to 4.5m!
        us = np.arange(W)[None, :]
        vs = np.arange(H)[:, None]
        wx = (us - 48.0) / 60.0 * 2.0
        wy = (vs - 32.0) / 60.0 * 2.0
        door_mask = (wx >= 0.22) & (wx <= 0.52) & (np.abs(wy) <= 0.20)
        dmap[0, door_mask, 0] = 4.5
        cmap = np.full((1, H, W), 3.0, dtype=np.float32)
        fse = FreeSpaceVoxelEvidence(voxel_size_m=0.04)
        fse.build_from_depth_maps(dmap, cmap, ext, intr, max_depth=5.0, pixel_stride=1)

        xw, yw = np.meshgrid(np.linspace(-1.0, 1.0, 60), np.linspace(-0.7, 0.7, 45))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = _make_dummy_scene_geom(pts, free_space=fse, voxel_size=0.015)

        with tempfile.TemporaryDirectory() as td:
            res = run_conditional_localized_repair(mesh, geom, Path(td), MeshBranchQAConfig(voxel_size=0.015, tsdf_voxel_length=0.015))
            self.assertEqual(res["b1_patches_accepted"], 1)
            self.assertGreaterEqual(len(res["case_c_preserved_gaps"]), 1)

    def test_18_manifold_patch_winding_flips_inverted_faces(self):
        existing = np.array([[0, 1, 2]], dtype=np.int32)
        # Patch sharing directed edge (1, 0) in opposite orientation vs WRONG same orientation (0, 1)
        wrong_patch = np.array([[0, 1, 3]], dtype=np.int32)
        aligned = align_patch_winding_to_mesh(wrong_patch, existing, {0, 1, 2})
        self.assertEqual(aligned[0].tolist(), [0, 3, 1])

    def test_19_global_qa_rejects_corrupted_repaired_and_keeps_observed_intact(self):
        from src.reconstruction.mesh_branches_and_qa import evaluate_mesh_candidate_qa
        H, W = 64, 64
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.1, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.1, 0.0, 0.0])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))
        dmaps = np.full((2, H, W, 1), 2.0, dtype=np.float32)
        wmaps = np.full((2, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 180, dtype=np.uint8) for _ in range(2)]
        xw, yw = np.meshgrid(np.linspace(-1.25, 1.25, 70), np.linspace(-1.25, 1.25, 70))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = build_canonical_scene_geometry(
            points=pts, colors=np.full((len(pts), 3), 180, dtype=np.uint8),
            cam_centers=[np.zeros(3)], extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=wmaps, voxel_size=0.02,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.08,
            qa_min_triangles=100, qa_min_obs_support_ratio=0.70, qa_min_completeness_ratio=0.50,
            enable_conditional_repair=True, qa_max_double_shell_ratio=0.12,
        )
        with tempfile.TemporaryDirectory() as td:
            res = run_global_tsdf_and_conditional_repair(geom, frames, Path(td), cfg=cfg)
            self.assertEqual(res["selection_status"], "PASS")
            obs_ply = Path(td) / "mesh_global_tsdf_observed.ply"
            self.assertTrue(obs_ply.is_file())
            obs_bytes = obs_ply.read_bytes()
            # Simulate a corrupted duplicate double-shell mesh at z=1.94 & verify global QA flags excessive double shells!
            corrupted = o3d.geometry.TriangleMesh(res["mesh_observed"])
            shift = o3d.geometry.TriangleMesh(res["mesh_observed"])
            shift.translate([0.0, 0.0, -0.06])
            corrupted += shift
            qa_bad = evaluate_mesh_candidate_qa("corrupted_repaired", "corrupted", corrupted, geom, cfg)
            self.assertFalse(qa_bad.passes_qa)
            self.assertEqual(obs_ply.read_bytes(), obs_bytes)

    def test_20_stage7_splat_sync_boosts_coverage_without_silhouette_bleed(self):
        H, W = 60, 80
        K = np.array([[80.0, 0.0, 40.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        ext = extrinsic_from_center_and_rotation(np.eye(3), np.zeros(3))[None, :, :]
        intr = K[None, :, :]
        # Wall at z=2.0 on left half (u < 40), foreground box at z=1.5 on right (u >= 40) refined to 1.48!
        dmap = np.full((1, H, W, 1), 2.0, dtype=np.float32)
        dmap[0, :, 40:, 0] = 1.50
        dmap[0, 10:15, 10:15, 0] = 0.0  # Contradicted void must stay 0.0!
        wmap = np.full((1, H, W), 2.0, dtype=np.float32)

        xs, ys = np.meshgrid(np.linspace(0.0, 0.65, 30), np.linspace(-0.4, 0.4, 30))
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full_like(xs, 1.48).ravel()]).astype(np.float32)

        _, _, st_nosplat = project_canonical_points_to_depth_maps(pts, ext, intr, dmap, wmap, max_depth_delta_m=0.06, splat_radius_px=0)
        synced, _, st_splat = project_canonical_points_to_depth_maps(pts, ext, intr, dmap, wmap, max_depth_delta_m=0.06, splat_radius_px=1)
        self.assertGreater(st_splat["updated_from_canonical_points"], st_nosplat["updated_from_canonical_points"])
        # Edge guard: pixel u=38 (two px left on 2.0m wall) stays strictly 2.0, never bleeds to 1.48!
        self.assertAlmostEqual(float(synced[0, 30, 38, 0]), 2.0, places=4)
        # Contradicted void stays 0.0!
        self.assertEqual(float(synced[0, 12, 12, 0]), 0.0)

    def test_21_evidence_preserving_cluster_filter_keeps_supported_small_islands(self):
        # 1 genuine 6-tri edge-connected strip supported by points near (0,0,2.0) + 1 bogus 6-tri floater at (5,5,5)
        verts = []
        tris = []
        # Cluster A: 6-triangle contiguous strip sharing edges near (0..0.03, 0..0.02, 2.0)
        for col in range(4):
            verts.append([0.01 * col, 0.0, 2.0])
            verts.append([0.01 * col, 0.02, 2.0])
        for col in range(3):
            b = col * 2
            tris.append([b, b + 2, b + 3])
            tris.append([b, b + 3, b + 1])
        # Cluster B: 6-triangle contiguous floater far away at (5.0..5.03, 5.0..5.02, 5.0)
        off = len(verts)
        for col in range(4):
            verts.append([5.0 + 0.01 * col, 5.0, 5.0])
            verts.append([5.0 + 0.01 * col, 5.02, 5.0])
        for col in range(3):
            b = off + col * 2
            tris.append([b, b + 2, b + 3])
            tris.append([b, b + 3, b + 1])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        sup_pts = np.array([[0.02, 0.01, 2.0], [0.04, 0.01, 2.0]], dtype=np.float64)
        cleaned, stat = _clean_mesh_topology_with_stats(mesh, min_cluster_faces=25, support_points=sup_pts, support_radius_m=0.03)
        self.assertEqual(stat["preserved_small_clusters"], 1)
        self.assertEqual(len(cleaned.triangles), 6)
        self.assertLess(np.max(np.asarray(cleaned.vertices)[:, 2]), 2.5)

    def test_22_tsdf_branch_reports_canonical_point_retention_metrics(self):
        H, W = 64, 64
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.1, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.1, 0.0, 0.0])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))
        dmaps = np.full((2, H, W, 1), 1.8, dtype=np.float32)
        wmaps = np.full((2, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 180, dtype=np.uint8) for _ in range(2)]
        xw, yw = np.meshgrid(np.linspace(-0.7, 0.7, 45), np.linspace(-0.7, 0.7, 45))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 1.8).ravel()]).astype(np.float32)
        geom = build_canonical_scene_geometry(
            points=pts, colors=np.full((len(pts), 3), 180, dtype=np.uint8),
            cam_centers=[np.zeros(3)], extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=wmaps, voxel_size=0.02,
        )
        cfg = MeshBranchQAConfig(voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.08)
        _, stats = build_global_tsdf_branch(geom, frames, cfg)
        self.assertGreater(stats["canonical_points_within_3vox_pct"], 95.0)
        self.assertIn("preserved_small_clusters", stats)

    def test_23_texture_baker_separates_observed_and_local_repair_materials(self):
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            # Triangles 0..1 observed wall, triangles 2..3 accepted local repair patch!
            verts = np.array([
                [-0.6, -0.5, 2.0], [0.0, -0.5, 2.0], [0.0, 0.5, 2.0], [-0.6, 0.5, 2.0],
                [0.6, -0.5, 2.0], [0.6, 0.5, 2.0],
            ], dtype=np.float64)
            tris = np.array([
                [0, 1, 2], [0, 2, 3],   # 0, 1 observed TSDF
                [1, 4, 5], [1, 5, 2],   # 2, 3 repaired patch
            ], dtype=np.int32)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(tris)
            img_p = td_p / "cam0.png"
            Image.new("RGB", (120, 90), (190, 180, 170)).save(img_p)
            K = np.array([[80.0, 0.0, 60.0], [0.0, 80.0, 45.0], [0.0, 0.0, 1.0]])
            baker = VisibilityAwareTextureBaker(
                mesh, extrinsic_from_center_and_rotation(np.eye(3), np.zeros(3))[None, :, :],
                K[None, :, :], [img_p], depth_maps=np.full((1, 90, 120, 1), 2.0, dtype=np.float32),
            )
            best_cam = np.zeros(4, dtype=np.int32)
            atlas_p = td_p / "atlas.png"
            rects, _ = baker.synthesize_atlas(best_cam, atlas_p, atlas_size=512)
            deliv, tqa = baker.export_canonical_deliverables(
                td_p, "sep_room", best_cam, rects, atlas_p, atlas_size=512, repaired_face_start_idx=2,
            )
            self.assertTrue(tqa["glb_reload_valid"])
            self.assertEqual(tqa["out_of_range_uv_count"], 0)
            self.assertEqual(tqa["glb_reloaded_mesh_count"], 2)
            self.assertTrue(tqa["provenance_separation"]["separate_material_nodes"])
            self.assertEqual(tqa["provenance_separation"]["observed_faces_count"], 2)
            self.assertEqual(tqa["provenance_separation"]["accepted_local_repair_faces_count"], 2)
            mtl_txt = Path(deliv["mtl"]).read_text()
            self.assertIn("Material_Photographic_Atlas", mtl_txt)
            self.assertIn("Material_Local_Repair_Photographic", mtl_txt)

    def test_24_gap_diagnostics_and_before_after_renders_generated(self):
        mesh = _make_grid_mesh_with_hole(nx=21, ny=21, span_x=0.8, span_y=0.8, hole_box=(-0.14, 0.14, -0.14, 0.14))
        H, W = 64, 64
        K = np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([-0.15, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.15, 0.0, 0.0])),
        ])
        intr = np.tile(K[None, :, :], (2, 1, 1))
        dmaps = np.full((2, H, W, 1), 2.0, dtype=np.float32)
        wmaps = np.full((2, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 190, dtype=np.uint8) for _ in range(2)]
        xw, yw = np.meshgrid(np.linspace(-0.8, 0.8, 50), np.linspace(-0.8, 0.8, 50))
        pts = np.column_stack([xw.ravel(), yw.ravel(), np.full_like(xw, 2.0).ravel()]).astype(np.float32)
        geom = build_canonical_scene_geometry(
            points=pts, colors=np.full((len(pts), 3), 190, dtype=np.uint8),
            cam_centers=[np.zeros(3)], extrinsics=ext, intrinsics=intr,
            synchronized_depth_maps=dmaps, validated_weight_maps=wmaps, voxel_size=0.02,
        )
        cfg = MeshBranchQAConfig(
            voxel_size=0.02, tsdf_voxel_length=0.02, tsdf_trunc=0.08,
            qa_min_triangles=100, qa_min_obs_support_ratio=0.70, qa_min_completeness_ratio=0.50,
            enable_conditional_repair=True,
        )
        with tempfile.TemporaryDirectory() as td:
            td_p = Path(td)
            res = run_global_tsdf_and_conditional_repair(geom, frames, td_p / "meshes", cfg=cfg, renders_dir=td_p / "renders")
            gap_json = td_p / "meshes" / "gap_classifications.json"
            self.assertTrue(gap_json.is_file())
            self.assertTrue((td_p / "renders" / "gap_diagnostics.png").is_file())
            self.assertTrue((td_p / "renders" / "before_after_repair_comparison.png").is_file())
            gdata = json.loads(gap_json.read_text())
            self.assertIn("gaps", gdata)


if __name__ == "__main__":
    unittest.main()
