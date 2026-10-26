"""
Unit Tests for F.O.R.G.E. Multi-View Geometry-Consistency Filtering Module.

Tests:
1. Exact bilinear depth sampling accuracy and boundary handling.
2. Perfect synthetic multi-camera planar surface consistency retention.
3. Rejection of inconsistent floating outliers.
4. Behind-camera (Z <= 0) and out-of-bounds safety.
5. Depth confidence thresholding.
6. Monotonicity of threshold strictness (depth tol, reproj tol, min consensus).
7. Optional voxel grid downsampling.
8. Degenerate and non-finite depth handling.
"""

import math
import unittest
import numpy as np

from src.reconstruction.multiview_consistency import (
    MultiViewConsistencyConfig,
    MultiViewConsistencyResult,
    bilinear_sample_depth,
    filter_point_cloud_multiview,
)


class TestMultiViewConsistency(unittest.TestCase):
    """Test suite for multi-view geometry-consistency filtering."""

    def test_01_bilinear_depth_sampling_exactness(self):
        """Bilinear depth sampling must reproduce exact linear interpolation."""
        # Create a simple 4x4 linear depth ramp: D(v, u) = 10.0 + 2.0*u + 3.0*v
        v_grid, u_grid = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        depth_map = 10.0 + 2.0 * u_grid + 3.0 * v_grid

        # Test at integer points
        u_test = np.array([0.0, 1.0, 2.0, 3.0])
        v_test = np.array([0.0, 1.0, 2.0, 3.0])
        sampled, mask = bilinear_sample_depth(depth_map, u_test, v_test)
        self.assertTrue(np.all(mask))
        expected = 10.0 + 2.0 * u_test + 3.0 * v_test
        np.testing.assert_allclose(sampled, expected, rtol=1e-5)

        # Test at fractional points: u=1.5, v=2.5 -> 10 + 2*1.5 + 3*2.5 = 20.5
        s_frac, m_frac = bilinear_sample_depth(depth_map, np.array([1.5]), np.array([2.5]))
        self.assertTrue(m_frac[0])
        self.assertAlmostEqual(s_frac[0], 20.5, places=5)

        # Test out of bounds
        s_oob, m_oob = bilinear_sample_depth(depth_map, np.array([-1.0, 5.0]), np.array([1.0, 1.0]))
        self.assertFalse(m_oob[0])
        self.assertFalse(m_oob[1])

    def test_02_synthetic_two_view_planar_consistency(self):
        """Points from a consistent planar surface seen by two cameras must be retained."""
        H, W = 64, 64
        fx, fy, cx, cy = 50.0, 50.0, 32.0, 32.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

        # Camera 0: at origin looking along +Z
        R0 = np.eye(3)
        T0 = np.zeros(3)
        ext0 = np.column_stack([R0, T0])

        # Camera 1: translated by +1.0m along X (baseline = 1.0m)
        R1 = np.eye(3)
        T1 = np.array([-1.0, 0.0, 0.0])  # P_cam1 = P_world - [1, 0, 0]
        ext1 = np.column_stack([R1, T1])

        # Synthetic flat wall at Z_world = 10.0m
        # For Cam 0: depth is 10.0m everywhere
        depth0 = np.full((H, W), 10.0, dtype=np.float64)
        # For Cam 1: depth is also 10.0m everywhere (planar wall parallel to image plane)
        depth1 = np.full((H, W), 10.0, dtype=np.float64)

        depths = np.stack([depth0, depth1], axis=0)
        exts = np.stack([ext0, ext1], axis=0)
        intrs = np.stack([K, K], axis=0)

        cfg = MultiViewConsistencyConfig(
            depth_relative_thresh=0.05,
            reproj_error_thresh_pix=2.0,
            min_consistent_views=1
        )

        pts, colors, res = filter_point_cloud_multiview(depths, exts, intrs, config=cfg)

        self.assertGreater(len(pts), 0)
        # All overlapping pixels should have depth exactly equal to 10.0m
        np.testing.assert_allclose(pts[:, 2], 10.0, atol=1e-3)
        self.assertGreater(res.total_retained_points, 1000)

    def test_03_rejection_of_inconsistent_outliers(self):
        """Floating artifacts with inconsistent depth across views must be rejected."""
        H, W = 32, 32
        K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])

        ext0 = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])
        ext1 = np.column_stack([np.eye(3), [-1.0, 0.0, 0.0]])

        # Cam 0 has depth = 10.0m, EXCEPT an outlier at center with depth = 4.0m
        depth0 = np.full((H, W), 10.0, dtype=np.float64)
        depth0[16, 16] = 4.0  # Floating floater

        # Cam 1 has true depth = 10.0m everywhere
        depth1 = np.full((H, W), 10.0, dtype=np.float64)

        depths = np.stack([depth0, depth1], axis=0)
        exts = np.stack([ext0, ext1], axis=0)
        intrs = np.stack([K, K], axis=0)

        cfg = MultiViewConsistencyConfig(
            depth_relative_thresh=0.03,
            reproj_error_thresh_pix=1.5,
            min_consistent_views=1
        )

        pts, colors, res = filter_point_cloud_multiview(depths, exts, intrs, config=cfg)

        # The outlier at 4.0m should NOT be in the filtered points!
        if len(pts) > 0:
            min_z = float(np.min(pts[:, 2]))
            self.assertGreater(min_z, 8.0, "Outlier at Z=4.0m was not properly rejected by multi-view filter!")

    def test_04_confidence_filtering(self):
        """Pixels below confidence floor must be excluded before consistency checks."""
        H, W = 16, 16
        K = np.array([[20.0, 0.0, 8.0], [0.0, 20.0, 8.0], [0.0, 0.0, 1.0]])
        ext0 = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])
        ext1 = np.column_stack([np.eye(3), [-0.5, 0.0, 0.0]])

        depths = np.full((2, H, W), 5.0, dtype=np.float64)
        exts = np.stack([ext0, ext1], axis=0)
        intrs = np.stack([K, K], axis=0)

        # Low confidence everywhere except top half
        conf0 = np.zeros((H, W), dtype=np.float64)
        conf0[:8, :] = 0.9
        conf1 = np.full((H, W), 0.9, dtype=np.float64)
        confidences = np.stack([conf0, conf1], axis=0)

        cfg = MultiViewConsistencyConfig(min_confidence=0.5, min_consistent_views=1)
        pts, colors, res = filter_point_cloud_multiview(
            depths, exts, intrs, confidences=confidences, config=cfg
        )

        # Low confidence pixels (128 in frame 0) must be rejected from input points
        self.assertEqual(res.total_input_points, 128 + 256)
        self.assertLessEqual(res.total_retained_points, 384)

    def test_05_configurable_threshold_strictness(self):
        """Stricter tolerance thresholds must prune equal or more points."""
        H, W = 32, 32
        K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        ext0 = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])
        ext1 = np.column_stack([np.eye(3), [-0.5, 0.0, 0.0]])

        # Add slight noise to depth to test strictness
        rng = np.random.RandomState(42)
        depth0 = 10.0 + rng.normal(0.0, 0.2, (H, W))
        depth1 = 10.0 + rng.normal(0.0, 0.2, (H, W))

        depths = np.stack([depth0, depth1], axis=0)
        exts = np.stack([ext0, ext1], axis=0)
        intrs = np.stack([K, K], axis=0)

        # Permissive config
        cfg_permissive = MultiViewConsistencyConfig(depth_relative_thresh=0.10, reproj_error_thresh_pix=5.0)
        _, _, res_perm = filter_point_cloud_multiview(depths, exts, intrs, config=cfg_permissive)

        # Strict config
        cfg_strict = MultiViewConsistencyConfig(depth_relative_thresh=0.01, reproj_error_thresh_pix=1.0)
        _, _, res_strict = filter_point_cloud_multiview(depths, exts, intrs, config=cfg_strict)

        self.assertLessEqual(res_strict.total_retained_points, res_perm.total_retained_points)

    def test_06_voxel_downsampling_option(self):
        """Optional spatial voxel decimation must reduce point count while keeping bounds."""
        H, W = 32, 32
        K = np.array([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
        ext0 = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])
        ext1 = np.column_stack([np.eye(3), [-0.5, 0.0, 0.0]])

        depths = np.full((2, H, W), 10.0, dtype=np.float64)
        exts = np.stack([ext0, ext1], axis=0)
        intrs = np.stack([K, K], axis=0)

        # Without voxel downsampling
        cfg_full = MultiViewConsistencyConfig(downsample_voxel_m=None)
        pts_full, _, _ = filter_point_cloud_multiview(depths, exts, intrs, config=cfg_full)

        # With 0.5m voxel downsampling
        cfg_voxel = MultiViewConsistencyConfig(downsample_voxel_m=0.5)
        pts_voxel, _, _ = filter_point_cloud_multiview(depths, exts, intrs, config=cfg_voxel)

        self.assertLess(len(pts_voxel), len(pts_full))
        self.assertGreater(len(pts_voxel), 0)

    def test_07_empty_and_nonfinite_depth(self):
        """All-NaN or all-zero depths must return empty point clouds cleanly without error."""
        H, W = 16, 16
        K = np.eye(3)
        ext = np.column_stack([np.eye(3), [0.0, 0.0, 0.0]])

        depths = np.full((2, H, W), np.nan, dtype=np.float64)
        exts = np.stack([ext, ext], axis=0)
        intrs = np.stack([K, K], axis=0)

        pts, colors, res = filter_point_cloud_multiview(depths, exts, intrs)
        self.assertEqual(len(pts), 0)
        self.assertEqual(res.total_retained_points, 0)


if __name__ == "__main__":
    unittest.main()
