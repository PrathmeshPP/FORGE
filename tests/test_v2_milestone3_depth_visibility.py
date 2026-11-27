"""
Automated Tests — F.O.R.G.E. V2 Milestone 3 (Multi-View Depth, Occlusion & Free-Space Validation).
Covers Part M Test #13: multi-view occlusion classification, floating ghost contradiction pruning,
non-removal of genuinely occluded surfaces, and KNOWN_FREE_SPACE voxel query.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reconstruction.canonical_camera import extrinsic_from_center_and_rotation
from src.reconstruction.depth_consistency_filter import (
    STATE_CONTRADICTED,
    STATE_OCCLUDED,
    STATE_SUPPORTED,
    DepthConsistencyConfig,
    MultiViewDepthConsistencyFilter,
)


class TestMultiviewDepthAndVisibility(unittest.TestCase):

    def test_13_occlusion_contradiction_and_freespace(self):
        """
        Scene with:
        - Cam 0 sees back wall at z=3.0m AND a floating ghost patch in middle at z=1.2m AND a side wall.
        - Cam 1 & 2 move laterally:
            * They confirm the back wall at z=3.0m (SUPPORTED).
            * They see a foreground pillar at z=1.5m blocking the left half of Cam 0's back wall -> OCCLUDED (preserved!).
            * They look STRAIGHT THROUGH Cam 0's floating ghost at z=1.2m into the z=3.0m wall -> CONTRADICTED (pruned!).
        """
        H, W = 60, 80
        K = np.array([[80.0, 0.0, 40.0], [0.0, 80.0, 30.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        intr = np.tile(K[None, :, :], (3, 1, 1))
        ext = np.stack([
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.0, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.35, 0.0, 0.0])),
            extrinsic_from_center_and_rotation(np.eye(3), np.array([0.55, 0.0, 0.0])),
        ])

        depths = np.full((3, H, W, 1), 3.0, dtype=np.float32)
        confs = np.full((3, H, W), 3.0, dtype=np.float32)
        frames = [np.full((H, W, 3), 180, dtype=np.uint8) for _ in range(3)]

        # In Cam 0 ONLY: inject a floating ghost right near image center (x in 38..46, y in 26..34) at d=1.2m!
        depths[0, 26:34, 38:46, 0] = 1.2
        # Notice Cam 1 & 2 see z=3.0m straight through where that z=1.2m ghost sits!
        # Let's check projection of (u=42, v=30, d=1.2) from Cam 0 into Cam 1 (x=0.35):
        # world x ≈ 0.03, cam1 x = 0.03 - 0.35 = -0.32 -> u_cam1 = 80*(-0.32/1.2)+40 = 18.7 (sees d=3.0m!)

        # Now in Cam 1 & Cam 2: place a real foreground occluder at z=1.4m covering u in [0..35] on Cam 1/2
        # Let's see where (u=28, v=30, d=3.0) from Cam 0 projects into Cam 1:
        # world x = (28-40)/80 * 3.0 = -0.45 -> cam1 x = -0.80 -> u_cam1 = 80*(-0.80/3.0)+40 = 18.7!
        # Wait: let's put the ghost strictly on the RIGHT half of Cam 0 (u=60..66) ->
        # world x = (63-40)/80*1.2 = 0.345 -> cam1 x = -0.005 -> u_cam1 = 39.7 (distinct from u=18.7!)
        depths[0, 26:34, 38:46, 0] = 3.0  # reset center
        depths[0, 26:34, 60:66, 0] = 1.2  # ghost on right side
        # Place foreground occluder ONLY on left side in Cam 1 & Cam 2 (u=8..28 in Cam 1, u=0..22 in Cam 2) at z=1.4m
        depths[1, 15:45, 5:26, 0] = 1.4
        depths[2, 15:45, 0:22, 0] = 1.4

        filt = MultiViewDepthConsistencyFilter(DepthConsistencyConfig(voxel_size_m=0.02, free_space_voxel_m=0.05))
        res = filt.validate_and_fuse(frames, depths, confs, ext, intr)

        st0 = res["pixel_state_maps"][0]
        # Ghost region (y=28..32, x=61..65) in Cam 0 MUST be CONTRADICTED and zeroed in validated_depth_maps!
        self.assertEqual(int(st0[30, 63]), STATE_CONTRADICTED)
        self.assertTrue(res["contradiction_masks"][0, 30, 63])
        self.assertEqual(float(res["validated_depth_maps"][0, 30, 63, 0]), 0.0)

        # Occluded back wall in Cam 0 (y=30, x=28) MUST be OCCLUDED (NOT contradicted!) and preserved in validated_depth_maps!
        self.assertEqual(int(st0[30, 28]), STATE_OCCLUDED)
        self.assertTrue(res["occlusion_masks"][0, 30, 28])
        self.assertGreater(float(res["validated_depth_maps"][0, 30, 28, 0]), 2.5)

        # Supported back wall in Cam 0 (y=30, x=50) MUST be SUPPORTED!
        self.assertEqual(int(st0[30, 50]), STATE_SUPPORTED)

        # Free space evidence check: point in open air at z=0.8m (in front of ghost/wall) MUST be KNOWN_FREE_SPACE;
        # point on the z=3.0m wall MUST NOT be KNOWN_FREE_SPACE!
        fse = res["free_space_evidence"]
        self.assertTrue(fse.query_free_space_mask(np.array([[0.0, 0.0, 0.8]]))[0])
        self.assertFalse(fse.query_free_space_mask(np.array([[0.3, 0.0, 3.0]]))[0])


if __name__ == "__main__":
    unittest.main()
