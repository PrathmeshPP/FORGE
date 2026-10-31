"""
Automated validation tests for indoor reconstruction geometry quality improvement.
Verifies rigid SE(3) scale lock, coordinate sanity, and proportion recovery.
"""

import json
from pathlib import Path
import numpy as np
import pytest

DIAG_PATH = Path("outputs/indoor/geometry_quality_improvement_video1/diagnostics/geometric_proportion_audit.json")
COLLISION_PATH = Path("outputs/indoor/geometry_quality_improvement_video1/collision_bounds.json")
POSES_PATH = Path("outputs/indoor/geometry_quality_improvement_video1/camera_geometry/camera_poses_rigid.npz")

@pytest.fixture
def audit_data():
    assert DIAG_PATH.exists(), f"Audit file not found at {DIAG_PATH}"
    with open(DIAG_PATH) as f:
        return json.load(f)

@pytest.fixture
def collision_data():
    assert COLLISION_PATH.exists(), f"Collision file not found at {COLLISION_PATH}"
    with open(COLLISION_PATH) as f:
        return json.load(f)

def test_room_height_recovery(audit_data):
    """Room height in Candidate D must be physically realistic (>= 2.2m)."""
    cand_d = audit_data["candidate_d"]
    room_height = cand_d["room_height_m"]
    assert room_height >= 2.2, f"Room height {room_height:.2f}m is unphysically low (expected >= 2.2m)"
    assert room_height <= 3.2, f"Room height {room_height:.2f}m is unphysically high (expected <= 3.2m)"

def test_furniture_scale_proportions(audit_data):
    """Furniture height must occupy natural percentage of room height (< 32% for bed, < 40% for desk)."""
    cand_d = audit_data["candidate_d"]
    bed_ratio = cand_d["bed_ratio_pct"]
    desk_ratio = cand_d["desk_ratio_pct"]
    assert bed_ratio < 32.0, f"Bed ratio {bed_ratio:.1f}% exceeds 32% ceiling (indicates scale inflation)"
    assert desk_ratio < 40.0, f"Desk ratio {desk_ratio:.1f}% exceeds 40% ceiling (indicates scale inflation)"

def test_missing_furniture_recovery(audit_data):
    """Chair region must contain valid reconstructed geometry vertices (> 10,000 points)."""
    cand_d = audit_data["candidate_d"]
    chair_points = cand_d["chair_points_count"]
    assert chair_points > 10000, f"Chair points {chair_points} insufficient (thin structure recovery failed)"

def test_rigid_alignment_poses():
    """Verify rigid alignment poses are valid 3x4 [R|t] camera matrices with det(R) == 1."""
    assert POSES_PATH.exists(), f"Poses not found at {POSES_PATH}"
    poses = np.load(POSES_PATH)
    exts = poses["extrinsics"]
    assert len(exts) == 63, f"Expected 63 camera poses, got {len(exts)}"
    for idx, E in enumerate(exts):
        R = E[:3, :3]
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0, atol=1e-3), f"Pose {idx} rotation matrix invalid, det={det}"
        assert E.shape == (3, 4), f"Pose {idx} shape must be (3, 4)"

def test_collision_bounds_structure(collision_data):
    """Collision geometry must include floor, ceiling, perimeter walls, and main furniture."""
    colliders = collision_data["colliders"]
    names = [c["name"] for c in colliders]
    required = ["Floor_Plane", "Ceiling_Plane", "Left_Wall", "Right_Wall", "Bed_Solid_Volume", "Desk_Solid_Volume"]
    for req in required:
        assert req in names, f"Missing required collider: {req}"
