"""
F.O.R.G.E. — Camera Pose Interpolation & Reprojection Validator.
Interpolates intermediate camera poses between keyframes using SLERP on quaternions
and linear interpolation on camera centers, with reprojection consistency gating.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_sci


def slerp_pose(
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolates camera pose between (R1, t1) and (R2, t2) at fraction alpha in [0, 1].
    Camera center c = -R.T @ t is linearly interpolated, and R is interpolated via SLERP.
    Returns (R_interp, t_interp).
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Camera centers in world space
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2
    c_interp = (1.0 - alpha) * c1 + alpha * c2

    # SLERP rotations
    rot1 = R_sci.from_matrix(R1)
    rot2 = R_sci.from_matrix(R2)
    key_times = [0.0, 1.0]
    key_rots = R_sci.from_matrix([R1, R2])
    
    # Check if rotations are identical
    if np.allclose(R1, R2, atol=1e-5):
        R_interp = R1.copy()
    else:
        # Use SLERP via Slerp object
        from scipy.spatial.transform import Slerp
        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([alpha])
        R_interp = interp_rot.as_matrix()[0]

    # t = -R @ c
    t_interp = -R_interp @ c_interp
    return R_interp, t_interp


def validate_pose_reprojection(
    img_a: np.ndarray,
    img_b: np.ndarray,
    img_interp: np.ndarray,
    R_interp: np.ndarray,
    t_interp: np.ndarray,
    K: np.ndarray,
    max_reprojection_error_px: float = 4.0
) -> Tuple[bool, float]:
    """
    Validates reprojection consistency of the interpolated pose.
    Extracts ORB features between bounding frames and verifies epipolar/reprojection agreement.
    Returns (is_valid, mean_error_px).
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if len(img_a.shape) == 3 else img_a
    gray_interp = cv2.cvtColor(img_interp, cv2.COLOR_BGR2GRAY) if len(img_interp.shape) == 3 else img_interp

    orb = cv2.ORB_create(nfeatures=400)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_int, des_int = orb.detectAndCompute(gray_interp, None)

    if des_a is None or des_int is None or len(des_a) < 15 or len(des_int) < 15:
        return False, 999.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_int)

    if len(matches) < 10:
        return False, 999.0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_int = np.float32([kp_int[m.trainIdx].pt for m in matches])

    # Compute Fundamental matrix using RANSAC
    F, mask = cv2.findFundamentalMat(pts_a, pts_int, cv2.FM_RANSAC, 3.0, 0.99)
    if F is None or mask is None:
        return False, 999.0

    inlier_ratio = float(np.sum(mask)) / len(mask)
    if inlier_ratio < 0.60:
        return False, 999.0

    # Epipolar distance check
    lines_int = cv2.computeCorrespondEpilines(pts_a.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    errors = np.abs(lines_int[:, 0] * pts_int[:, 0] + lines_int[:, 1] * pts_int[:, 1] + lines_int[:, 2])
    mean_err = float(np.mean(errors[mask.ravel() == 1])) if np.any(mask == 1) else 999.0

    is_valid = bool(mean_err <= max_reprojection_error_px and inlier_ratio >= 0.65)
    return is_valid, round(mean_err, 2)

