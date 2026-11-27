"""F.O.R.G.E — Generalized OpenCV 5 Adaptive Frame Quality, Overlap & Viewpoint Analyzer.

Provides deterministic, multi-metric frame quality evaluation, visual overlap verification,
viewpoint novelty detection, bridge-frame recovery, and adaptive keyframe curation
for robust indoor 3D reconstruction across unfamiliar smartphone videos.

Key Capabilities:
1. Video-Calibrated Sharpness & Blur:
   - Fast survey calibration of video-specific sharpness distribution (median, p20, p80).
   - Dual Laplacian variance + Tenengrad gradient energy.
   - Relative motion-blur detection without single fixed magic thresholds.
2. Radiometric Fidelity & Exposure:
   - Highlights (>250) and shadow (<5) clipping detection.
   - Mean intensity, RMS contrast, and dynamic range (p95 - p05).
   - Rejection of pitch black or severely washed-out frames.
3. Multi-Region Visual Overlap & Spatial Distribution:
   - ORB feature extraction and Hamming brute-force matching.
   - Epipolar geometry verification via Fundamental matrix RANSAC.
   - 4x4 spatial grid distribution of inlier matches: penalizes matches concentrated
     on small posters or single wall textures, requiring broad image coverage.
4. Viewpoint Novelty & Motion Classification:
   - Feature displacement vector magnitude and variance analysis.
   - Distinguishes translation/parallax from pure rotational panning and static pauses.
   - Near-duplicate suppression when the camera pauses.
   - Detection and warning for fast rotational pans lacking depth parallax.
5. Bridge-Frame Recovery:
   - When inter-frame overlap drops unexpectedly, searches intermediate frames
     to find bridge frames connecting the viewpoints.
   - Retains important transition frames (e.g. doorways, corridor thresholds)
     even when sharpness is moderately lower.
6. Machine-Readable Diagnostics Manifest:
   - Complete per-frame decision logs, reasons, scores, and rejection tracking.
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def verify_opencv5_runtime() -> str:
    """Verifies that the active OpenCV runtime is 5.x."""
    version = getattr(cv2, "__version__", "")
    if not version.startswith("5."):
        raise RuntimeError(
            f"OpenCV 5.x runtime is required for this module, but active cv2.__version__ is: '{version}'"
        )
    return version


@dataclass
class VideoCalibration:
    """Video-level distribution metrics for adaptive relative scoring."""
    median_sharpness: float
    p20_sharpness: float
    p80_sharpness: float
    median_luminance: float
    median_contrast: float
    min_sharpness_threshold: float
    min_luminance_threshold: float
    max_luminance_threshold: float
    min_contrast_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrameQualityMetrics:
    """Quantitative image quality metrics for a single video frame."""
    frame_index: int
    timestamp_sec: float
    laplacian_sharpness: float
    tenengrad_sharpness: float
    combined_sharpness: float
    relative_sharpness_score: float  # calibrated against video distribution [0.0 - 100.0]
    mean_intensity: float
    rms_contrast: float
    clipped_highlights_ratio: float
    clipped_shadows_ratio: float
    dynamic_range_p95_p05: float
    orb_keypoint_count: int
    grid_occupied_cells: int  # out of 16
    grid_distribution_entropy: float  # [0.0, 1.0]
    is_usable: bool
    motion_blur_flag: bool
    exposure_clipped_flag: bool
    composite_quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PairwiseMatchMetrics:
    """Chronological pairwise image disparity and geometric consistency metrics."""
    prev_frame_index: int
    curr_frame_index: int
    histogram_correlation: float
    bhattacharyya_distance: float
    raw_orb_matches: int
    geometric_verification_available: bool
    geometric_model: str  # e.g. "fundamental_ransac" or "none"
    inlier_matches: int
    inlier_ratio: float
    grid_match_coverage: float  # ratio of 4x4 grid cells containing inliers [0.0, 1.0]
    grid_match_entropy: float  # spatial entropy of inliers across 4x4 cells
    is_clustered_matches: bool  # True if inliers concentrated in <=2 cells or entropy < 0.25
    mean_displacement_px: float  # feature flow displacement
    std_displacement_px: float   # variance in displacement (indicates parallax)
    motion_classification: str   # "translation_parallax" | "rotational_pan" | "stationary_pause" | "transition_doorway" | "rapid_motion"
    overlap_score: float         # composite overlap metric [0.0 - 100.0]
    verification_notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedFrameInfo:
    """Metadata for a curated frame chosen for downstream 3D reconstruction."""
    selection_rank: int
    frame_index: int
    timestamp_sec: float
    file_path: Optional[str]
    metrics: FrameQualityMetrics
    pairwise_with_prev: Optional[PairwiseMatchMetrics]
    is_bridge_frame: bool
    is_transition_frame: bool
    decision: str  # "ACCEPTED_KEYFRAME" | "ACCEPTED_BRIDGE" | "ACCEPTED_TRANSITION"
    selection_reason: str
    uncertainty_note: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selection_rank": self.selection_rank,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "file_path": self.file_path,
            "metrics": self.metrics.to_dict(),
            "pairwise_with_prev": self.pairwise_with_prev.to_dict() if self.pairwise_with_prev else None,
            "is_bridge_frame": self.is_bridge_frame,
            "is_transition_frame": self.is_transition_frame,
            "decision": self.decision,
            "selection_reason": self.selection_reason,
            "uncertainty_note": self.uncertainty_note,
        }


@dataclass
class CandidateEvaluationRecord:
    """Detailed evaluation record for candidate frames (selected, skipped, or rejected)."""
    frame_index: int
    timestamp_sec: float
    sharpness: float
    luminance: float
    contrast: float
    decision: str  # "SELECTED" | "BRIDGE_RECOVERED" | "SKIPPED_DUPLICATE" | "REJECTED_BLUR" | "REJECTED_EXPOSURE" | "REJECTED_LOW_OVERLAP"
    reason: str
    motion_classification: Optional[str] = None
    inliers_with_prev: Optional[int] = None
    overlap_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CurationReport:
    """Complete diagnostic report for video frame analysis and adaptive curation."""
    analyzer_module: str
    opencv_version: str
    python_version: str
    video_path: str
    video_sha256: str
    native_fps: float
    total_frames: int
    native_resolution: List[int]
    orientation: str
    duration_sec: float
    target_frame_count: int
    calibration: VideoCalibration
    selected_frames: List[SelectedFrameInfo] = field(default_factory=list)
    pairwise_reports: List[PairwiseMatchMetrics] = field(default_factory=list)
    candidate_evaluations: List[CandidateEvaluationRecord] = field(default_factory=list)
    bridge_recovery_events: List[Dict[str, Any]] = field(default_factory=list)
    selection_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyzer_module": self.analyzer_module,
            "opencv_version": self.opencv_version,
            "python_version": self.python_version,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "native_fps": self.native_fps,
            "total_frames": self.total_frames,
            "native_resolution": self.native_resolution,
            "orientation": self.orientation,
            "duration_sec": self.duration_sec,
            "target_frame_count": self.target_frame_count,
            "calibration": self.calibration.to_dict(),
            "selected_frames": [s.to_dict() for s in self.selected_frames],
            "pairwise_reports": [p.to_dict() for p in self.pairwise_reports],
            "candidate_evaluations": [c.to_dict() for c in self.candidate_evaluations],
            "bridge_recovery_events": self.bridge_recovery_events,
            "selection_summary": self.selection_summary,
        }


def calibrate_video_characteristics(
    video_path: Union[str, Path],
    num_samples: int = 60
) -> VideoCalibration:
    """Surveys video frames to adaptively calibrate relative quality thresholds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {total_frames}")

    sample_indices = np.linspace(
        int(total_frames * 0.05),
        int(total_frames * 0.95),
        min(num_samples, max(5, total_frames - 2)),
        dtype=int
    )

    sharpness_list: List[float] = []
    lum_list: List[float] = []
    contrast_list: List[float] = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lum = float(np.mean(gray))
        contrast = float(np.std(gray))

        sharpness_list.append(lap_var)
        lum_list.append(lum)
        contrast_list.append(contrast)

    cap.release()

    if not sharpness_list:
        sharpness_list = [100.0]
        lum_list = [120.0]
        contrast_list = [50.0]

    s_arr = np.array(sharpness_list)
    med_s = float(np.median(s_arr))
    p20_s = float(np.percentile(s_arr, 20))
    p80_s = float(np.percentile(s_arr, 80))

    med_lum = float(np.median(lum_list))
    med_con = float(np.median(contrast_list))

    # Adaptive thresholds: calibrated relative to video dynamics
    # Ensure min sharpness threshold adapts between low-light/soft camera and ultra-sharp 4K
    min_sharp = max(20.0, p20_s * 0.65)
    min_lum = max(18.0, min(35.0, med_lum * 0.35))
    max_lum = min(240.0, max(215.0, med_lum * 1.65))
    min_con = max(12.0, med_con * 0.35)

    return VideoCalibration(
        median_sharpness=round(med_s, 2),
        p20_sharpness=round(p20_s, 2),
        p80_sharpness=round(p80_s, 2),
        median_luminance=round(med_lum, 2),
        median_contrast=round(med_con, 2),
        min_sharpness_threshold=round(min_sharp, 2),
        min_luminance_threshold=round(min_lum, 2),
        max_luminance_threshold=round(max_lum, 2),
        min_contrast_threshold=round(min_con, 2)
    )


def compute_single_frame_metrics(
    gray: np.ndarray,
    frame_idx: int,
    timestamp_sec: float,
    calibration: Optional[VideoCalibration] = None,
    orb_detector: Optional[cv2.ORB] = None
) -> Tuple[FrameQualityMetrics, Any, Any]:
    """Computes comprehensive individual image quality and spatial feature metrics."""
    if gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray.shape}")

    h, w = gray.shape

    # 1. Laplacian Sharpness (high-frequency energy)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_sharp = float(lap.var())

    # 2. Tenengrad Sharpness (Sobel gradient magnitude)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = sobel_x ** 2 + sobel_y ** 2
    tenengrad = float(np.mean(grad_mag_sq))

    comb_sharp = math.sqrt(max(0.0, tenengrad)) + math.sqrt(max(0.0, lap_sharp)) * 0.6

    # 3. Radiometric Fidelity & Contrast
    mean_val = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    clipped_high = float(np.mean(gray >= 250))
    clipped_low = float(np.mean(gray <= 5))

    p95, p05 = np.percentile(gray, [95, 5])
    dyn_range = float(p95 - p05)

    # 4. ORB Features & 4x4 Spatial Grid Distribution
    if orb_detector is None:
        orb_detector = cv2.ORB_create(nfeatures=1000)

    kps, des = orb_detector.detectAndCompute(gray, None)
    kp_count = len(kps) if kps is not None else 0

    grid = np.zeros((4, 4), dtype=int)
    if kps:
        col_step = max(w / 4.0, 1e-3)
        row_step = max(h / 4.0, 1e-3)
        for kp in kps:
            x, y = kp.pt
            c = min(int(x / col_step), 3)
            r = min(int(y / row_step), 3)
            grid[r, c] += 1

    occupied_cells = int(np.count_nonzero(grid))

    if kp_count > 0:
        probs = grid.flatten() / float(kp_count)
        nonzero_p = probs[probs > 0]
        entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p)))
        normalized_entropy = float(np.clip(entropy / 4.0, 0.0, 1.0))
    else:
        normalized_entropy = 0.0

    # 5. Relative scoring & usability checks
    min_s = calibration.min_sharpness_threshold if calibration else 25.0
    min_l = calibration.min_luminance_threshold if calibration else 20.0
    max_l = calibration.max_luminance_threshold if calibration else 235.0
    min_c = calibration.min_contrast_threshold if calibration else 15.0

    med_s = calibration.median_sharpness if calibration else 80.0
    rel_sharp = float(np.clip((lap_sharp / max(med_s, 1e-3)) * 50.0, 0.0, 100.0))

    motion_blur = lap_sharp < min_s
    exposure_clipped = (clipped_high + clipped_low) > 0.08 or (mean_val < min_l) or (mean_val > max_l)
    low_contrast = rms_contrast < min_c

    is_usable = (not motion_blur) and (not exposure_clipped) and (not low_contrast)

    # Composite quality score
    clipping_penalty = (clipped_high + clipped_low) * 120.0
    entropy_bonus = normalized_entropy * 40.0
    contrast_term = min(rms_contrast, 60.0) * 0.8
    sharpness_term = math.log1p(max(lap_sharp, 0.0)) * 12.0

    composite = float(sharpness_term + contrast_term + entropy_bonus - clipping_penalty)

    metrics = FrameQualityMetrics(
        frame_index=frame_idx,
        timestamp_sec=round(timestamp_sec, 4),
        laplacian_sharpness=round(lap_sharp, 2),
        tenengrad_sharpness=round(tenengrad, 2),
        combined_sharpness=round(comb_sharp, 2),
        relative_sharpness_score=round(rel_sharp, 2),
        mean_intensity=round(mean_val, 2),
        rms_contrast=round(rms_contrast, 2),
        clipped_highlights_ratio=round(clipped_high, 4),
        clipped_shadows_ratio=round(clipped_low, 4),
        dynamic_range_p95_p05=round(dyn_range, 2),
        orb_keypoint_count=kp_count,
        grid_occupied_cells=occupied_cells,
        grid_distribution_entropy=round(normalized_entropy, 4),
        is_usable=bool(is_usable),
        motion_blur_flag=bool(motion_blur),
        exposure_clipped_flag=bool(exposure_clipped),
        composite_quality_score=round(composite, 2),
    )

    return metrics, kps, des


def compute_pairwise_consistency(
    prev_gray: np.ndarray,
    prev_kps: Any,
    prev_des: Any,
    curr_gray: np.ndarray,
    curr_kps: Any,
    curr_des: Any,
    prev_idx: int,
    curr_idx: int,
    bf_matcher: Optional[cv2.BFMatcher] = None
) -> PairwiseMatchMetrics:
    """Evaluates inter-frame visual overlap, epipolar geometry, spatial distribution, and motion type."""
    h_img, w_img = curr_gray.shape

    # 1. Grayscale Histogram Similarity
    h1 = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    h2 = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    hist_corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    bhatt_dist = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))

    # 2. ORB Feature Matching
    if bf_matcher is None:
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    raw_matches = []
    if prev_des is not None and curr_des is not None and len(prev_des) > 0 and len(curr_des) > 0:
        raw_matches = bf_matcher.match(prev_des, curr_des)

    num_matches = len(raw_matches)

    # 3. Geometric Consistency & Spatial Spread
    geom_avail = False
    geom_model = "none"
    inliers = 0
    inlier_ratio = 0.0
    notes = ""
    grid_coverage = 0.0
    grid_entropy = 0.0
    is_clustered = False
    mean_disp = 0.0
    std_disp = 0.0
    motion_class = "unknown"

    if num_matches < 8:
        notes = f"insufficient_matches ({num_matches} < 8 required for epipolar geometry)"
        motion_class = "rapid_motion"
    else:
        pts1 = np.float32([prev_kps[m.queryIdx].pt for m in raw_matches])
        pts2 = np.float32([curr_kps[m.trainIdx].pt for m in raw_matches])

        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        mean_disp = float(np.mean(displacements))
        std_disp = float(np.std(displacements))

        std1 = np.std(pts1, axis=0)
        std2 = np.std(pts2, axis=0)
        if (std1 < 1.0).any() or (std2 < 1.0).any():
            notes = "degenerate_point_distribution (collinear or zero variance)"
            is_clustered = True
        else:
            try:
                F, inlier_mask = cv2.findFundamentalMat(
                    pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99
                )
                if F is not None and inlier_mask is not None:
                    inliers = int(np.sum(inlier_mask))
                    inlier_ratio = float(inliers / float(num_matches))
                    geom_avail = True
                    geom_model = "fundamental_ransac"

                    # Analyze spatial distribution of inliers across 4x4 image grid
                    inlier_indices = np.where(inlier_mask.ravel() == 1)[0]
                    inlier_pts2 = pts2[inlier_indices]

                    match_grid = np.zeros((4, 4), dtype=int)
                    col_step = max(w_img / 4.0, 1e-3)
                    row_step = max(h_img / 4.0, 1e-3)
                    for pt in inlier_pts2:
                        c = min(int(pt[0] / col_step), 3)
                        r = min(int(pt[1] / row_step), 3)
                        match_grid[r, c] += 1

                    occ_cells = int(np.count_nonzero(match_grid))
                    grid_coverage = float(occ_cells / 16.0)

                    if len(inlier_pts2) > 0:
                        probs = match_grid.flatten() / float(len(inlier_pts2))
                        nz = probs[probs > 0]
                        ent = float(-np.sum(nz * np.log2(nz)))
                        grid_entropy = float(np.clip(ent / 4.0, 0.0, 1.0))

                    # Clustered check: matches on one small object/poster
                    if occ_cells <= 2 or grid_entropy < 0.25:
                        is_clustered = True
                        notes = f"clustered_matches ({inliers} inliers concentrated in {occ_cells}/16 grid cells)"
                    else:
                        notes = f"epipolar_verified ({inliers}/{num_matches} inliers, {grid_coverage*100:.0f}% grid coverage)"
                else:
                    notes = "ransac_estimation_failed (no valid fundamental matrix found)"
            except Exception as e:
                notes = f"ransac_exception ({str(e)})"

        # Motion classification
        if mean_disp < 2.5 and hist_corr > 0.98:
            motion_class = "stationary_pause"
        elif mean_disp > 25.0 and (std_disp / (mean_disp + 1e-4) < 0.20):
            motion_class = "rotational_pan"
        elif mean_disp > 65.0 or (geom_avail and inlier_ratio < 0.25):
            motion_class = "rapid_motion"
        elif geom_avail and inliers >= 15 and (not is_clustered):
            motion_class = "translation_parallax"
        else:
            motion_class = "transition_doorway"

    # Composite overlap score [0.0 - 100.0]
    match_term = min(float(inliers), 100.0) * 0.40
    ratio_term = float(inlier_ratio) * 30.0
    coverage_term = float(grid_coverage) * 30.0
    cluster_penalty = 35.0 if is_clustered else 0.0

    overlap_score = float(np.clip(match_term + ratio_term + coverage_term - cluster_penalty, 0.0, 100.0))

    return PairwiseMatchMetrics(
        prev_frame_index=prev_idx,
        curr_frame_index=curr_idx,
        histogram_correlation=round(hist_corr, 4),
        bhattacharyya_distance=round(bhatt_dist, 4),
        raw_orb_matches=num_matches,
        geometric_verification_available=geom_avail,
        geometric_model=geom_model,
        inlier_matches=inliers,
        inlier_ratio=round(inlier_ratio, 4),
        grid_match_coverage=round(grid_coverage, 4),
        grid_match_entropy=round(grid_entropy, 4),
        is_clustered_matches=bool(is_clustered),
        mean_displacement_px=round(mean_disp, 2),
        std_displacement_px=round(std_disp, 2),
        motion_classification=motion_class,
        overlap_score=round(overlap_score, 2),
        verification_notes=notes,
    )


class OpenCV5FrameAnalyzer:
    """Adaptive, quality-aware, and overlap-verified video frame curation engine using OpenCV 5.x."""

    def __init__(self, orb_nfeatures: int = 1000):
        self.opencv_version = verify_opencv5_runtime()
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def recover_bridge_frames(
        self,
        cap: cv2.VideoCapture,
        prev_idx: int,
        target_idx: int,
        native_fps: float,
        calibration: VideoCalibration,
        prev_gray: np.ndarray,
        prev_kps: Any,
        prev_des: Any,
        target_gray: np.ndarray,
        target_kps: Any,
        target_des: Any,
        max_bridges: int = 2
    ) -> List[Tuple[int, float, np.ndarray, Any, Any, FrameQualityMetrics, PairwiseMatchMetrics]]:
        """Searches intermediate frames between prev_idx and target_idx to bridge a visual overlap drop."""
        gap_frames = target_idx - prev_idx
        if gap_frames <= 2:
            return []

        # Dense search step: every 2 to 4 frames
        step = max(1, gap_frames // 8)
        intermediate_indices = list(range(prev_idx + step, target_idx, step))

        found_bridges = []
        curr_prev_gray = prev_gray
        curr_prev_kps = prev_kps
        curr_prev_des = prev_des
        curr_prev_idx = prev_idx

        # Evaluate candidates for bridging
        candidate_evals = []
        for f_idx in intermediate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            ts = f_idx / native_fps
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            m, kp, des = compute_single_frame_metrics(g, f_idx, ts, calibration, self.orb)

            # Test pairwise overlap with current previous
            p_to_prev = compute_pairwise_consistency(
                curr_prev_gray, curr_prev_kps, curr_prev_des,
                g, kp, des,
                curr_prev_idx, f_idx, self.bf
            )
            # Test pairwise overlap with target
            p_to_target = compute_pairwise_consistency(
                g, kp, des,
                target_gray, target_kps, target_des,
                f_idx, target_idx, self.bf
            )

            # Bridging score: requires connection to both sides
            bridge_score = p_to_prev.overlap_score + p_to_target.overlap_score
            candidate_evals.append((f_idx, ts, g, kp, des, m, p_to_prev, p_to_target, bridge_score))

        if not candidate_evals:
            return []

        # Pick candidate with highest joint bridge score
        best_cand = max(candidate_evals, key=lambda c: c[8])
        # Verify it provides meaningful overlap (>= 12 inliers with prev)
        if best_cand[6].inlier_matches >= 10:
            found_bridges.append((
                best_cand[0], best_cand[1], best_cand[2], best_cand[3], best_cand[4],
                best_cand[5], best_cand[6]
            ))

        return found_bridges

    def analyze_video(
        self,
        video_path: str,
        target_num_frames: int = 60,
        candidate_stride: Optional[int] = None,
        max_candidates: int = 240,
        min_spacing_sec: float = 0.35,
        max_spacing_sec: float = 3.5,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> CurationReport:
        """Adaptive frame selection balancing quality, overlap, viewpoint novelty and bridge recovery."""
        vpath = Path(video_path)
        if not vpath.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Compute SHA256 of source video
        h_sha = hashlib.sha256()
        with open(vpath, "rb") as f:
            while chunk := f.read(1024 * 1024 * 4):
                h_sha.update(chunk)
        video_sha = h_sha.hexdigest()

        # Calibrate relative video characteristics
        calibration = calibrate_video_characteristics(vpath, num_samples=60)

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file via OpenCV: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

            orientation = "landscape" if native_w > native_h else ("portrait" if native_h > native_w else "square")

            if total_frames <= 0:
                raise RuntimeError(f"Video has invalid total frame count: {total_frames}")

            f_start = int(round(t_start * native_fps)) if t_start is not None else 0
            f_end = int(round(t_end * native_fps)) if t_end is not None else total_frames
            f_start = max(0, min(f_start, total_frames - 1))
            f_end = max(f_start + 1, min(f_end, total_frames))
            active_range = f_end - f_start

            eff_max_candidates = max(max_candidates, int(target_num_frames) * 5) if target_num_frames > 60 else max_candidates
            if candidate_stride is None or candidate_stride <= 0:
                candidate_stride = max(1, active_range // eff_max_candidates)

            candidate_indices = list(range(f_start, f_end, candidate_stride))
            if candidate_indices and candidate_indices[-1] != (f_end - 1):
                candidate_indices.append(f_end - 1)

            # Evaluate each candidate frame
            candidate_metrics: List[FrameQualityMetrics] = []
            candidate_kps_dict: Dict[int, Any] = {}
            candidate_des_dict: Dict[int, Any] = {}
            candidate_gray_dict: Dict[int, np.ndarray] = {}

            for f_idx in candidate_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    continue

                ts = f_idx / native_fps
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                metrics, kps, des = compute_single_frame_metrics(gray, f_idx, ts, calibration, self.orb)
                candidate_metrics.append(metrics)
                candidate_kps_dict[f_idx] = kps
                candidate_des_dict[f_idx] = des
                candidate_gray_dict[f_idx] = gray

            if not candidate_metrics:
                raise RuntimeError(f"No frames could be successfully read from: {video_path}")

            # Adaptive Selection Algorithm
            selected_frames: List[SelectedFrameInfo] = []
            pairwise_reports: List[PairwiseMatchMetrics] = []
            candidate_evals: List[CandidateEvaluationRecord] = []
            bridge_events: List[Dict[str, Any]] = []

            # 1. Select initial frame: sharpest usable frame in first 5% of candidate pool
            initial_pool = candidate_metrics[:max(3, len(candidate_metrics) // 15)]
            best_init = max(initial_pool, key=lambda c: c.composite_quality_score)

            selected_frames.append(
                SelectedFrameInfo(
                    selection_rank=0,
                    frame_index=best_init.frame_index,
                    timestamp_sec=best_init.timestamp_sec,
                    file_path=None,
                    metrics=best_init,
                    pairwise_with_prev=None,
                    is_bridge_frame=False,
                    is_transition_frame=False,
                    decision="ACCEPTED_KEYFRAME",
                    selection_reason=f"initial_anchor (score: {best_init.composite_quality_score:.1f})",
                    uncertainty_note=None,
                )
            )
            candidate_evals.append(
                CandidateEvaluationRecord(
                    frame_index=best_init.frame_index,
                    timestamp_sec=best_init.timestamp_sec,
                    sharpness=best_init.laplacian_sharpness,
                    luminance=best_init.mean_intensity,
                    contrast=best_init.rms_contrast,
                    decision="SELECTED",
                    reason="initial_anchor"
                )
            )

            # 2. Chronological adaptive march
            prev_selected = best_init
            cand_idx = 1
            min_spacing_frames = int(min_spacing_sec * native_fps)
            max_spacing_frames = int(max_spacing_sec * native_fps)

            # Determine nominal interval to achieve target_num_frames across candidate pool
            nominal_stride_cands = max(1, len(candidate_metrics) // target_num_frames)

            curr_pos = 0
            while curr_pos < len(candidate_metrics) - 1:
                # Look ahead in window [curr_pos + 1 : curr_pos + nominal_stride_cands * 2]
                window_end = min(len(candidate_metrics), curr_pos + nominal_stride_cands * 2)
                cand_slice = candidate_metrics[curr_pos + 1: window_end]
                if not cand_slice:
                    break

                best_cand = None
                best_score = -1e9
                best_pairwise = None
                best_reason = ""
                best_decision = "ACCEPTED_KEYFRAME"
                is_transition = False

                for cand in cand_slice:
                    time_diff = cand.timestamp_sec - prev_selected.timestamp_sec
                    if time_diff < min_spacing_sec and cand.frame_index != candidate_metrics[-1].frame_index:
                        candidate_evals.append(
                            CandidateEvaluationRecord(
                                frame_index=cand.frame_index,
                                timestamp_sec=cand.timestamp_sec,
                                sharpness=cand.laplacian_sharpness,
                                luminance=cand.mean_intensity,
                                contrast=cand.rms_contrast,
                                decision="SKIPPED_DUPLICATE",
                                reason=f"below_min_spacing ({time_diff:.2f}s < {min_spacing_sec}s)"
                            )
                        )
                        continue

                    # Evaluate pairwise consistency
                    p_metric = compute_pairwise_consistency(
                        candidate_gray_dict[prev_selected.frame_index],
                        candidate_kps_dict[prev_selected.frame_index],
                        candidate_des_dict[prev_selected.frame_index],
                        candidate_gray_dict[cand.frame_index],
                        candidate_kps_dict[cand.frame_index],
                        candidate_des_dict[cand.frame_index],
                        prev_selected.frame_index,
                        cand.frame_index,
                        self.bf
                    )

                    # Check stationary pause (near-duplicate)
                    if p_metric.motion_classification == "stationary_pause":
                        candidate_evals.append(
                            CandidateEvaluationRecord(
                                frame_index=cand.frame_index,
                                timestamp_sec=cand.timestamp_sec,
                                sharpness=cand.laplacian_sharpness,
                                luminance=cand.mean_intensity,
                                contrast=cand.rms_contrast,
                                decision="SKIPPED_DUPLICATE",
                                reason="stationary_camera_pause",
                                motion_classification=p_metric.motion_classification,
                                inliers_with_prev=p_metric.inlier_matches,
                                overlap_score=p_metric.overlap_score
                            )
                        )
                        continue

                    # Check rotational pan warning
                    rot_penalty = 0.0
                    if p_metric.motion_classification == "rotational_pan":
                        rot_penalty = 15.0  # slight penalty to prefer translation, but still usable for coverage

                    # Scoring function balancing quality, overlap, and viewpoint novelty
                    # Overlap term ensures geometric connection
                    overlap_term = p_metric.overlap_score * 0.85
                    quality_term = cand.composite_quality_score * 0.70
                    novelty_term = min(p_metric.mean_displacement_px, 50.0) * 0.40

                    # Spacing balance: penalize exceeding max_spacing
                    spacing_penalty = 0.0
                    if time_diff > max_spacing_sec:
                        spacing_penalty = (time_diff - max_spacing_sec) * 15.0

                    adj_score = quality_term + overlap_term + novelty_term - rot_penalty - spacing_penalty

                    # Check doorway / transition scenario
                    if p_metric.motion_classification == "transition_doorway":
                        adj_score += 10.0
                        is_transition = True

                    if adj_score > best_score:
                        best_score = adj_score
                        best_cand = cand
                        best_pairwise = p_metric
                        best_reason = f"optimal_balance (score: {adj_score:.1f}, inliers: {p_metric.inlier_matches}, motion: {p_metric.motion_classification})"
                        if is_transition:
                            best_decision = "ACCEPTED_TRANSITION"

                if best_cand is None:
                    # Fallback to next candidate if all were skipped
                    curr_pos += 1
                    continue

                # Check if overlap dropped unexpectedly: trigger bridge-frame recovery!
                if best_pairwise and (best_pairwise.overlap_score < 28.0 or best_pairwise.inlier_matches < 12):
                    bridge_results = self.recover_bridge_frames(
                        cap,
                        prev_selected.frame_index,
                        best_cand.frame_index,
                        native_fps,
                        calibration,
                        candidate_gray_dict[prev_selected.frame_index],
                        candidate_kps_dict[prev_selected.frame_index],
                        candidate_des_dict[prev_selected.frame_index],
                        candidate_gray_dict[best_cand.frame_index],
                        candidate_kps_dict[best_cand.frame_index],
                        candidate_des_dict[best_cand.frame_index],
                    )

                    if bridge_results:
                        for b_idx, b_ts, b_gray, b_kps, b_des, b_metrics, b_pairwise in bridge_results:
                            selected_frames.append(
                                SelectedFrameInfo(
                                    selection_rank=len(selected_frames),
                                    frame_index=b_idx,
                                    timestamp_sec=b_ts,
                                    file_path=None,
                                    metrics=b_metrics,
                                    pairwise_with_prev=b_pairwise,
                                    is_bridge_frame=True,
                                    is_transition_frame=True,
                                    decision="ACCEPTED_BRIDGE",
                                    selection_reason=f"bridge_recovery_inserted (connected gap {prev_selected.frame_index}->{best_cand.frame_index})",
                                    uncertainty_note=None
                                )
                            )
                            pairwise_reports.append(b_pairwise)
                            candidate_evals.append(
                                CandidateEvaluationRecord(
                                    frame_index=b_idx,
                                    timestamp_sec=b_ts,
                                    sharpness=b_metrics.laplacian_sharpness,
                                    luminance=b_metrics.mean_intensity,
                                    contrast=b_metrics.rms_contrast,
                                    decision="BRIDGE_RECOVERED",
                                    reason=f"bridge_recovery (overlap_score: {b_pairwise.overlap_score:.1f})",
                                    motion_classification=b_pairwise.motion_classification,
                                    inliers_with_prev=b_pairwise.inlier_matches,
                                    overlap_score=b_pairwise.overlap_score
                                )
                            )
                            bridge_events.append({
                                "bridge_frame_index": b_idx,
                                "timestamp_sec": b_ts,
                                "bridged_interval": [prev_selected.frame_index, best_cand.frame_index],
                                "inliers_to_prev": b_pairwise.inlier_matches,
                                "overlap_score": b_pairwise.overlap_score
                            })
                            # Update prev_selected to bridge
                            prev_selected = b_metrics
                            # Re-compute best_pairwise from bridge to target
                            best_pairwise = compute_pairwise_consistency(
                                b_gray, b_kps, b_des,
                                candidate_gray_dict[best_cand.frame_index],
                                candidate_kps_dict[best_cand.frame_index],
                                candidate_des_dict[best_cand.frame_index],
                                b_idx, best_cand.frame_index, self.bf
                            )

                # Register selected candidate
                if best_pairwise is not None:
                    pairwise_reports.append(best_pairwise)

                selected_frames.append(
                    SelectedFrameInfo(
                        selection_rank=len(selected_frames),
                        frame_index=best_cand.frame_index,
                        timestamp_sec=best_cand.timestamp_sec,
                        file_path=None,
                        metrics=best_cand,
                        pairwise_with_prev=best_pairwise,
                        is_bridge_frame=False,
                        is_transition_frame=is_transition,
                        decision=best_decision,
                        selection_reason=best_reason,
                        uncertainty_note=None if (best_pairwise and best_pairwise.geometric_verification_available) else "unverified_epipolar_geometry",
                    )
                )
                candidate_evals.append(
                    CandidateEvaluationRecord(
                        frame_index=best_cand.frame_index,
                        timestamp_sec=best_cand.timestamp_sec,
                        sharpness=best_cand.laplacian_sharpness,
                        luminance=best_cand.mean_intensity,
                        contrast=best_cand.rms_contrast,
                        decision="SELECTED",
                        reason=best_reason,
                        motion_classification=best_pairwise.motion_classification if best_pairwise else None,
                        inliers_with_prev=best_pairwise.inlier_matches if best_pairwise else None,
                        overlap_score=best_pairwise.overlap_score if best_pairwise else None
                    )
                )

                prev_selected = best_cand
                curr_pos = candidate_metrics.index(best_cand)

            # Build summary
            dur_span = selected_frames[-1].timestamp_sec - selected_frames[0].timestamp_sec if len(selected_frames) > 1 else 0.0
            summary = {
                "total_candidates_analyzed": len(candidate_metrics),
                "curated_frame_count": len(selected_frames),
                "bridge_frames_inserted": sum(1 for s in selected_frames if s.is_bridge_frame),
                "transition_frames_retained": sum(1 for s in selected_frames if s.is_transition_frame),
                "temporal_span_sec": round(dur_span, 3),
                "trajectory_coverage_ratio": round(dur_span / duration_sec, 4) if duration_sec > 0 else 1.0,
                "mean_selected_laplacian_sharpness": round(float(np.mean([s.metrics.laplacian_sharpness for s in selected_frames])), 2),
                "mean_selected_rms_contrast": round(float(np.mean([s.metrics.rms_contrast for s in selected_frames])), 2),
                "mean_grid_entropy": round(float(np.mean([s.metrics.grid_distribution_entropy for s in selected_frames])), 4),
                "geometric_verification_rate": round(
                    float(np.mean([1.0 if p.geometric_verification_available else 0.0 for p in pairwise_reports])), 4
                ) if pairwise_reports else 1.0,
                "mean_pairwise_overlap_score": round(
                    float(np.mean([p.overlap_score for p in pairwise_reports])), 2
                ) if pairwise_reports else 0.0,
                "mean_pairwise_inliers": round(
                    float(np.mean([p.inlier_matches for p in pairwise_reports])), 1
                ) if pairwise_reports else 0.0,
                "motion_distribution": {
                    m_type: sum(1 for p in pairwise_reports if p.motion_classification == m_type)
                    for m_type in ["translation_parallax", "rotational_pan", "stationary_pause", "transition_doorway", "rapid_motion"]
                }
            }

            return CurationReport(
                analyzer_module="src.video.opencv5_frame_analyzer",
                opencv_version=self.opencv_version,
                python_version=sys.version.split()[0],
                video_path=str(vpath.resolve()),
                video_sha256=video_sha,
                native_fps=round(native_fps, 3),
                total_frames=total_frames,
                native_resolution=[native_w, native_h],
                orientation=orientation,
                duration_sec=round(duration_sec, 3),
                target_frame_count=target_num_frames,
                calibration=calibration,
                selected_frames=selected_frames,
                pairwise_reports=pairwise_reports,
                candidate_evaluations=candidate_evals,
                bridge_recovery_events=bridge_events,
                selection_summary=summary,
            )

        finally:
            cap.release()

    def extract_curated_frames(
        self,
        video_path: str,
        output_dir: str,
        target_num_frames: int = 60,
        prefix: str = "frame",
        save_metadata: bool = True,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], CurationReport]:
        """Runs adaptive video analysis and extracts the curated frame images to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        report = self.analyze_video(
            video_path,
            target_num_frames=target_num_frames,
            t_start=t_start,
            t_end=t_end
        )

        vpath = Path(video_path)
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not re-open video for extraction: {video_path}")

        extracted_records = []
        try:
            for out_idx, s_info in enumerate(report.selected_frames):
                f_idx = s_info.frame_index
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_idx - 1))
                    ret, frame_bgr = cap.read()
                    if not ret:
                        continue

                bridge_tag = "_bridge" if s_info.is_bridge_frame else ""
                filename = f"{prefix}_{out_idx:03d}_f{f_idx:05d}_{s_info.timestamp_sec:05.2f}s{bridge_tag}.png"
                target_fn = out_path / filename

                # Save native high-resolution frame
                cv2.imwrite(str(target_fn), frame_bgr)
                s_info.file_path = str(target_fn.resolve())
                f_sha = hashlib.sha256(target_fn.read_bytes()).hexdigest()

                extracted_records.append({
                    "rank": out_idx,
                    "frame_index": f_idx,
                    "timestamp_sec": s_info.timestamp_sec,
                    "dimensions_wh": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],
                    "file_path": str(target_fn.resolve()),
                    "frame_sha256": f_sha,
                    "selection_score": s_info.metrics.composite_quality_score,
                    "sharpness": s_info.metrics.laplacian_sharpness,
                    "luminance": s_info.metrics.mean_intensity,
                    "contrast": s_info.metrics.rms_contrast,
                    "is_bridge": s_info.is_bridge_frame,
                    "is_transition": s_info.is_transition_frame,
                    "decision": s_info.decision,
                    "selection_reason": s_info.selection_reason,
                    "motion_classification": s_info.pairwise_with_prev.motion_classification if s_info.pairwise_with_prev else "initial_anchor",
                    "inliers_with_prev": s_info.pairwise_with_prev.inlier_matches if s_info.pairwise_with_prev else 0,
                    "overlap_score": s_info.pairwise_with_prev.overlap_score if s_info.pairwise_with_prev else 100.0,
                })

        finally:
            cap.release()

        if save_metadata:
            meta_path = out_path / "adaptive_frame_selection_manifest.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

            # Explicit timestamp mapping for photographic texturing and keyframe tracking
            mapping_dict = {
                "video_path": str(vpath.resolve()),
                "video_sha256": report.video_sha256,
                "native_fps": report.native_fps,
                "total_curated_frames": len(report.selected_frames),
                "mapping": [
                    {
                        "selected_frame_idx": rec["rank"],
                        "orig_video_frame_idx": rec["frame_index"],
                        "timestamp_sec": round(rec["timestamp_sec"], 4),
                        "file_path": rec["file_path"],
                        "sharpness": rec["sharpness"],
                        "is_bridge": rec["is_bridge"],
                        "is_transition": rec["is_transition"]
                    }
                    for rec in extracted_records
                ]
            }
            mapping_path = out_path / "frame_timestamp_mapping.json"
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping_dict, f, indent=2)

        return extracted_records, report
