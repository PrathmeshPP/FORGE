"""F.O.R.G.E — OpenCV 5 Frame Quality & Viewpoint Diversity Analyzer.

Provides deterministic, multi-metric frame quality evaluation and viewpoint curation
for video-to-3D reconstruction using the active OpenCV 5 runtime.

Key Capabilities:
1. Multi-metric sharpness: Dual evaluation using Laplacian variance and Tenengrad (Sobel gradient energy).
2. Radiometric fidelity: Mean intensity, RMS contrast, clipped-highlight and clipped-shadow proportions.
3. Feature distribution: ORB keypoint density, 4x4 spatial grid cell occupancy, and distribution entropy.
4. Temporal & visual disparity: Grayscale histogram correlation and Bhattacharyya distance.
5. Geometric covisibility verification: Feature correspondence matching and uncalibrated epipolar geometry
   verification via Fundamental matrix RANSAC. (Honest reporting: if matches < 8 or RANSAC fails,
   geometric verification is reported as unavailable rather than fabricated).
6. Viewpoint-preserving curation: Divides video into chronological windows to guarantee full temporal
   trajectory coverage, selecting the optimal frame in each window while penalizing degenerate near-duplicates.
"""

from __future__ import annotations

import os
import sys
import json
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def verify_opencv5_runtime() -> str:
    """Verifies that the active OpenCV runtime is 5.x.
    
    Raises:
        RuntimeError: If cv2 is not available or its version does not begin with '5.'.
    """
    version = getattr(cv2, "__version__", "")
    if not version.startswith("5."):
        raise RuntimeError(
            f"OpenCV 5.x runtime is required for this module, but active cv2.__version__ is: '{version}'"
        )
    return version


@dataclass
class FrameQualityMetrics:
    """Quantitative image quality metrics for a single video frame."""
    frame_index: int
    timestamp_sec: float
    laplacian_sharpness: float
    tenengrad_sharpness: float
    mean_intensity: float
    rms_contrast: float
    clipped_highlights_ratio: float
    clipped_shadows_ratio: float
    dynamic_range_p95_p05: float
    orb_keypoint_count: int
    grid_occupied_cells: int  # out of 16
    grid_distribution_entropy: float  # [0.0, 1.0]
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
            "selection_reason": self.selection_reason,
            "uncertainty_note": self.uncertainty_note,
        }


@dataclass
class CurationReport:
    """Complete diagnostic report for video frame analysis and curation."""
    analyzer_module: str
    opencv_version: str
    python_version: str
    video_path: str
    native_fps: float
    total_frames: int
    native_resolution: List[int]
    duration_sec: float
    target_frame_count: int
    candidate_pool_size: int
    candidates: List[FrameQualityMetrics] = field(default_factory=list)
    pairwise_reports: List[PairwiseMatchMetrics] = field(default_factory=list)
    selected_frames: List[SelectedFrameInfo] = field(default_factory=list)
    selection_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyzer_module": self.analyzer_module,
            "opencv_version": self.opencv_version,
            "python_version": self.python_version,
            "video_path": self.video_path,
            "native_fps": self.native_fps,
            "total_frames": self.total_frames,
            "native_resolution": self.native_resolution,
            "duration_sec": self.duration_sec,
            "target_frame_count": self.target_frame_count,
            "candidate_pool_size": self.candidate_pool_size,
            "candidates": [c.to_dict() for c in self.candidates],
            "pairwise_reports": [p.to_dict() for p in self.pairwise_reports],
            "selected_frames": [s.to_dict() for s in self.selected_frames],
            "selection_summary": self.selection_summary,
        }


def compute_single_frame_metrics(
    gray: np.ndarray,
    frame_idx: int,
    timestamp_sec: float,
    orb_detector: Optional[cv2.ORB] = None
) -> Tuple[FrameQualityMetrics, Any, Any]:
    """Computes comprehensive individual image quality and spatial feature metrics.
    
    Returns:
        Tuple of (FrameQualityMetrics, keypoints, descriptors)
    """
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

    # 3. Radiometric Fidelity & Contrast
    mean_val = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    clipped_high = float(np.mean(gray >= 250))
    clipped_low = float(np.mean(gray <= 5))

    p95, p05 = np.percentile(gray, [95, 5])
    dyn_range = float(p95 - p05)

    # 4. ORB Features & Spatial Distribution
    if orb_detector is None:
        orb_detector = cv2.ORB_create(nfeatures=1000)

    kps, des = orb_detector.detectAndCompute(gray, None)
    kp_count = len(kps) if kps is not None else 0

    # 4x4 spatial grid distribution
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

    # Normalized spatial entropy
    if kp_count > 0:
        probs = grid.flatten() / float(kp_count)
        nonzero_p = probs[probs > 0]
        entropy = float(-np.sum(nonzero_p * np.log2(nonzero_p)))
        # Max entropy for 16 uniform cells is log2(16) = 4.0
        normalized_entropy = float(np.clip(entropy / 4.0, 0.0, 1.0))
    else:
        normalized_entropy = 0.0

    # 5. Preliminary composite score (interpretable combination)
    # Rewards focus, contrast, and uniform feature spread; penalizes severe clipping
    clipping_penalty = (clipped_high + clipped_low) * 100.0  # percentage
    entropy_bonus = normalized_entropy * 50.0
    contrast_term = min(rms_contrast, 60.0)
    sharpness_term = math.log1p(max(lap_sharp, 0.0)) * 10.0

    composite = float(sharpness_term + contrast_term + entropy_bonus - clipping_penalty)

    metrics = FrameQualityMetrics(
        frame_index=frame_idx,
        timestamp_sec=round(timestamp_sec, 4),
        laplacian_sharpness=round(lap_sharp, 2),
        tenengrad_sharpness=round(tenengrad, 2),
        mean_intensity=round(mean_val, 2),
        rms_contrast=round(rms_contrast, 2),
        clipped_highlights_ratio=round(clipped_high, 4),
        clipped_shadows_ratio=round(clipped_low, 4),
        dynamic_range_p95_p05=round(dyn_range, 2),
        orb_keypoint_count=kp_count,
        grid_occupied_cells=occupied_cells,
        grid_distribution_entropy=round(normalized_entropy, 4),
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
    """Evaluates inter-frame visual disparity and geometric consistency."""
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

    # 3. Geometric Consistency Verification (Fundamental Matrix RANSAC)
    # Strictly check whether reliable estimation is possible
    geom_avail = False
    geom_model = "none"
    inliers = 0
    inlier_ratio = 0.0
    notes = ""

    if num_matches < 8:
        notes = f"insufficient_matches ({num_matches} < 8 required for epipolar geometry)"
    else:
        pts1 = np.float32([prev_kps[m.queryIdx].pt for m in raw_matches])
        pts2 = np.float32([curr_kps[m.trainIdx].pt for m in raw_matches])

        # Verify points are not all identical/degenerate
        std1 = np.std(pts1, axis=0)
        std2 = np.std(pts2, axis=0)
        if (std1 < 1.0).any() or (std2 < 1.0).any():
            notes = "degenerate_point_distribution (collinear or zero variance)"
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
                    notes = f"epipolar_verified ({inliers}/{num_matches} inliers, {inlier_ratio*100:.1f}%)"
                else:
                    notes = "ransac_estimation_failed (no valid fundamental matrix found)"
            except Exception as e:
                notes = f"ransac_exception ({str(e)})"

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
        verification_notes=notes,
    )


class OpenCV5FrameAnalyzer:
    """Deterministic, quality-aware video frame curation engine using OpenCV 5.x."""

    def __init__(self, orb_nfeatures: int = 1000):
        self.opencv_version = verify_opencv5_runtime()
        self.orb = cv2.ORB_create(nfeatures=orb_nfeatures)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def analyze_video(
        self,
        video_path: str,
        target_num_frames: int = 6,
        candidate_stride: Optional[int] = None,
        max_candidates: int = 60,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> CurationReport:
        """Analyzes candidate frames across the video and selects a high-quality, diverse subset.
        
        Args:
            video_path: Path to video file.
            target_num_frames: Number of curated frames desired.
            candidate_stride: Stride in video frames between candidate evaluations. If None,
                              computed adaptively to yield ~max_candidates.
            max_candidates: Maximum candidate frames evaluated across duration.
            t_start: Optional start timestamp in seconds.
            t_end: Optional end timestamp in seconds.
            
        Returns:
            CurationReport containing candidate metrics, pairwise analysis, and selections.
        """
        vpath = Path(video_path)
        if not vpath.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file via OpenCV: {video_path}")

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

            if total_frames <= 0:
                raise RuntimeError(f"Video has invalid total frame count: {total_frames}")

            # Determine candidate evaluation pool
            f_start = int(round(t_start * native_fps)) if t_start is not None else 0
            f_end = int(round(t_end * native_fps)) if t_end is not None else total_frames
            f_start = max(0, min(f_start, total_frames - 1))
            f_end = max(f_start + 1, min(f_end, total_frames))
            corridor_frames = f_end - f_start

            if candidate_stride is None or candidate_stride <= 0:
                candidate_stride = max(1, corridor_frames // max_candidates)

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

                metrics, kps, des = compute_single_frame_metrics(gray, f_idx, ts, self.orb)
                candidate_metrics.append(metrics)
                candidate_kps_dict[f_idx] = kps
                candidate_des_dict[f_idx] = des
                candidate_gray_dict[f_idx] = gray

            if not candidate_metrics:
                raise RuntimeError(f"No frames could be successfully read from: {video_path}")

            # Execute Chronological Window Curation
            # Divide candidates into target_num_frames windows to guarantee 100% temporal coverage
            target_k = min(target_num_frames, len(candidate_metrics))
            window_size = len(candidate_metrics) / float(target_k)

            selected_frames: List[SelectedFrameInfo] = []
            pairwise_reports: List[PairwiseMatchMetrics] = []

            prev_selected_idx: Optional[int] = None

            for win_idx in range(target_k):
                start_i = int(win_idx * window_size)
                end_i = int((win_idx + 1) * window_size) if win_idx < (target_k - 1) else len(candidate_metrics)
                window_candidates = candidate_metrics[start_i:end_i]

                if not window_candidates:
                    continue

                # Score candidates in window
                best_candidate = window_candidates[0]
                best_adjusted_score = -1e9
                best_reason = "window_baseline"
                best_uncertainty = None
                best_pairwise = None

                for cand in window_candidates:
                    score = cand.composite_quality_score
                    reason = f"highest_composite_score ({score:.1f})"
                    uncertainty = None
                    p_metric = None

                    # If previous frame selected, check inter-frame dynamics
                    if prev_selected_idx is not None:
                        p_metric = compute_pairwise_consistency(
                            candidate_gray_dict[prev_selected_idx],
                            candidate_kps_dict[prev_selected_idx],
                            candidate_des_dict[prev_selected_idx],
                            candidate_gray_dict[cand.frame_index],
                            candidate_kps_dict[cand.frame_index],
                            candidate_des_dict[cand.frame_index],
                            prev_selected_idx,
                            cand.frame_index,
                            self.bf
                        )

                        # Near-duplicate penalty: if image is virtually identical (hist corr > 0.99)
                        if p_metric.histogram_correlation > 0.99:
                            score -= 30.0
                            reason += "; penalized_near_duplicate"

                        # Viewpoint change reward: if geometry is verified and baseline exists
                        if p_metric.geometric_verification_available and p_metric.inlier_ratio > 0.40:
                            score += 15.0
                            reason += f"; epipolar_inliers_verified ({p_metric.inlier_matches})"
                        elif not p_metric.geometric_verification_available:
                            uncertainty = f"geometric_verification_unavailable: {p_metric.verification_notes}"

                    # Mild penalty for extreme clipping
                    if (cand.clipped_highlights_ratio + cand.clipped_shadows_ratio) > 0.05:
                        score -= 20.0
                        reason += "; exposure_clipped_penalty"

                    if score > best_adjusted_score:
                        best_adjusted_score = score
                        best_candidate = cand
                        best_reason = reason
                        best_uncertainty = uncertainty
                        best_pairwise = p_metric

                # Register selected frame
                if best_pairwise is not None:
                    pairwise_reports.append(best_pairwise)

                selected_frames.append(
                    SelectedFrameInfo(
                        selection_rank=len(selected_frames),
                        frame_index=best_candidate.frame_index,
                        timestamp_sec=best_candidate.timestamp_sec,
                        file_path=None,
                        metrics=best_candidate,
                        pairwise_with_prev=best_pairwise,
                        selection_reason=best_reason,
                        uncertainty_note=best_uncertainty,
                    )
                )

                prev_selected_idx = best_candidate.frame_index

            # Build report
            summary = {
                "total_candidates_analyzed": len(candidate_metrics),
                "curated_frame_count": len(selected_frames),
                "temporal_span_sec": (
                    round(selected_frames[-1].timestamp_sec - selected_frames[0].timestamp_sec, 3)
                    if len(selected_frames) > 1 else 0.0
                ),
                "trajectory_coverage_ratio": (
                    round((selected_frames[-1].timestamp_sec - selected_frames[0].timestamp_sec) / duration_sec, 4)
                    if (len(selected_frames) > 1 and duration_sec > 0) else 1.0
                ),
                "mean_selected_laplacian_sharpness": round(
                    float(np.mean([s.metrics.laplacian_sharpness for s in selected_frames])), 2
                ),
                "mean_selected_rms_contrast": round(
                    float(np.mean([s.metrics.rms_contrast for s in selected_frames])), 2
                ),
                "mean_grid_entropy": round(
                    float(np.mean([s.metrics.grid_distribution_entropy for s in selected_frames])), 4
                ),
                "geometric_verification_rate": (
                    round(
                        float(np.mean([1.0 if p.geometric_verification_available else 0.0 for p in pairwise_reports])),
                        4
                    ) if pairwise_reports else 1.0
                ),
            }

            return CurationReport(
                analyzer_module="src.video.opencv5_frame_analyzer",
                opencv_version=self.opencv_version,
                python_version=sys.version.split()[0],
                video_path=str(vpath.resolve()),
                native_fps=round(native_fps, 3),
                total_frames=total_frames,
                native_resolution=[native_w, native_h],
                duration_sec=round(duration_sec, 3),
                target_frame_count=target_num_frames,
                candidate_pool_size=len(candidate_metrics),
                candidates=candidate_metrics,
                pairwise_reports=pairwise_reports,
                selected_frames=selected_frames,
                selection_summary=summary,
            )

        finally:
            cap.release()

    def extract_curated_frames(
        self,
        video_path: str,
        output_dir: str,
        target_num_frames: int = 6,
        prefix: str = "frame",
        save_metadata: bool = True,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], CurationReport]:
        """Runs video analysis and extracts the curated frame images to disk in chronological order.
        
        Returns:
            Tuple of (list of frame dictionaries compatible with ExtractedFrame/SampledFrame, CurationReport)
        """
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
                    # Fallback to adjacent frame if seek fails
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_idx - 1))
                    ret, frame_bgr = cap.read()
                    if not ret:
                        continue

                filename = f"{prefix}_{out_idx:04d}_{s_info.timestamp_sec:.3f}s.png"
                target_fn = out_path / filename

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(target_fn)

                s_info.file_path = str(target_fn.resolve())

                extracted_records.append({
                    "frame_index": out_idx,
                    "video_frame_index": f_idx,
                    "timestamp_seconds": s_info.timestamp_sec,
                    "file_path": str(target_fn.resolve()),
                    "width": frame_rgb.shape[1],
                    "height": frame_rgb.shape[0],
                    "sharpness": s_info.metrics.laplacian_sharpness,
                    "contrast": s_info.metrics.rms_contrast,
                    "composite_score": s_info.metrics.composite_quality_score,
                    "selection_reason": s_info.selection_reason,
                })

        finally:
            cap.release()

        if save_metadata:
            meta_path = out_path / "opencv5_curation_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)

        return extracted_records, report
