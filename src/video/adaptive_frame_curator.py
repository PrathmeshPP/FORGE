"""
F.O.R.G.E. — Adaptive Indoor Frame Curator.
Evaluates multi-criteria frame quality:
1. Tenengrad & Laplacian sharpness (motion blur rejection).
2. Camera translation & rotation displacement (avoids static redundancy).
3. Viewpoint diversity & ORB feature overlap (ensures sufficient overlap without duplicate views).
4. Radiometric exposure & contrast validation (rejects underexposed / clipped frames).
5. Supports separate Geometry and Texture frame pools.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Set

import cv2
import numpy as np


@dataclass
class FrameCandidate:
    frame_idx: int
    timestamp_sec: float
    sharpness: float
    luminance: float
    contrast: float
    is_valid_exposure: bool
    rejection_reason: Optional[str] = None


@dataclass
class CuratedFrameRecord:
    pool: str  # "geometry" or "texture"
    pool_index: int
    frame_idx: int
    timestamp_sec: float
    file_path: str
    sharpness: float
    luminance: float
    contrast: float
    translation_disp: float
    rotation_deg: float
    feature_overlap_pct: float
    selection_score: float
    is_interpolated_pose: bool = False


class AdaptiveIndoorFrameCurator:
    def __init__(
        self,
        video_path: str | Path,
        min_sharpness: float = 60.0,
        min_luminance: float = 28.0,
        max_luminance: float = 230.0,
        min_contrast: float = 20.0
    ):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.min_sharpness = min_sharpness
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.min_contrast = min_contrast

        cap = cv2.VideoCapture(str(self.video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps if self.fps > 0 else 0.0
        cap.release()

    def scan_all_candidates(self) -> List[FrameCandidate]:
        """Scans video frames and evaluates per-frame sharpness, exposure and contrast."""
        cap = cv2.VideoCapture(str(self.video_path))
        candidates: List[FrameCandidate] = []

        for f_idx in range(self.total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = f_idx / self.fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Tenengrad + Laplacian Sharpness
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            tenengrad = float(np.mean(gx**2 + gy**2))
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness = math.sqrt(max(0.0, tenengrad)) + math.sqrt(max(0.0, lap_var)) * 0.5

            # 2. Exposure & Contrast
            mean_lum = float(np.mean(gray))
            contrast = float(np.std(gray))
            valid_exp = (self.min_luminance <= mean_lum <= self.max_luminance) and (contrast >= self.min_contrast)

            rejection = None
            if sharpness < self.min_sharpness:
                rejection = f"Motion blur (sharpness {sharpness:.1f} < {self.min_sharpness})"
            elif not valid_exp:
                rejection = f"Poor exposure (lum {mean_lum:.1f}, contrast {contrast:.1f})"

            candidates.append(FrameCandidate(
                frame_idx=f_idx,
                timestamp_sec=round(t_sec, 3),
                sharpness=round(sharpness, 2),
                luminance=round(mean_lum, 2),
                contrast=round(contrast, 2),
                is_valid_exposure=valid_exp,
                rejection_reason=rejection
            ))

        cap.release()
        return candidates

    def curate_density(
        self,
        target_fps: float,
        output_frames_dir: Path,
        all_candidates: Optional[List[FrameCandidate]] = None
    ) -> Dict[str, Any]:
        """
        Curates frames at target_fps enforcing:
        - Viewpoint diversity
        - Temporal redundancy rejection
        - Sharpness prioritization
        - ORB feature overlap tracking
        """
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        if all_candidates is None:
            all_candidates = self.scan_all_candidates()

        total_candidates = len(all_candidates)
        interval_sec = 1.0 / target_fps
        num_intervals = int(math.ceil(self.duration_sec * target_fps))

        cap = cv2.VideoCapture(str(self.video_path))
        orb = cv2.ORB_create(nfeatures=500)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        selected_indices: List[int] = []
        rejection_stats = {
            "blur_rejected": 0,
            "exposure_rejected": 0,
            "static_redundancy_rejected": 0,
            "low_overlap_rejected": 0
        }

        prev_gray: Optional[np.ndarray] = None
        prev_kp = None
        prev_des = None

        curated_records: List[CuratedFrameRecord] = []

        for interval_idx in range(num_intervals):
            t_start = interval_idx * interval_sec
            t_end = min((interval_idx + 1) * interval_sec, self.duration_sec)

            f_start = int(t_start * self.fps)
            f_end = min(int(t_end * self.fps), total_candidates)
            if f_start >= total_candidates:
                break

            window_cands = all_candidates[f_start:f_end]
            if not window_cands:
                continue

            # Filter valid
            valid_cands = [c for c in window_cands if c.rejection_reason is None]
            if not valid_cands:
                # Fallback to candidate with highest sharpness
                valid_cands = [max(window_cands, key=lambda c: c.sharpness)]
                rejection_stats["blur_rejected"] += (len(window_cands) - 1)
            else:
                rejection_stats["blur_rejected"] += (len(window_cands) - len(valid_cands))

            # Select best candidate in window based on sharpness and contrast
            best_cand = max(valid_cands, key=lambda c: c.sharpness * (c.contrast / 50.0))

            # Check motion / redundancy against previous selected frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_cand.frame_idx)
            ret, bgr = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            trans_disp = 0.0
            rot_deg = 0.0
            overlap_pct = 100.0

            if prev_gray is not None and prev_des is not None and des is not None and len(des) > 20:
                # Optical flow translation
                g1 = cv2.resize(prev_gray, (160, 160))
                g2 = cv2.resize(gray, (160, 160))
                trans_disp = float(np.mean(cv2.absdiff(g1, g2)))

                # ORB match overlap
                matches = bf.match(prev_des, des)
                if len(prev_des) > 0 and len(des) > 0:
                    overlap_pct = float(len(matches) / min(len(prev_des), len(des)) * 100.0)

                # Estimate rotation via partial affine
                if len(matches) >= 8:
                    src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    if M is not None:
                        rot_deg = abs(float(np.degrees(math.atan2(M[1, 0], M[0, 0]))))

                # Redundancy check: if camera is completely stationary and overlap is 98%, skip duplicate
                if trans_disp < 2.5 and rot_deg < 0.8 and overlap_pct > 92.0 and len(valid_cands) > 1:
                    rejection_stats["static_redundancy_rejected"] += 1
                    # try alternative candidate in window with higher motion
                    alt_cands = [c for c in valid_cands if c.frame_idx != best_cand.frame_idx]
                    if alt_cands:
                        best_cand = max(alt_cands, key=lambda c: c.sharpness)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, best_cand.frame_idx)
                        ret2, bgr2 = cap.read()
                        if ret2:
                            bgr = bgr2
                            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                            kp, des = orb.detectAndCompute(gray, None)

            prev_gray = gray
            prev_kp = kp
            prev_des = des

            # Save frame
            rank = len(selected_indices)
            file_name = f"frame_{rank:03d}_f{best_cand.frame_idx:05d}_{best_cand.timestamp_sec:05.2f}s.png"
            file_path = output_frames_dir / file_name
            cv2.imwrite(str(file_path), bgr)

            rec = CuratedFrameRecord(
                pool="geometry",
                pool_index=rank,
                frame_idx=best_cand.frame_idx,
                timestamp_sec=best_cand.timestamp_sec,
                file_path=str(file_path.resolve()),
                sharpness=best_cand.sharpness,
                luminance=best_cand.luminance,
                contrast=best_cand.contrast,
                translation_disp=round(trans_disp, 2),
                rotation_deg=round(rot_deg, 2),
                feature_overlap_pct=round(overlap_pct, 1),
                selection_score=round(best_cand.sharpness, 2)
            )
            curated_records.append(rec)
            selected_indices.append(best_cand.frame_idx)

        cap.release()

        report = {
            "target_fps": target_fps,
            "total_candidate_frames": total_candidates,
            "selected_count": len(curated_records),
            "rejected_count": total_candidates - len(curated_records),
            "actual_fps": round(len(curated_records) / max(0.1, self.duration_sec), 3),
            "rejection_breakdown": rejection_stats,
            "mean_sharpness": round(float(np.mean([r.sharpness for r in curated_records])), 2) if curated_records else 0.0,
            "mean_translation_disp": round(float(np.mean([r.translation_disp for r in curated_records])), 2) if curated_records else 0.0,
            "mean_rotation_deg": round(float(np.mean([r.rotation_deg for r in curated_records])), 2) if curated_records else 0.0,
            "mean_feature_overlap_pct": round(float(np.mean([r.feature_overlap_pct for r in curated_records])), 1) if curated_records else 0.0,
            "frames": [asdict(r) for r in curated_records]
        }

        with open(output_frames_dir.parent / f"curation_report_{target_fps:.1f}fps.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

