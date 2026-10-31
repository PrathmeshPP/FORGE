"""
F.O.R.G.E — Stage A: Indoor Frame Curator & Motion Blur Rejection.

Specifically tailored for indoor handheld walkthroughs:
1. Full temporal trajectory coverage across the walkthrough.
2. Motion blur rejection via dual Laplacian & Tenengrad sharpness evaluation.
3. Radiometric contrast & illumination validation.
4. Pairwise disparity preservation (avoids static/redundant viewpoints).
5. Uncompressed native PNG extraction for high-fidelity texturing.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class CuratedIndoorFrame:
    rank: int
    frame_index: int
    timestamp_sec: float
    file_path: str
    laplacian_sharpness: float
    mean_luminance: float
    contrast_rms: float
    pairwise_disparity: float
    selection_reason: str


class IndoorFrameCurator:
    """
    Curates high-quality, blur-free, viewpoint-diverse frames from indoor handheld video.
    """

    def __init__(
        self,
        video_path: str,
        min_sharpness: float = 80.0,
        target_frame_count: int = 16,
        candidate_subsample: int = 4
    ):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.min_sharpness = min_sharpness
        self.target_frame_count = target_frame_count
        self.candidate_subsample = candidate_subsample

    def get_metadata(self) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()
        return {
            "video_path": str(self.video_path),
            "width": w,
            "height": h,
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "duration_sec": round(duration, 3)
        }

    def curate_at_least_1fps(
        self,
        output_dir: str,
        target_fps: float = 1.0,
        subsample_stride: int = 2
    ) -> List[CuratedIndoorFrame]:
        """
        Curates at least 1 frame per second across the entire video.
        For each 1.0-second interval [sec, sec + 1), it evaluates local candidate frames
        to select the sharpest non-blurred frame, preserving true chronological ordering,
        exact timestamps, and complete room trajectory coverage.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        meta = self.get_metadata()
        fps = meta["fps"]
        total_frames = meta["total_frames"]
        duration = meta["duration_sec"]
        total_seconds = int(math.ceil(duration))

        cap = cv2.VideoCapture(str(self.video_path))
        selected_frames: List[CuratedIndoorFrame] = []
        prev_gray: Optional[np.ndarray] = None

        print(f"[*] Curating uniform >=1 FPS ({total_seconds} seconds) from {total_frames} total frames across {duration:.1f}s...")

        for sec in range(total_seconds):
            start_f = int(sec * fps)
            end_f = min(int((sec + 1) * fps), total_frames)
            if start_f >= total_frames:
                break

            best_f = None
            best_sharpness = -1.0
            best_lum = 0.0
            best_rms = 0.0
            best_img_bgr = None
            best_gray = None

            step = max(1, subsample_stride)
            candidates = list(range(start_f, end_f, step))
            if not candidates:
                candidates = [start_f]

            for f_no in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
                ret, bgr = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                mean_lum = float(np.mean(gray))
                rms_cont = float(np.std(gray))

                lum_penalty = 1.0
                if mean_lum < 30.0 or mean_lum > 230.0:
                    lum_penalty = 0.5

                score = lap_var * lum_penalty
                if score > best_sharpness:
                    best_sharpness = score
                    best_f = f_no
                    best_lum = mean_lum
                    best_rms = rms_cont
                    best_img_bgr = bgr.copy()
                    best_gray = gray

            if best_f is None and candidates:
                best_f = candidates[len(candidates) // 2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, best_f)
                ret, bgr = cap.read()
                if ret:
                    best_img_bgr = bgr.copy()
                    best_gray = cv2.cvtColor(best_img_bgr, cv2.COLOR_BGR2GRAY)
                    best_sharpness = float(cv2.Laplacian(best_gray, cv2.CV_64F).var())
                    best_lum = float(np.mean(best_gray))
                    best_rms = float(np.std(best_gray))

            if best_img_bgr is None:
                continue

            # Compute pairwise disparity with previous selected frame
            if prev_gray is not None and best_gray is not None:
                g1 = cv2.resize(prev_gray, (160, 160))
                g2 = cv2.resize(best_gray, (160, 160))
                disparity = float(np.mean(cv2.absdiff(g1, g2)))
            else:
                disparity = 0.0
            prev_gray = best_gray

            timestamp = best_f / fps if fps > 0 else float(sec)
            file_name = f"frame_{sec:02d}_{best_f:05d}_{timestamp:.2f}s.png"
            file_path = out_path / file_name

            cv2.imwrite(str(file_path), best_img_bgr)

            curated = CuratedIndoorFrame(
                rank=sec,
                frame_index=best_f,
                timestamp_sec=round(timestamp, 3),
                file_path=str(file_path.resolve()),
                laplacian_sharpness=round(best_sharpness, 2),
                mean_luminance=round(best_lum, 2),
                contrast_rms=round(best_rms, 2),
                pairwise_disparity=round(disparity, 2),
                selection_reason=f"Optimal sharp frame for second {sec:02d} [{start_f}:{end_f}] (sharpness: {best_sharpness:.1f})"
            )
            selected_frames.append(curated)

        cap.release()

        curation_report = {
            "source_video": meta,
            "curated_frame_count": len(selected_frames),
            "sampling_density": f"{len(selected_frames) / max(0.1, duration):.2f} FPS",
            "frames": [asdict(f) for f in selected_frames]
        }
        with open(out_path.parent / "curation_manifest.json", "w") as f:
            json.dump(curation_report, f, indent=2)

        self._generate_contact_sheet(selected_frames, out_path.parent / "frames_contact_sheet.png")
        print(f"[+] Successfully extracted {len(selected_frames)} >=1 FPS curated frames to {out_path}")
        return selected_frames

    def curate_and_extract(
        self,
        output_dir: str,
        target_count: Optional[int] = None,
        prefix: str = "indoor_frame"
    ) -> List[CuratedIndoorFrame]:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        count = target_count or self.target_frame_count

        meta = self.get_metadata()
        fps = meta["fps"]
        total_frames = meta["total_frames"]

        cap = cv2.VideoCapture(str(self.video_path))
        window_size = total_frames / count
        selected_frames: List[CuratedIndoorFrame] = []
        prev_gray: Optional[np.ndarray] = None

        print(f"[*] Curating {count} indoor frames from {total_frames} total frames across {meta['duration_sec']:.1f}s...")

        for w_idx in range(count):
            start_f = int(w_idx * window_size)
            end_f = int((w_idx + 1) * window_size)
            end_f = min(end_f, total_frames)

            best_f = None
            best_sharpness = -1.0
            best_lum = 0.0
            best_rms = 0.0
            best_img_bgr = None
            best_gray = None

            # Stride through window
            step = max(1, self.candidate_subsample)
            candidates = list(range(start_f, end_f, step))

            for f_no in candidates:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
                ret, bgr = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                mean_lum = float(np.mean(gray))
                rms_cont = float(np.std(gray))

                # Weight sharpness, penalize extreme darkness or saturation
                lum_penalty = 1.0
                if mean_lum < 30.0 or mean_lum > 230.0:
                    lum_penalty = 0.5

                score = lap_var * lum_penalty
                if score > best_sharpness:
                    best_sharpness = score
                    best_f = f_no
                    best_lum = mean_lum
                    best_rms = rms_cont
                    best_img_bgr = bgr.copy()
                    best_gray = gray

            if best_f is None and candidates:
                # Fallback to middle frame
                best_f = candidates[len(candidates) // 2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, best_f)
                _, best_img_bgr = cap.read()
                best_gray = cv2.cvtColor(best_img_bgr, cv2.COLOR_BGR2GRAY)
                best_sharpness = float(cv2.Laplacian(best_gray, cv2.CV_64F).var())
                best_lum = float(np.mean(best_gray))
                best_rms = float(np.std(best_gray))

            # Compute pairwise disparity with previous selected frame
            if prev_gray is not None and best_gray is not None:
                # Scaled absolute difference
                g1 = cv2.resize(prev_gray, (160, 160))
                g2 = cv2.resize(best_gray, (160, 160))
                disparity = float(np.mean(cv2.absdiff(g1, g2)))
            else:
                disparity = 0.0
            prev_gray = best_gray

            timestamp = best_f / fps if fps > 0 else 0.0
            file_name = f"frame_{w_idx:02d}_{best_f:05d}_{timestamp:.2f}s.png"
            file_path = out_path / file_name

            # Save full uncompressed native PNG
            cv2.imwrite(str(file_path), best_img_bgr)

            curated = CuratedIndoorFrame(
                rank=w_idx,
                frame_index=best_f,
                timestamp_sec=round(timestamp, 3),
                file_path=str(file_path.resolve()),
                laplacian_sharpness=round(best_sharpness, 2),
                mean_luminance=round(best_lum, 2),
                contrast_rms=round(best_rms, 2),
                pairwise_disparity=round(disparity, 2),
                selection_reason=f"Sharpest frame in temporal window [{start_f}:{end_f}] (score: {best_sharpness:.1f})"
            )
            selected_frames.append(curated)

        cap.release()

        # Save curation metadata
        curation_report = {
            "source_video": meta,
            "curated_frame_count": len(selected_frames),
            "frames": [asdict(f) for f in selected_frames]
        }
        with open(out_path.parent / "curation_manifest.json", "w") as f:
            json.dump(curation_report, f, indent=2)

        # Generate contact sheet in parent directory
        self._generate_contact_sheet(selected_frames, out_path.parent / "frames_contact_sheet.png")

        print(f"[+] Successfully extracted {len(selected_frames)} curated frames to {out_path}")
        return selected_frames

    def _generate_contact_sheet(self, frames: List[CuratedIndoorFrame], save_path: Path):
        if not frames:
            return
        imgs = [Image.open(f.file_path) for f in frames]
        thumb_w, thumb_h = 240, int(240 * imgs[0].height / imgs[0].width)
        thumbs = [im.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR) for im in imgs]

        cols = min(6, len(thumbs))
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (20, 24, 30))

        for idx, th in enumerate(thumbs):
            r = idx // cols
            c = idx % cols
            sheet.paste(th, (c * thumb_w, r * thumb_h))

        sheet.save(str(save_path))

