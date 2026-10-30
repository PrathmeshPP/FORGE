"""
F.O.R.G.E. Website Backend — Video Validator Service
Validates video containers, decodability, and frame properties using OpenCV.
Does NOT rely on browser MIME type or file extension alone.
"""

import math
from pathlib import Path
from typing import Tuple, List, Optional
import cv2

from web_backend.config import FORGE_ACCEPTED_EXTENSIONS, FORGE_MAX_VIDEO_SECONDS
from web_backend.models import PreflightMetadata


class VideoValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_and_probe_video(
    file_path: Path,
    original_filename: str,
    file_size_bytes: int
) -> PreflightMetadata:
    """
    Substantively inspects the uploaded file using OpenCV.
    Ensures container is decodable, frames can be read, and duration/resolution are valid.
    """
    # 1. Extension Allowlist Check
    ext = Path(original_filename).suffix.lower()
    if ext not in FORGE_ACCEPTED_EXTENSIONS:
        raise VideoValidationError(
            "UNSUPPORTED_VIDEO",
            f"Unsupported container extension '{ext}'. Accepted containers: {', '.join(FORGE_ACCEPTED_EXTENSIONS)}."
        )

    if file_size_bytes <= 0:
        raise VideoValidationError("INVALID_VIDEO", "Uploaded file is zero bytes / empty.")

    # 2. OpenCV Container and Stream Probe
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        raise VideoValidationError(
            "INVALID_VIDEO",
            "Video file could not be decoded. The file may be corrupt, truncated, or not a valid video stream."
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 3. Read at least one frame to confirm actual decodable pixel data
        ret, first_frame = cap.read()
        if not ret or first_frame is None or first_frame.size == 0:
            raise VideoValidationError(
                "INVALID_VIDEO",
                "Video stream contains no decodable frames. File appears to be truncated or invalid."
            )

        # Compute duration
        if fps > 0 and frame_count > 0:
            duration_seconds = frame_count / fps
        else:
            duration_seconds = 0.0

        if width <= 0 or height <= 0:
            raise VideoValidationError(
                "INVALID_VIDEO",
                f"Invalid video spatial dimensions ({width}x{height})."
            )

        rejection_reasons: List[str] = []
        warnings: List[str] = []

        # Duration Constraints
        if duration_seconds > FORGE_MAX_VIDEO_SECONDS:
            rejection_reasons.append(
                f"Video duration ({duration_seconds:.1f}s) exceeds maximum configured limit of {FORGE_MAX_VIDEO_SECONDS:.0f}s."
            )
        elif duration_seconds < 3.0 and duration_seconds > 0:
            rejection_reasons.append("Video duration is under 3.0 seconds; insufficient temporal coverage.")

        # Resolution Constraints
        if width < 640 or height < 480:
            rejection_reasons.append(f"Resolution {width}x{height} is too low for geometric feature extraction.")
        elif width < 1280 or height < 720:
            warnings.append("Resolution is below 720p HD. Spatial reconstruction accuracy may be degraded.")

        def gcd(a: int, b: int) -> int:
            return a if b == 0 else gcd(b, a % b)

        div = gcd(width, height) if gcd(width, height) > 0 else 1
        aspect = f"{width // div}:{height // div}"
        file_size_mb = file_size_bytes / (1024 * 1024)

        status = "rejected" if rejection_reasons else ("warning" if warnings else "passed")

        if rejection_reasons:
            raise VideoValidationError("INVALID_VIDEO", "; ".join(rejection_reasons))

        return PreflightMetadata(
            filename=original_filename,
            file_size_bytes=file_size_bytes,
            file_size_mb=round(file_size_mb, 2),
            mime_type="video/mp4" if ext == ".mp4" else "video/quicktime",
            container_valid=True,
            duration_seconds=round(duration_seconds, 2),
            width=width,
            height=height,
            aspect_ratio=aspect,
            framerate_fps=round(fps, 2) if fps > 0 else None,
            estimated_frames=frame_count if frame_count > 0 else None,
            status=status,
            rejection_reasons=rejection_reasons,
            warnings=warnings
        )
    finally:
        cap.release()

