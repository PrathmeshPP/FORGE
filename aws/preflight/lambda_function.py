"""F.O.R.G.E — AWS Lambda Video Preflight Quality Service.

Event-driven AWS Lambda worker using OpenCV 5.x and Amazon S3 to perform
substantive, automated quality inspection and viewpoint feasibility analysis
on uploaded drone video sequences prior to foundation model 3D reconstruction.

Architecture:
    [Drone Video Upload to S3]
             │ (s3:ObjectCreated:Put to FORGE/inputs/videos/*.mp4)
             ▼
    [AWS Lambda Preflight Worker (OpenCV 5.x)]
      ├── 1. Probes container metadata (dimensions, FPS, frame count, duration)
      ├── 2. Uniformly samples temporal corridor frames
      ├── 3. Computes OpenCV 5 image metrics (Laplacian sharpness, Tenengrad, ORB density)
      ├── 4. Evaluates 3D reconstruction feasibility decision gate
      ├── 5. Generates visual contact sheet mosaic (JPEG)
      └── 6. Persists structured JSON report to FORGE/outputs/aws_preflight/
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import tempfile
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import boto3

# Global configuration defaults (overridable via Lambda environment variables)
DEFAULT_BUCKET = os.environ.get("BUCKET_NAME", "forge-drone3d-prathmesh-2026-8xq472")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "FORGE/outputs/aws_preflight/")
INPUT_PREFIX = os.environ.get("INPUT_PREFIX", "FORGE/inputs/videos/")
DEFAULT_SAMPLE_FRAMES = int(os.environ.get("SAMPLE_FRAMES", "8"))
MIN_LAPLACIAN_SHARPNESS = float(os.environ.get("MIN_LAPLACIAN_SHARPNESS", "15.0"))
MIN_ORB_KEYPOINTS = int(os.environ.get("MIN_ORB_KEYPOINTS", "100"))
MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "1280"))
MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "720"))
MIN_FRAMES = int(os.environ.get("MIN_FRAMES", "30"))


def verify_opencv5_runtime() -> str:
    """Verifies that the active OpenCV runtime is strictly version 5.x."""
    version = getattr(cv2, "__version__", "")
    if not version.startswith("5."):
        raise RuntimeError(
            f"OpenCV 5.x runtime is required for F.O.R.G.E. preflight, but active cv2.__version__ is: '{version}'"
        )
    return version


def analyze_frame(
    frame_bgr: np.ndarray,
    frame_idx: int,
    timestamp_sec: float,
    orb: Optional[cv2.ORB] = None
) -> Dict[str, Any]:
    """Computes quantitative quality and feature metrics for a single frame."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Laplacian Sharpness (high-frequency variance)
    lap_sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 2. Tenengrad Gradient Energy (Sobel operators)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x ** 2 + sobel_y ** 2))

    # 3. Invariant Feature Density (ORB)
    if orb is None:
        orb = cv2.ORB_create(nfeatures=500)
    kps = orb.detect(gray, None)
    orb_count = len(kps)

    # 4. Radiometric Metrics
    mean_val = float(np.mean(gray))
    rms_contrast = float(np.std(gray))
    p05, p95 = np.percentile(gray, [5, 95])
    dynamic_range = float(p95 - p05)
    clipped_highlights = float(np.mean(gray >= 250))
    clipped_shadows = float(np.mean(gray <= 5))

    return {
        "frame_index": frame_idx,
        "timestamp_sec": round(timestamp_sec, 3),
        "laplacian_sharpness": round(lap_sharp, 2),
        "tenengrad_sharpness": round(tenengrad, 2),
        "orb_keypoint_count": orb_count,
        "mean_intensity": round(mean_val, 2),
        "rms_contrast": round(rms_contrast, 2),
        "dynamic_range": round(dynamic_range, 2),
        "clipped_highlights_ratio": round(clipped_highlights, 4),
        "clipped_shadows_ratio": round(clipped_shadows, 4),
    }


def generate_contact_sheet(
    sampled_frames: List[Tuple[np.ndarray, Dict[str, Any]]],
    thumb_w: int = 320,
    thumb_h: int = 180,
    cols: int = 4
) -> np.ndarray:
    """Assembles a labeled contact sheet mosaic from sampled video frames."""
    n = len(sampled_frames)
    if n == 0:
        return np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)

    rows = math.ceil(n / cols)
    sheet_h = rows * thumb_h
    sheet_w = cols * thumb_w
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, (frame, metrics) in enumerate(sampled_frames):
        r = i // cols
        c = i % cols
        resized = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # Annotate with frame info and sharpness
        label1 = f"F#{metrics['frame_index']} | {metrics['timestamp_sec']:.1f}s"
        label2 = f"Lap:{metrics['laplacian_sharpness']:.0f} | ORB:{metrics['orb_keypoint_count']}"

        cv2.rectangle(resized, (0, thumb_h - 40), (thumb_w, thumb_h), (0, 0, 0), -1)
        cv2.putText(resized, label1, (8, thumb_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(resized, label2, (8, thumb_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = resized

    return sheet


def evaluate_decision_gate(
    video_meta: Dict[str, Any],
    frame_metrics: List[Dict[str, Any]]
) -> Tuple[str, str, List[str]]:
    """Evaluates video feasibility for 3D reconstruction based on quantitative thresholds."""
    reasons: List[str] = []
    flags: List[str] = []

    mean_lap = float(np.mean([m["laplacian_sharpness"] for m in frame_metrics])) if frame_metrics else 0.0
    mean_orb = float(np.mean([m["orb_keypoint_count"] for m in frame_metrics])) if frame_metrics else 0.0
    mean_dr = float(np.mean([m["dynamic_range"] for m in frame_metrics])) if frame_metrics else 0.0

    # Dimension checks
    if video_meta["width"] < MIN_WIDTH or video_meta["height"] < MIN_HEIGHT:
        reasons.append(f"Resolution {video_meta['width']}x{video_meta['height']} below minimum {MIN_WIDTH}x{MIN_HEIGHT}")
        flags.append("LOW_RESOLUTION")
    else:
        reasons.append(f"Resolution {video_meta['width']}x{video_meta['height']} meets criteria")

    # Frame count check
    if video_meta["total_frames"] < MIN_FRAMES:
        reasons.append(f"Sequence has only {video_meta['total_frames']} frames; minimum is {MIN_FRAMES}")
        flags.append("INSUFFICIENT_FRAMES")

    # Sharpness check
    if mean_lap < MIN_LAPLACIAN_SHARPNESS:
        reasons.append(f"Mean Laplacian sharpness {mean_lap:.1f} below threshold {MIN_LAPLACIAN_SHARPNESS}")
        flags.append("LOW_SHARPNESS")
    else:
        reasons.append(f"Mean Laplacian sharpness {mean_lap:.1f} indicates adequate focus")

    # Feature richness check
    if mean_orb < MIN_ORB_KEYPOINTS:
        reasons.append(f"Mean ORB keypoints {mean_orb:.1f} below threshold {MIN_ORB_KEYPOINTS}")
        flags.append("LOW_FEATURE_DENSITY")
    else:
        reasons.append(f"Mean ORB keypoints {mean_orb:.1f} confirms textural richness")

    # Final Decision
    if not flags:
        decision = "APPROVED_FOR_3D_RECONSTRUCTION"
        label = "Pass: Video meets all geospatial 3D reconstruction quality gates"
    elif "LOW_RESOLUTION" in flags or "INSUFFICIENT_FRAMES" in flags:
        decision = "REJECTED"
        label = f"Fail: Video fails structural criteria ({', '.join(flags)})"
    else:
        decision = "NEEDS_REVIEW"
        label = f"Needs Review: Moderate degradation detected ({', '.join(flags)})"

    return decision, label, reasons


def run_preflight_analysis(
    video_path: str,
    s3_bucket: str,
    s3_key: str,
    num_samples: int = DEFAULT_SAMPLE_FRAMES,
    s3_client: Optional[Any] = None
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Executes the complete video inspection and quality gate analysis."""
    t0 = time.perf_counter()
    opencv_ver = verify_opencv5_runtime()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV {opencv_ver} could not open video container: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        file_size = os.path.getsize(video_path)

        if total_frames <= 0:
            raise RuntimeError(f"Video container reports 0 total frames: {video_path}")

        video_meta = {
            "source_s3_bucket": s3_bucket,
            "source_s3_key": s3_key,
            "video_filename": Path(s3_key).name,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
        }

        # Calculate bounded sampling indices uniformly across video duration
        sample_count = min(num_samples, total_frames)
        if sample_count == 1:
            sample_indices = [total_frames // 2]
        else:
            sample_indices = [int(round(i * (total_frames - 1) / (sample_count - 1))) for i in range(sample_count)]

        orb = cv2.ORB_create(nfeatures=500)
        sampled_records: List[Dict[str, Any]] = []
        sampled_frames_with_meta: List[Tuple[np.ndarray, Dict[str, Any]]] = []

        for f_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            t_sec = f_idx / fps
            metrics = analyze_frame(frame, f_idx, t_sec, orb=orb)
            sampled_records.append(metrics)
            sampled_frames_with_meta.append((frame, metrics))

        if not sampled_records:
            raise RuntimeError("Failed to decode any frames from video")

        # Evaluate quality gate
        decision, decision_label, reasons = evaluate_decision_gate(video_meta, sampled_records)

        # Contact sheet
        contact_sheet = generate_contact_sheet(sampled_frames_with_meta)

        elapsed_sec = round(time.perf_counter() - t0, 3)

        report = {
            "service": "F.O.R.G.E. AWS Video Preflight Worker",
            "version": "1.0.0",
            "opencv_runtime_version": opencv_ver,
            "python_version": sys.version.split()[0],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "analysis_time_sec": elapsed_sec,
            "video_metadata": video_meta,
            "sampling": {
                "requested_samples": num_samples,
                "evaluated_samples": len(sampled_records),
                "sample_frame_indices": sample_indices,
            },
            "summary_metrics": {
                "mean_laplacian_sharpness": round(float(np.mean([m["laplacian_sharpness"] for m in sampled_records])), 2),
                "mean_tenengrad_sharpness": round(float(np.mean([m["tenengrad_sharpness"] for m in sampled_records])), 2),
                "mean_orb_keypoint_count": round(float(np.mean([m["orb_keypoint_count"] for m in sampled_records])), 1),
                "mean_dynamic_range": round(float(np.mean([m["dynamic_range"] for m in sampled_records])), 2),
                "mean_intensity": round(float(np.mean([m["mean_intensity"] for m in sampled_records])), 2),
            },
            "preflight_decision": {
                "status": decision,
                "label": decision_label,
                "reasons": reasons,
                "approved_for_curation": (decision == "APPROVED_FOR_3D_RECONSTRUCTION"),
            },
            "per_frame_metrics": sampled_records,
        }

        return report, contact_sheet

    finally:
        cap.release()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point.
    
    Accepts:
      1. S3 Event Notification ({'Records': [{'s3': {'bucket': {'name': ...}, 'object': {'key': ...}}}]})
      2. Direct test payload ({'bucket': '...', 'key': 'FORGE/inputs/videos/...', 'sample_frames': 8})
    """
    print("F.O.R.G.E. AWS Video Preflight Worker started.")
    s3_client = boto3.client("s3")

    # 1. Parse Event
    bucket = DEFAULT_BUCKET
    key = None
    sample_frames = DEFAULT_SAMPLE_FRAMES

    if "Records" in event and len(event["Records"]) > 0:
        record = event["Records"][0]
        s3_data = record.get("s3", {})
        bucket = s3_data.get("bucket", {}).get("name", DEFAULT_BUCKET)
        raw_key = s3_data.get("object", {}).get("key", "")
        key = urllib.parse.unquote_plus(raw_key)
    else:
        bucket = event.get("bucket", DEFAULT_BUCKET)
        key = event.get("key")
        sample_frames = int(event.get("sample_frames", DEFAULT_SAMPLE_FRAMES))

    if not key:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "No S3 object key provided in event."})
        }

    # Security check: Ensure key is within the expected input prefix and is an MP4
    if not key.endswith(".mp4") and not key.endswith(".mov"):
        print(f"Skipping non-video key: {key}")
        return {
            "statusCode": 200,
            "body": json.dumps({"message": f"Skipped non-video object: {key}"})
        }

    stem = Path(key).stem
    report_key = f"{OUTPUT_PREFIX}{stem}_preflight_report.json"
    contact_sheet_key = f"{OUTPUT_PREFIX}{stem}_contact_sheet.jpg"

    print(f"Processing S3 Video: s3://{bucket}/{key}")
    print(f"Target Report:      s3://{bucket}/{report_key}")

    # 2. Download video to ephemeral storage (/tmp)
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
        local_video_path = os.path.join(tmp_dir, Path(key).name)
        local_report_path = os.path.join(tmp_dir, "preflight_report.json")
        local_sheet_path = os.path.join(tmp_dir, "contact_sheet.jpg")

        print("Downloading video from S3...")
        s3_client.download_file(bucket, key, local_video_path)
        print(f"Downloaded {os.path.getsize(local_video_path)} bytes to {local_video_path}")

        # 3. Execute OpenCV 5 Preflight Analysis
        report, contact_sheet = run_preflight_analysis(
            video_path=local_video_path,
            s3_bucket=bucket,
            s3_key=key,
            num_samples=sample_frames,
            s3_client=s3_client
        )

        # 4. Upload Report and Contact Sheet to S3
        with open(local_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        s3_client.upload_file(
            local_report_path,
            bucket,
            report_key,
            ExtraArgs={"ContentType": "application/json"}
        )
        print(f"Uploaded preflight report to s3://{bucket}/{report_key}")

        if contact_sheet is not None:
            cv2.imwrite(local_sheet_path, contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            s3_client.upload_file(
                local_sheet_path,
                bucket,
                contact_sheet_key,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
            print(f"Uploaded contact sheet to s3://{bucket}/{contact_sheet_key}")

    print("Preflight analysis completed successfully.")
    return {
        "statusCode": 200,
        "body": {
            "status": "SUCCESS",
            "decision": report["preflight_decision"]["status"],
            "opencv_version": report["opencv_runtime_version"],
            "video_key": key,
            "report_s3_key": report_key,
            "contact_sheet_s3_key": contact_sheet_key,
            "summary_metrics": report["summary_metrics"],
        }
    }


if __name__ == "__main__":
    # Local CLI test harness
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
        print(f"Running local test on: {test_video}")
        rep, sheet = run_preflight_analysis(test_video, "local-bucket", "local/video.mp4", num_samples=6)
        print(json.dumps(rep, indent=2))
        if sheet is not None:
            cv2.imwrite("preflight_contact_sheet.jpg", sheet)
            print("Wrote preflight_contact_sheet.jpg")
    else:
        print("Usage: python lambda_function.py <local_video_path>")
