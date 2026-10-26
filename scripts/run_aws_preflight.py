"""F.O.R.G.E — AWS Video Preflight CLI & Integration Runner.

Interfaces between local reconstruction workflows and the AWS Lambda / S3
Video Preflight Service.
- Online Mode: Invokes the remote AWS Lambda worker against S3 video assets.
- Offline/Local Mode: Runs the exact same OpenCV 5 preflight evaluation engine
  locally on workstation video files to evaluate 3D reconstruction feasibility.

Usage:
    # Run against S3 asset via AWS Lambda
    python scripts/run_aws_preflight.py --s3-key "FORGE/inputs/videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4"

    # Run locally on a video file
    python scripts/run_aws_preflight.py --video "data/source_videos/desert_road.mp4"
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# Import preflight engine from aws/preflight/lambda_function.py
sys.path.insert(0, str(WORKSPACE_ROOT / "aws" / "preflight"))
from lambda_function import run_preflight_analysis, verify_opencv5_runtime


DEFAULT_BUCKET = "forge-drone3d-prathmesh-2026-8xq472"
DEFAULT_REGION = "ap-southeast-2"
DEFAULT_FUNCTION_NAME = "forge-video-preflight-worker"
DEFAULT_KEY = "FORGE/inputs/videos/dji_fly_20260727_172754_0049_1785175051818_video.mp4"


def run_remote_aws_preflight(
    s3_key: str = DEFAULT_KEY,
    bucket: str = DEFAULT_BUCKET,
    region: str = DEFAULT_REGION,
    function_name: str = DEFAULT_FUNCTION_NAME,
    sample_frames: int = 8
) -> Optional[Dict[str, Any]]:
    """Invokes the AWS Lambda preflight worker via boto3."""
    try:
        import boto3
        session = boto3.Session(region_name=region)
        creds = session.get_credentials()
        if not creds:
            print("[AWS] No credentials found in botocore provider chain.")
            return None

        lambda_client = session.client("lambda")
        payload = {
            "bucket": bucket,
            "key": s3_key,
            "sample_frames": sample_frames
        }

        print(f"[AWS] Invoking Lambda function '{function_name}'...")
        print(f"      Target: s3://{bucket}/{s3_key}")
        response = lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="RequestResponse",
            Payload=json.dumps(payload)
        )

        resp_payload = json.loads(response["Payload"].read().decode("utf-8"))
        return resp_payload

    except Exception as e:
        print(f"[AWS] Remote Lambda invocation unavailable: {type(e).__name__} ({e})")
        return None


def run_local_preflight(
    video_path: str,
    output_dir: str = str(WORKSPACE_ROOT / "outputs" / "aws_preflight"),
    sample_frames: int = 8,
    simulated_s3_key: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the OpenCV 5 preflight evaluation engine locally."""
    import cv2
    vpath = Path(video_path)
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    key_repr = simulated_s3_key or f"FORGE/inputs/videos/{vpath.name}"
    stem = vpath.stem

    print(f"\n[F.O.R.G.E Preflight] Running OpenCV 5 Analysis on {vpath.name}...")
    report, sheet = run_preflight_analysis(
        video_path=str(vpath),
        s3_bucket=DEFAULT_BUCKET,
        s3_key=key_repr,
        num_samples=sample_frames
    )

    # Save local report and contact sheet
    report_file = out_p / f"{stem}_preflight_report.json"
    sheet_file = out_p / f"{stem}_contact_sheet.jpg"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved Preflight Report: {report_file}")

    if sheet is not None:
        cv2.imwrite(str(sheet_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"  Saved Contact Sheet:    {sheet_file}")

    return report


def display_preflight_summary(report: Dict[str, Any]) -> None:
    """Formats and prints the preflight quality report."""
    meta = report.get("video_metadata", {})
    metrics = report.get("summary_metrics", {})
    gate = report.get("preflight_decision", {})

    print("\n" + "=" * 65)
    print("F.O.R.G.E. — VIDEO PREFLIGHT QUALITY REPORT")
    print("=" * 65)
    print(f"OpenCV Runtime:     {report.get('opencv_runtime_version', 'Unknown')}")
    print(f"Video Container:    {meta.get('video_filename', 'Unknown')}")
    print(f"Dimensions:         {meta.get('width', 0)}x{meta.get('height', 0)} @ {meta.get('fps', 0)} FPS")
    print(f"Corridor Duration:  {meta.get('duration_sec', 0)}s ({meta.get('total_frames', 0)} frames)")
    print(f"File Size:          {meta.get('file_size_mb', 0)} MB")
    print("-" * 65)
    print("QUANTITATIVE QUALITY METRICS (OpenCV 5.x):")
    print(f"  Mean Laplacian Sharpness: {metrics.get('mean_laplacian_sharpness', 0):.1f}")
    print(f"  Mean Tenengrad Energy:    {metrics.get('mean_tenengrad_sharpness', 0):.1f}")
    print(f"  Mean ORB Keypoints:       {metrics.get('mean_orb_keypoint_count', 0):.1f}")
    print(f"  Mean Dynamic Range:       {metrics.get('mean_dynamic_range', 0):.1f}")
    print(f"  Mean Intensity:           {metrics.get('mean_intensity', 0):.1f}")
    print("-" * 65)
    print(f"PREFLIGHT DECISION:  [{gate.get('status', 'UNKNOWN')}]")
    print(f"Decision Label:      {gate.get('label', '')}")
    print("Evaluator Reasons:")
    for r in gate.get("reasons", []):
        print(f"  * {r}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="F.O.R.G.E AWS Video Preflight Runner")
    parser.add_argument("--video", default=None, help="Local video file to analyze")
    parser.add_argument("--s3-key", default=None, help="S3 object key to process via AWS Lambda")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS Region (default: ap-southeast-2)")
    parser.add_argument("--samples", type=int, default=8, help="Number of corridor frames to sample")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "aws_preflight"), help="Local output directory")
    args = parser.parse_args()

    # Step 1: Attempt remote AWS Lambda invocation if s3-key is provided or no local video is given
    if args.s3_key:
        print("[Mode] AWS Lambda Remote Invocation")
        remote_resp = run_remote_aws_preflight(
            s3_key=args.s3_key,
            bucket=args.bucket,
            region=args.region,
            sample_frames=args.samples
        )
        if remote_resp:
            print("\n[AWS Lambda Response]:")
            print(json.dumps(remote_resp, indent=2))
            return 0

    # Step 2: Fall back to local OpenCV 5 preflight evaluation
    target_video = args.video
    if not target_video:
        # Default to available local video
        candidates = [
            Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4"),
            WORKSPACE_ROOT / "data" / "source_videos" / "desert_road.mp4",
            WORKSPACE_ROOT / "data" / "source_videos" / "forest_road.mp4",
            WORKSPACE_ROOT / "data" / "source_videos" / "dji_fly_20260727_172754_0049_1785175051818_video.mp4",
        ]
        for c in candidates:
            if c.is_file():
                target_video = str(c)
                break

    if not target_video:
        print("Error: No video file found. Please provide --video <path>.")
        sys.exit(1)

    report = run_local_preflight(
        video_path=target_video,
        output_dir=args.output_dir,
        sample_frames=args.samples,
        simulated_s3_key=args.s3_key or DEFAULT_KEY
    )
    display_preflight_summary(report)

    status = report.get("preflight_decision", {}).get("status")
    if status == "APPROVED_FOR_3D_RECONSTRUCTION":
        return 0
    elif status == "NEEDS_REVIEW":
        return 2
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
