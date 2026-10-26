"""Tests for F.O.R.G.E. AWS Lambda Video Preflight Service.

Validates:
1. Strict OpenCV 5.x runtime verification.
2. Single-frame quantitative metric extraction (Laplacian sharpness, Tenengrad, ORB).
3. 3D reconstruction feasibility decision gate logic.
4. Visual contact sheet mosaic generation and dimensions.
5. AWS Lambda handler event parsing and S3 mock upload roundtrip.
6. Real video preflight execution and report schema compliance.
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

sys.path.insert(0, str(WORKSPACE_ROOT / "aws" / "preflight"))
from lambda_function import (
    verify_opencv5_runtime,
    analyze_frame,
    generate_contact_sheet,
    evaluate_decision_gate,
    run_preflight_analysis,
    lambda_handler,
    DEFAULT_BUCKET,
    OUTPUT_PREFIX,
)


class TestAWSPreflightService(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="forge_aws_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_video(self, num_frames: int = 35, width: int = 1280, height: int = 720, fps: float = 30.0) -> Path:
        vid_p = Path(self.temp_dir) / "synthetic_drone_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vid_p), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.full((height, width, 3), 100 + (i % 30), dtype=np.uint8)
            # Add strong features
            cv2.circle(frame, (width // 2, height // 2), 60, (20, 220, 20), -1)
            cv2.rectangle(frame, (100, 100), (300, 300), (220, 20, 20), -1)
            # Add high frequency noise patch
            noise = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            frame[400:500, 400:500] = noise
            writer.write(frame)

        writer.release()
        return vid_p

    def test_01_opencv5_runtime_guard(self):
        """Verify that OpenCV runtime is 5.x and rejects legacy versions."""
        v = verify_opencv5_runtime()
        self.assertTrue(v.startswith("5."), f"Expected OpenCV 5.x, got '{v}'")

        with patch.object(cv2, "__version__", "4.10.0"):
            with self.assertRaises(RuntimeError):
                verify_opencv5_runtime()

    def test_02_single_frame_analysis(self):
        """Verify quantitative frame metrics extraction."""
        test_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        # Add high contrast edges
        cv2.rectangle(test_frame, (100, 100), (500, 500), (255, 255, 255), -1)
        cv2.circle(test_frame, (800, 400), 100, (0, 0, 0), -1)

        metrics = analyze_frame(test_frame, frame_idx=0, timestamp_sec=0.0)
        self.assertEqual(metrics["frame_index"], 0)
        self.assertEqual(metrics["timestamp_sec"], 0.0)
        self.assertGreater(metrics["laplacian_sharpness"], 0.0)
        self.assertGreater(metrics["tenengrad_sharpness"], 0.0)
        self.assertGreater(metrics["orb_keypoint_count"], 0)
        self.assertGreater(metrics["dynamic_range"], 0.0)

    def test_03_decision_gate_evaluation(self):
        """Verify decision gate states for compliant and non-compliant inputs."""
        # Case A: Compliant Video
        meta_pass = {"width": 1920, "height": 1080, "total_frames": 100}
        metrics_pass = [
            {"laplacian_sharpness": 50.0, "orb_keypoint_count": 300, "dynamic_range": 120.0}
        ]
        decision, label, reasons = evaluate_decision_gate(meta_pass, metrics_pass)
        self.assertEqual(decision, "APPROVED_FOR_3D_RECONSTRUCTION")

        # Case B: Low Resolution (Rejected)
        meta_low_res = {"width": 640, "height": 480, "total_frames": 100}
        decision, label, reasons = evaluate_decision_gate(meta_low_res, metrics_pass)
        self.assertEqual(decision, "REJECTED")

        # Case C: Insufficient Frames (Rejected)
        meta_short = {"width": 1920, "height": 1080, "total_frames": 15}
        decision, label, reasons = evaluate_decision_gate(meta_short, metrics_pass)
        self.assertEqual(decision, "REJECTED")

        # Case D: Blurry Video (Needs Review)
        metrics_blur = [
            {"laplacian_sharpness": 5.0, "orb_keypoint_count": 200, "dynamic_range": 100.0}
        ]
        decision, label, reasons = evaluate_decision_gate(meta_pass, metrics_blur)
        self.assertEqual(decision, "NEEDS_REVIEW")

    def test_04_contact_sheet_generation(self):
        """Verify contact sheet mosaic composition and dimensions."""
        frame = np.full((180, 320, 3), 150, dtype=np.uint8)
        metrics = {"frame_index": 1, "timestamp_sec": 0.5, "laplacian_sharpness": 30.0, "orb_keypoint_count": 150}
        samples = [(frame, metrics)] * 8

        sheet = generate_contact_sheet(samples, thumb_w=320, thumb_h=180, cols=4)
        self.assertEqual(sheet.shape, (360, 1280, 3))  # 2 rows x 4 cols

    def test_05_lambda_handler_with_mocked_s3(self):
        """Verify full Lambda handler workflow with mocked S3 client."""
        vid_p = self._create_synthetic_video(num_frames=35, width=1280, height=720)

        mock_s3 = MagicMock()
        # Mock download_file to copy our synthetic video
        def fake_download(bucket, key, local_path):
            shutil.copy(vid_p, local_path)
        mock_s3.download_file.side_effect = fake_download

        event = {
            "bucket": DEFAULT_BUCKET,
            "key": "FORGE/inputs/videos/test_drone_flight.mp4",
            "sample_frames": 4
        }

        with patch("boto3.client", return_value=mock_s3):
            resp = lambda_handler(event, None)

        self.assertEqual(resp["statusCode"], 200)
        body = resp["body"]
        self.assertEqual(body["status"], "SUCCESS")
        self.assertIn("decision", body)
        self.assertTrue(body["opencv_version"].startswith("5."))
        self.assertIn("FORGE/outputs/aws_preflight/test_drone_flight_preflight_report.json", body["report_s3_key"])
        self.assertIn("FORGE/outputs/aws_preflight/test_drone_flight_contact_sheet.jpg", body["contact_sheet_s3_key"])

        # Verify S3 upload_file was called for both report and contact sheet
        self.assertEqual(mock_s3.upload_file.call_count, 2)

    def test_06_preflight_on_real_drone_video(self):
        """Verify preflight analysis on an available real video."""
        real_video = WORKSPACE_ROOT / "data" / "source_videos" / "desert_road.mp4"
        if not real_video.is_file():
            self.skipTest("desert_road.mp4 not found for testing.")

        report, sheet = run_preflight_analysis(
            video_path=str(real_video),
            s3_bucket=DEFAULT_BUCKET,
            s3_key="FORGE/inputs/videos/desert_road.mp4",
            num_samples=4
        )

        self.assertEqual(report["service"], "F.O.R.G.E. AWS Video Preflight Worker")
        self.assertTrue(report["opencv_runtime_version"].startswith("5."))
        self.assertEqual(report["video_metadata"]["width"], 1280)
        self.assertEqual(report["video_metadata"]["height"], 720)
        self.assertEqual(len(report["per_frame_metrics"]), 4)
        self.assertIn("preflight_decision", report)
        self.assertIsNotNone(sheet)


if __name__ == "__main__":
    unittest.main()
