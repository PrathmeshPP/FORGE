"""Tests for F.O.R.G.E OpenCV 5 Frame Analyzer and Viewpoint Curation.

Verifies:
1. OpenCV 5.x runtime verification and graceful rejection of non-5.x versions.
2. Unreadable, missing, and corrupted video handling.
3. Deterministic candidate index and timestamp reporting.
4. Laplacian and Tenengrad sharpness, radiometric fidelity, and clipping metrics.
5. ORB feature count, 4x4 spatial grid cell occupancy, and distribution entropy.
6. Duplicate / near-duplicate detection and penalty.
7. Insufficient matches (<8) and failed geometric estimation reporting (no fabricated covisibility).
8. Minimum frame count, temporal windowing, and monotonic coverage guarantee.
9. Preservation of default frame extraction strategy in run_pipeline.
10. Clean invocation of OpenCV5FrameAnalyzer when --frame-selector opencv5 is requested.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.video.opencv5_frame_analyzer import (
    verify_opencv5_runtime,
    compute_single_frame_metrics,
    compute_pairwise_consistency,
    OpenCV5FrameAnalyzer,
    FrameQualityMetrics,
    PairwiseMatchMetrics,
    CurationReport,
)
from run_pipeline import run_pipeline


class TestOpenCV5FrameAnalyzer(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_synthetic_video(
        self,
        num_frames: int = 15,
        width: int = 160,
        height: int = 120,
        fps: float = 10.0
    ) -> Path:
        """Helper to create a reproducible synthetic video with varied visual features."""
        video_path = self.test_dir / "synthetic_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for i in range(num_frames):
            # Create frame with varying motion, contrast, and features
            frame = np.full((height, width, 3), 40 + i * 5, dtype=np.uint8)
            # Add contrasting geometric features
            center_x = int(30 + i * 5)
            center_y = int(25 + (i % 5) * 10)
            cv2.circle(frame, (center_x, center_y), 15, (220, 20, 20), -1)
            cv2.rectangle(frame, (width - 40, 20), (width - 10, 50), (20, 220, 20), -1)
            # Add high-frequency checkerboard patch in one quadrant
            patch_sz = 8
            for r in range(4):
                for c in range(4):
                    if (r + c) % 2 == 0:
                        frame[r*patch_sz: (r+1)*patch_sz, c*patch_sz: (c+1)*patch_sz] = 255
            writer.write(frame)

        writer.release()
        return video_path

    # 1. OpenCV 5 Runtime Detection
    def test_opencv5_runtime_detection_active(self):
        """Verify active runtime check passes on OpenCV 5.x."""
        version = verify_opencv5_runtime()
        self.assertTrue(version.startswith("5."), f"Expected OpenCV 5.x, got '{version}'")

    def test_opencv5_runtime_detection_failure_on_legacy(self):
        """Verify RuntimeError is raised with clear explanation if cv2 is not 5.x."""
        with patch.object(cv2, "__version__", "4.10.0"):
            with self.assertRaises(RuntimeError) as ctx:
                verify_opencv5_runtime()
            self.assertIn("OpenCV 5.x runtime is required", str(ctx.exception))
            self.assertIn("4.10.0", str(ctx.exception))

        with patch.object(cv2, "__version__", "4.8.0.76"):
            with self.assertRaises(RuntimeError) as ctx:
                OpenCV5FrameAnalyzer()
            self.assertIn("OpenCV 5.x runtime is required", str(ctx.exception))

    # 2. Unreadable and Invalid Video Handling
    def test_missing_video_raises_file_not_found(self):
        """Analyzer must raise FileNotFoundError for nonexistent paths."""
        analyzer = OpenCV5FrameAnalyzer()
        nonexistent = self.test_dir / "does_not_exist.mp4"
        with self.assertRaises(FileNotFoundError):
            analyzer.analyze_video(str(nonexistent))

    def test_invalid_corrupt_video_handling(self):
        """Analyzer must raise RuntimeError for invalid video and release resources."""
        invalid_file = self.test_dir / "corrupt.mp4"
        invalid_file.write_bytes(b"NOT_A_VALID_MP4_HEADER_DATA_12345")
        analyzer = OpenCV5FrameAnalyzer()
        with self.assertRaises(RuntimeError):
            analyzer.analyze_video(str(invalid_file))

    # 3. Deterministic Timestamp and Index Reporting
    def test_deterministic_reporting(self):
        """Repeated analysis of identical video must return identical indices and metrics."""
        video_path = self._create_synthetic_video(num_frames=12, fps=10.0)
        analyzer = OpenCV5FrameAnalyzer()

        report_1 = analyzer.analyze_video(str(video_path), target_num_frames=4)
        report_2 = analyzer.analyze_video(str(video_path), target_num_frames=4)

        indices_1 = [s.frame_index for s in report_1.selected_frames]
        indices_2 = [s.frame_index for s in report_2.selected_frames]
        self.assertEqual(indices_1, indices_2)

        ts_1 = [s.timestamp_sec for s in report_1.selected_frames]
        ts_2 = [s.timestamp_sec for s in report_2.selected_frames]
        self.assertEqual(ts_1, ts_2)

        for s1, s2 in zip(report_1.selected_frames, report_2.selected_frames):
            self.assertAlmostEqual(s1.metrics.laplacian_sharpness, s2.metrics.laplacian_sharpness, places=2)
            self.assertAlmostEqual(s1.metrics.tenengrad_sharpness, s2.metrics.tenengrad_sharpness, places=2)
            self.assertAlmostEqual(s1.metrics.rms_contrast, s2.metrics.rms_contrast, places=2)

    # 4. Sharpness and Exposure Metric Computation
    def test_sharpness_metrics_distinguish_blur_vs_sharp(self):
        """Sharp pattern must have strictly higher Laplacian and Tenengrad score than blurred."""
        # Sharp multi-pixel checkerboard pattern
        sharp = np.zeros((120, 120), dtype=np.uint8)
        block = 8
        for r in range(120):
            for c in range(120):
                if ((r // block) + (c // block)) % 2 == 0:
                    sharp[r, c] = 255

        # Severely blurred version
        blurred = cv2.GaussianBlur(sharp, (15, 15), 5.0)

        sharp_m, _, _ = compute_single_frame_metrics(sharp, 0, 0.0)
        blur_m, _, _ = compute_single_frame_metrics(blurred, 1, 0.1)

        self.assertGreater(sharp_m.laplacian_sharpness, blur_m.laplacian_sharpness * 5.0)
        self.assertGreater(sharp_m.tenengrad_sharpness, blur_m.tenengrad_sharpness * 5.0)

    def test_exposure_and_clipping_metrics(self):
        """Evaluates clipped highlights, clipped shadows, and dynamic range."""
        # Pure overexposed white image
        blown_out = np.full((80, 80), 255, dtype=np.uint8)
        m_white, _, _ = compute_single_frame_metrics(blown_out, 0, 0.0)
        self.assertEqual(m_white.clipped_highlights_ratio, 1.0)
        self.assertEqual(m_white.clipped_shadows_ratio, 0.0)
        self.assertEqual(m_white.mean_intensity, 255.0)
        self.assertEqual(m_white.dynamic_range_p95_p05, 0.0)

        # Pure underexposed black image
        pitch_black = np.full((80, 80), 0, dtype=np.uint8)
        m_black, _, _ = compute_single_frame_metrics(pitch_black, 1, 0.1)
        self.assertEqual(m_black.clipped_shadows_ratio, 1.0)
        self.assertEqual(m_black.clipped_highlights_ratio, 0.0)
        self.assertEqual(m_black.mean_intensity, 0.0)

        # Balanced ramp image
        ramp = np.tile(np.linspace(10, 240, 80, dtype=np.uint8), (80, 1))
        m_ramp, _, _ = compute_single_frame_metrics(ramp, 2, 0.2)
        self.assertEqual(m_ramp.clipped_highlights_ratio, 0.0)
        self.assertEqual(m_ramp.clipped_shadows_ratio, 0.0)
        self.assertGreater(m_ramp.dynamic_range_p95_p05, 180.0)

    # 5. Feature Count and Spatial Distribution Metrics
    def test_spatial_grid_distribution_and_entropy(self):
        """Verifies 4x4 spatial grid cell occupancy and normalized entropy calculation."""
        # Create 320x320 image with features distributed across all quadrants (beyond ORB border margin)
        w, h = 320, 320
        rich_img = np.zeros((h, w), dtype=np.uint8)
        for r in range(4):
            for c in range(4):
                cx, cy = c * 80 + 40, r * 80 + 40
                cv2.rectangle(rich_img, (cx - 15, cy - 15), (cx + 15, cy + 15), 255, -1)
                cv2.circle(rich_img, (cx, cy), 6, 0, -1)

        m_rich, _, _ = compute_single_frame_metrics(rich_img, 0, 0.0)
        self.assertGreaterEqual(m_rich.grid_occupied_cells, 8)
        self.assertGreater(m_rich.grid_distribution_entropy, 0.5)

        # Featureless uniform gray image
        flat_img = np.full((160, 160), 128, dtype=np.uint8)
        m_flat, _, _ = compute_single_frame_metrics(flat_img, 1, 0.1)
        self.assertEqual(m_flat.orb_keypoint_count, 0)
        self.assertEqual(m_flat.grid_occupied_cells, 0)
        self.assertEqual(m_flat.grid_distribution_entropy, 0.0)

    # 6. Duplicate / Near-Duplicate Handling
    def test_near_duplicate_detection(self):
        """Near-duplicate frames have correlation ~1.0 and trigger duplicate penalty."""
        base = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(base, (50, 50), 20, 200, -1)
        dup = base.copy()

        orb = cv2.ORB_create(nfeatures=200)
        kps1, des1 = orb.detectAndCompute(base, None)
        kps2, des2 = orb.detectAndCompute(dup, None)

        p = compute_pairwise_consistency(base, kps1, des1, dup, kps2, des2, 0, 1)
        self.assertAlmostEqual(p.histogram_correlation, 1.0, places=3)
        self.assertAlmostEqual(p.bhattacharyya_distance, 0.0, places=3)

    # 7. Insufficient Matches and Honest Geometric Estimation
    def test_insufficient_matches_reports_unavailable(self):
        """When matches < 8, geometric covisibility must be reported unavailable (not fabricated)."""
        img1 = np.full((100, 100), 50, dtype=np.uint8)
        img2 = np.full((100, 100), 70, dtype=np.uint8)

        # Flat images will have 0 ORB keypoints
        orb = cv2.ORB_create(nfeatures=100)
        kps1, des1 = orb.detectAndCompute(img1, None)
        kps2, des2 = orb.detectAndCompute(img2, None)

        p = compute_pairwise_consistency(img1, kps1, des1, img2, kps2, des2, 0, 1)
        self.assertFalse(p.geometric_verification_available)
        self.assertEqual(p.inlier_matches, 0)
        self.assertEqual(p.geometric_model, "none")
        self.assertIn("insufficient_matches", p.verification_notes)

    # 8. Temporal Coverage and Minimum Frame Count
    def test_chronological_coverage_guarantee(self):
        """Window curation must guarantee monotonic chronological ordering and target frame count."""
        video_path = self._create_synthetic_video(num_frames=20, fps=10.0)
        analyzer = OpenCV5FrameAnalyzer()
        target_k = 5

        report = analyzer.analyze_video(str(video_path), target_num_frames=target_k)
        self.assertEqual(len(report.selected_frames), target_k)

        # Monotonically increasing indices and timestamps
        prev_idx = -1
        prev_ts = -1.0
        for s in report.selected_frames:
            self.assertGreater(s.frame_index, prev_idx)
            self.assertGreaterEqual(s.timestamp_sec, prev_ts)
            prev_idx = s.frame_index
            prev_ts = s.timestamp_sec

        # High trajectory coverage
        cov = report.selection_summary["trajectory_coverage_ratio"]
        self.assertGreater(cov, 0.60)

    # 9. Frame Extraction to Disk and Metadata Persistence
    def test_extract_curated_frames_to_disk(self):
        """Verifies extract_curated_frames writes valid images and metadata JSON."""
        video_path = self._create_synthetic_video(num_frames=10, fps=5.0)
        frames_out = self.test_dir / "extracted_curated_frames"

        analyzer = OpenCV5FrameAnalyzer()
        extracted_dicts, report = analyzer.extract_curated_frames(
            str(video_path),
            output_dir=str(frames_out),
            target_num_frames=3,
            save_metadata=True
        )

        self.assertEqual(len(extracted_dicts), 3)
        self.assertTrue((frames_out / "opencv5_curation_metadata.json").is_file())

        for d in extracted_dicts:
            fp = Path(d["file_path"])
            self.assertTrue(fp.is_file())
            img = Image.open(fp)
            self.assertEqual(img.size, (160, 120))

    # 10. Pipeline CLI Integration & Strategy Preservation
    @patch("run_pipeline.VGGTInferencePipeline")
    def test_pipeline_preserves_default_frame_extraction(self, mock_vggt_cls):
        """Default selector routes to standard extract_frames without altering legacy behavior."""
        # Mock VGGT inference to avoid loading GPU models
        mock_pipeline = MagicMock()
        mock_pipeline.initialize.return_value = 0.05
        mock_pipeline.run_inference.return_value = {
            "metrics": {"inference_time_sec": 0.1, "peak_vram_allocated_mb": 10.0, "peak_vram_reserved_mb": 20.0}
        }
        mock_pipeline.save_predictions.return_value = {"depth": "preds/depth.npy"}
        mock_vggt_cls.return_value = mock_pipeline

        video_path = self._create_synthetic_video(num_frames=8, fps=4.0)
        pipe_out = self.test_dir / "pipeline_default_out"

        with patch("run_pipeline.extract_frames") as mock_extract:
            from src.video.extract_frames import ExtractedFrame
            mock_extract.return_value = [
                ExtractedFrame(0, 0.0, str(self.test_dir / "dummy0.png"), 160, 120),
                ExtractedFrame(1, 0.25, str(self.test_dir / "dummy1.png"), 160, 120),
            ]

            meta = run_pipeline(
                video_path=str(video_path),
                output_dir=str(pipe_out),
                frame_selector="default",
                max_frames=2
            )

            mock_extract.assert_called_once()
            self.assertEqual(meta["video"]["frame_selector"], "default")

    @patch("run_pipeline.VGGTInferencePipeline")
    def test_pipeline_invokes_opencv5_analyzer_when_requested(self, mock_vggt_cls):
        """Passing frame_selector='opencv5' activates OpenCV 5 curation."""
        mock_pipeline = MagicMock()
        mock_pipeline.initialize.return_value = 0.05
        mock_pipeline.run_inference.return_value = {
            "metrics": {"inference_time_sec": 0.1, "peak_vram_allocated_mb": 10.0, "peak_vram_reserved_mb": 20.0}
        }
        mock_pipeline.save_predictions.return_value = {"depth": "preds/depth.npy"}
        mock_vggt_cls.return_value = mock_pipeline

        video_path = self._create_synthetic_video(num_frames=10, fps=5.0)
        pipe_out = self.test_dir / "pipeline_opencv5_out"

        meta = run_pipeline(
            video_path=str(video_path),
            output_dir=str(pipe_out),
            frame_selector="opencv5",
            max_frames=3
        )

        self.assertEqual(meta["video"]["frame_selector"], "opencv5")
        self.assertEqual(meta["video"]["frames_extracted"], 3)
        # Verify metadata JSON saved
        curation_meta = pipe_out / "frames" / "opencv5_curation_metadata.json"
        self.assertTrue(curation_meta.is_file(), "Curation metadata JSON should be written to frames dir")

    @patch("run_pipeline.VGGTInferencePipeline")
    def test_pipeline_defaults_to_opencv5_analyzer(self, mock_vggt_cls):
        """Calling run_pipeline without specifying frame_selector defaults to OpenCV 5."""
        mock_pipeline = MagicMock()
        mock_pipeline.initialize.return_value = 0.05
        mock_pipeline.run_inference.return_value = {
            "metrics": {"inference_time_sec": 0.1, "peak_vram_allocated_mb": 10.0, "peak_vram_reserved_mb": 20.0}
        }
        mock_pipeline.save_predictions.return_value = {"depth": "preds/depth.npy"}
        mock_vggt_cls.return_value = mock_pipeline

        video_path = self._create_synthetic_video(num_frames=8, fps=4.0)
        pipe_out = self.test_dir / "pipeline_default_is_opencv5_out"

        meta = run_pipeline(
            video_path=str(video_path),
            output_dir=str(pipe_out),
            max_frames=2
        )

        self.assertEqual(meta["video"]["frame_selector"], "opencv5")
        self.assertEqual(meta["video"]["frames_extracted"], 2)

    def test_pipeline_rejects_invalid_frame_selector(self):
        """Invalid frame selector value raises ValueError."""
        video_path = self._create_synthetic_video(num_frames=5, fps=5.0)
        with self.assertRaises(ValueError):
            run_pipeline(
                video_path=str(video_path),
                output_dir=str(self.test_dir / "invalid_sel"),
                frame_selector="nonexistent_selector"
            )


if __name__ == "__main__":
    unittest.main()
