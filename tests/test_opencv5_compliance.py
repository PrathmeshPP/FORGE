"""F.O.R.G.E — OpenCV 5 Compliance & Verification Smoke Test.

Executable test suite for OpenCV AI Competition 2026 compliance.
Verifies:
1. Active cv2 runtime is strictly OpenCV 5.x.
2. Real drone video can be opened via cv2.VideoCapture and decoded.
3. Substantive F.O.R.G.E. OpenCV 5 preprocessing and viewpoint curation runs end-to-end.
4. Clean cleanup of all temporary test artifacts.

Usage:
    python tests/test_opencv5_compliance.py
    python -m unittest tests/test_opencv5_compliance.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import argparse
import unittest
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.video.opencv5_frame_analyzer import (
    verify_opencv5_runtime,
    OpenCV5FrameAnalyzer,
    FrameQualityMetrics,
    compute_single_frame_metrics,
)
from src.video.extract_frames import extract_frames


# Candidate real video locations in order of preference
CANDIDATE_VIDEOS: List[Path] = [
    Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4"),
    WORKSPACE_ROOT / "data" / "source_videos" / "desert_road.mp4",
    WORKSPACE_ROOT / "data" / "source_videos" / "forest_road.mp4",
    WORKSPACE_ROOT / "data" / "source_videos" / "dji_fly_20260727_172754_0049_1785175051818_video.mp4",
]


def resolve_test_video(explicit_path: Optional[str] = None) -> Path:
    """Resolves an existing drone video file for testing."""
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Explicitly specified test video does not exist: {explicit_path}")

    env_path = os.environ.get("FORGE_TEST_VIDEO")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p

    for candidate in CANDIDATE_VIDEOS:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No available test video found. Provide a valid video path via --video or set FORGE_TEST_VIDEO."
    )


class TestOpenCV5Compliance(unittest.TestCase):
    """Smoke test asserting OpenCV 5 runtime compliance and video pipeline execution."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.video_path = resolve_test_video()
        except FileNotFoundError:
            cls.video_path = None

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="forge_cv5_test_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_opencv5_runtime_version(self):
        """Verify that cv2.__version__ is installed and begins with '5.'"""
        version = cv2.__version__
        print(f"\n[Test 1] OpenCV Runtime Version: {version}")
        self.assertTrue(
            version.startswith("5."),
            f"Expected OpenCV 5.x runtime, but found: '{version}'. Submission requires OpenCV 5."
        )
        verified_version = verify_opencv5_runtime()
        self.assertEqual(version, verified_version)

    def test_02_video_capture_and_decoding(self):
        """Open a real drone video, verify container metadata, and decode frames."""
        if self.video_path is None:
            self.skipTest("No drone video available for testing.")

        print(f"\n[Test 2] Opening Video: {self.video_path.name}")
        cap = cv2.VideoCapture(str(self.video_path))
        self.assertTrue(cap.isOpened(), f"cv2.VideoCapture failed to open: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"         Dimensions: {width}x{height}")
        print(f"         FPS:        {fps:.2f}")
        print(f"         Frames:     {frame_count}")

        self.assertGreater(width, 0, "Invalid frame width")
        self.assertGreater(height, 0, "Invalid frame height")
        self.assertGreater(fps, 0.0, "Invalid FPS")
        self.assertGreater(frame_count, 0, "Invalid frame count")

        # Decode the first 5 frames
        frames_decoded = 0
        for i in range(min(5, frame_count)):
            ret, frame = cap.read()
            self.assertTrue(ret, f"Failed to decode frame {i}")
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (height, width, 3))
            # Test color conversion to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.assertEqual(rgb.shape, (height, width, 3))
            frames_decoded += 1

        cap.release()
        self.assertEqual(frames_decoded, min(5, frame_count))
        print(f"         Successfully decoded {frames_decoded} frames.")

    def test_03_opencv5_frame_analyzer_pipeline(self):
        """Exercise the substantive F.O.R.G.E. OpenCV 5 quality & viewpoint curation engine."""
        if self.video_path is None:
            self.skipTest("No drone video available for testing.")

        print(f"\n[Test 3] Exercising OpenCV5FrameAnalyzer on {self.video_path.name}...")
        out_dir = Path(self.temp_dir) / "curated_frames"

        analyzer = OpenCV5FrameAnalyzer(orb_nfeatures=500)

        # Extract 3 curated frames across a quick window
        extracted_dicts, curation_report = analyzer.extract_curated_frames(
            video_path=str(self.video_path),
            output_dir=str(out_dir),
            target_num_frames=3,
            save_metadata=True,
            t_start=0.0,
            t_end=5.0
        )

        self.assertEqual(len(extracted_dicts), 3)
        self.assertEqual(len(curation_report.selected_frames), 3)

        # Verify output files on disk
        meta_file = out_dir / "opencv5_curation_metadata.json"
        self.assertTrue(meta_file.is_file(), "Curation metadata JSON was not written")

        for d in extracted_dicts:
            fp = Path(d["file_path"])
            self.assertTrue(fp.is_file(), f"Extracted frame file does not exist: {fp}")
            self.assertGreater(fp.stat().st_size, 0, f"Extracted frame file is empty: {fp}")
            self.assertGreater(d["width"], 0)
            self.assertGreater(d["height"], 0)

        # Verify quantitative metrics were substantively calculated
        for s in curation_report.selected_frames:
            m = s.metrics
            self.assertGreater(m.laplacian_sharpness, 0.0, "Laplacian sharpness must be > 0")
            self.assertGreater(m.tenengrad_sharpness, 0.0, "Tenengrad sharpness must be > 0")
            self.assertGreaterEqual(m.dynamic_range_p95_p05, 0.0)
            self.assertGreater(m.composite_quality_score, 0.0)
            print(f"         Selected Frame {s.frame_index:04d} (t={s.timestamp_sec:.2f}s): "
                  f"LapSharp={m.laplacian_sharpness:.1f}, "
                  f"Tenengrad={m.tenengrad_sharpness:.1f}, "
                  f"Score={m.composite_quality_score:.2f}")

    def test_04_standard_frame_extraction(self):
        """Verify standard extraction (extract_frames) works seamlessly under OpenCV 5."""
        if self.video_path is None:
            self.skipTest("No drone video available for testing.")

        print(f"\n[Test 4] Verifying extract_frames under OpenCV 5...")
        out_dir = Path(self.temp_dir) / "std_frames"
        frames = extract_frames(
            video_path=str(self.video_path),
            output_dir=str(out_dir),
            target_fps=1.0,
            max_frames=2
        )
        self.assertEqual(len(frames), 2)
        for f in frames:
            p = Path(f.file_path)
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size, 0)
        print(f"         Successfully extracted {len(frames)} frames with extract_frames.")


def run_standalone_test(video_path_arg: Optional[str] = None) -> bool:
    """Executes the test suite with formatted console reporting."""
    print("=" * 70)
    print("F.O.R.G.E. — OPENCV 5 COMPLIANCE & VERIFICATION SUITE")
    print("=" * 70)
    print(f"Active Python:  {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"cv2 Version:    {getattr(cv2, '__version__', 'NOT FOUND')}")
    print(f"cv2 File:       {getattr(cv2, '__file__', 'NOT FOUND')}")

    try:
        vid = resolve_test_video(video_path_arg)
        TestOpenCV5Compliance.video_path = vid
        print(f"Target Video:   {vid.resolve()} ({vid.stat().st_size / (1024*1024):.1f} MB)")
    except FileNotFoundError as e:
        print(f"Target Video:   FAILED TO LOCATE: {e}")
        return False

    print("-" * 70)

    suite = unittest.TestSuite()
    suite.addTest(TestOpenCV5Compliance("test_01_opencv5_runtime_version"))
    suite.addTest(TestOpenCV5Compliance("test_02_video_capture_and_decoding"))
    suite.addTest(TestOpenCV5Compliance("test_03_opencv5_frame_analyzer_pipeline"))
    suite.addTest(TestOpenCV5Compliance("test_04_standard_frame_extraction"))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("RESULT: ALL OPENCV 5 COMPLIANCE TESTS PASSED (100% SUCCESS)")
    else:
        print(f"RESULT: FAILED ({len(result.failures)} failures, {len(result.errors)} errors)")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F.O.R.G.E OpenCV 5 Compliance Smoke Test")
    parser.add_argument("--video", default=None, help="Path to test drone video file")
    args = parser.parse_args()

    success = run_standalone_test(args.video)
    sys.exit(0 if success else 1)
