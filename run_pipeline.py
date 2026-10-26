"""
F.O.R.G.E — Real Video Pipeline Runner
End-to-end execution: Video Ingestion -> Frame Extraction -> VGGT-Omega Spatial Inference -> Output Persistence.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import torch

# Ensure src package is importable
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.video.extract_frames import extract_frames, ExtractedFrame
from src.vggt.inference import VGGTInferencePipeline, DEFAULT_CHECKPOINT_PATH


def run_pipeline(
    video_path: str,
    output_dir: str = str(WORKSPACE_ROOT / "outputs"),
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    target_fps: float = 1.0,
    max_frames: int = 3,
    image_resolution: int = 512,
    frame_selector: str = "opencv5",
    device: Optional[str] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> dict:
    """
    Executes the complete video-to-3D-predictions pipeline.
    """
    pipeline_t0 = time.perf_counter()
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    out_root = Path(output_dir)
    frames_dir = out_root / "frames"
    preds_dir = out_root / "predictions"
    frames_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("F.O.R.G.E Phase 2A — Real Video Pipeline")
    print("=" * 65)
    print(f"Input Video:        {video_path}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Output Directory:   {output_dir}")
    print(f"Frame Selector:     {frame_selector}")
    print(f"Target FPS:         {target_fps}")
    print(f"Max Frames:         {max_frames}")
    print(f"Resolution:         {image_resolution}x{image_resolution}")
    if t_start is not None or t_end is not None:
        print(f"Time Range:         {t_start}s to {t_end}s")
    print("-" * 65)

    # 1. Video Frame Extraction
    print("\n[Step 1/3] Extracting frames from video...")
    ext_t0 = time.perf_counter()
    if frame_selector == "opencv5":
        print("  Using OpenCV 5 Frame Quality & Viewpoint Diversity Analyzer...")
        from src.video.opencv5_frame_analyzer import OpenCV5FrameAnalyzer
        analyzer = OpenCV5FrameAnalyzer()
        extracted_dicts, curation_report = analyzer.extract_curated_frames(
            video_path=video_path,
            output_dir=str(frames_dir),
            target_num_frames=max_frames,
            save_metadata=True,
            t_start=t_start,
            t_end=t_end
        )
        extracted_frames = [
            ExtractedFrame(
                frame_index=d["frame_index"],
                timestamp_seconds=d["timestamp_seconds"],
                file_path=d["file_path"],
                width=d["width"],
                height=d["height"]
            )
            for d in extracted_dicts
        ]
    elif frame_selector == "default":
        extracted_frames = extract_frames(
            video_path=video_path,
            output_dir=str(frames_dir),
            target_fps=target_fps,
            max_frames=max_frames
        )
    else:
        raise ValueError(f"Unknown frame selector '{frame_selector}'. Choose from ['default', 'opencv5'].")

    extract_time = time.perf_counter() - ext_t0
    print(f"  Extracted {len(extracted_frames)} frames in {extract_time:.3f}s")
    for f in extracted_frames:
        print(f"   - Frame {f.frame_index:03d}: t={f.timestamp_seconds:.3f}s -> {Path(f.file_path).name}")

    # 2. VGGT-Omega Spatial Inference
    print(f"\n[Step 2/3] Initializing VGGT-Omega and running spatial inference on {device}...")
    pipeline = VGGTInferencePipeline(
        checkpoint_path=checkpoint_path,
        device=device,
        image_resolution=image_resolution
    )
    load_time = pipeline.initialize()
    if load_time > 0:
        print(f"  Model loaded into {device} in {load_time:.2f}s")

    frame_paths = [f.file_path for f in extracted_frames]
    infer_results = pipeline.run_inference(frame_paths)
    metrics = infer_results["metrics"]

    print(f"  Inference completed in {metrics['inference_time_sec']:.3f}s")
    print(f"  Peak VRAM Allocated: {metrics['peak_vram_allocated_mb']:.2f} MB")
    print(f"  Peak VRAM Reserved:  {metrics['peak_vram_reserved_mb']:.2f} MB")

    # 3. Output Persistence
    print("\n[Step 3/3] Saving prediction arrays and pipeline metadata...")
    saved_files = pipeline.save_predictions(infer_results, str(preds_dir))
    for name, path in saved_files.items():
        print(f"   - Saved {name}: {Path(path).name}")

    total_pipeline_time = time.perf_counter() - pipeline_t0

    # Build metadata.json
    metadata = {
        "pipeline": "F.O.R.G.E Phase 2A — Real Video Inference Pipeline",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "video": {
            "source_path": str(Path(video_path).resolve()),
            "frame_selector": frame_selector,
            "frames_extracted": len(extracted_frames),
            "target_fps": target_fps,
            "max_frames_cap": max_frames,
            "frame_list": [
                {
                    "index": f.frame_index,
                    "timestamp_s": f.timestamp_seconds,
                    "filename": Path(f.file_path).name,
                    "width": f.width,
                    "height": f.height
                }
                for f in extracted_frames
            ]
        },
        "model": {
            "checkpoint_path": checkpoint_path,
            "resolution": image_resolution,
            "device": device
        },
        "performance": {
            "model_load_time_sec": round(load_time, 3),
            "frame_extraction_time_sec": round(extract_time, 3),
            "vggt_inference_time_sec": metrics["inference_time_sec"],
            "total_pipeline_time_sec": round(total_pipeline_time, 3),
            "peak_vram_allocated_mb": metrics["peak_vram_allocated_mb"],
            "peak_vram_reserved_mb": metrics["peak_vram_reserved_mb"]
        },
        "outputs": {
            "frames_directory": str(frames_dir.resolve()),
            "predictions_directory": str(preds_dir.resolve()),
            "saved_prediction_files": saved_files
        },
        "status": "SUCCESS"
    }

    metadata_path = out_root / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   - Saved metadata.json: {metadata_path.resolve()}")

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION: SUCCESS")
    print(f"Total time elapsed: {total_pipeline_time:.2f}s")
    print("=" * 65)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run F.O.R.G.E Real Video Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs"), help="Outputs directory")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_PATH, help="VGGT-Omega checkpoint path")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=3, help="Max frames to sample (default: 3)")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution for VGGT (default: 512)")
    parser.add_argument(
        "--frame-selector",
        choices=["opencv5", "default"],
        default="opencv5",
        help="Frame selection strategy: 'opencv5' (OpenCV 5 quality/diversity curation, default) or 'default' (fixed FPS extraction fallback)"
    )
    parser.add_argument("--device", default=None, help="Torch device (default: auto cuda if available else cpu)")
    parser.add_argument("--t-start", type=float, default=None, help="Optional start timestamp in seconds")
    parser.add_argument("--t-end", type=float, default=None, help="Optional end timestamp in seconds")
    args = parser.parse_args()

    meta = run_pipeline(
        video_path=args.video,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        target_fps=args.fps,
        max_frames=args.max_frames,
        image_resolution=args.resolution,
        frame_selector=args.frame_selector,
        device=args.device,
        t_start=args.t_start,
        t_end=args.t_end
    )
