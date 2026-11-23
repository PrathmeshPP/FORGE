import sys
import os
import time
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vggt.inference import VGGTInferencePipeline

frames_dir = PROJECT_ROOT / "outputs/indoor/outdoor_pipeline_applied_video1/frames"
frame_files = sorted(list(frames_dir.glob("*.png")))[:8]
print(f"Testing {len(frame_files)} frames with VGGT 512...")

pipe = VGGTInferencePipeline(image_resolution=512)
t0 = time.perf_counter()
pipe.initialize()
print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

res = pipe.run_inference([str(p) for p in frame_files])
print(f"Inference SUCCESS!")
print(f"Inference time: {res['metrics']['inference_time_sec']:.2f}s")
print(f"Peak VRAM allocated: {res['metrics']['peak_vram_allocated_mb']:.1f} MB")
print(f"Peak VRAM reserved: {res['metrics']['peak_vram_reserved_mb']:.1f} MB")
print(f"Extrinsics shape: {res['extrinsics'].shape}")
print(f"Depth shape: {res['depth'].shape}")
print(f"Confidence shape: {res['depth_conf'].shape}")

