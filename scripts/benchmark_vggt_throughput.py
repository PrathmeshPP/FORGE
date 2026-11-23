"""
Benchmark script to measure VGGT-Omega throughput and GPU VRAM usage on RTX 4050 Laptop GPU.
Tests different window sizes and resolutions to find the optimal balance of speed, memory, and geometric quality.
"""

import os
import sys
import time
import torch
import numpy as np

for p in [r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega", r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vggt.inference import load_vggt_model, DEFAULT_CHECKPOINT_PATH
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

frames_dir = Path("outputs/indoor/room_001/attempt_4_new_video/frames")
test_frames = sorted([str(p) for p in frames_dir.glob("*.png")])[:8]

print(f"Device: {torch.cuda.get_device_name(0)}")
total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
print(f"Total VRAM: {total_vram_mb:.0f} MB")

t0 = time.perf_counter()
model = load_vggt_model(DEFAULT_CHECKPOINT_PATH, "cuda:0")
t_init = time.perf_counter() - t0
print(f"Model init time: {t_init:.2f} s")

# Benchmark at res=320 with window_size=6 and window_size=8
for res in [320]:
    for win_len in [6, 8]:
        sample_paths = test_frames[:win_len]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

        # Warmup
        inp = load_and_preprocess_images(sample_paths, image_resolution=res).to("cuda:0")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            _ = model(inp)
        torch.cuda.synchronize()

        # Timed run
        t_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            out = model(inp)
            pose_enc = out.get("pose_enc")
            ext, intr = encoding_to_camera(pose_enc, image_size_hw=inp.shape[-2:])
        torch.cuda.synchronize()
        dur = time.perf_counter() - t_start

        peak_vram = torch.cuda.max_memory_allocated(0) / (1024**2)
        reserved_vram = torch.cuda.max_memory_reserved(0) / (1024**2)
        fps = win_len / dur

        print(f"\n--- Res={res}, Window={win_len} ---")
        print(f"Input shape: {inp.shape}")
        print(f"Inference duration: {dur:.3f} s")
        print(f"Window Throughput: {fps:.2f} FPS")
        print(f"Peak VRAM allocated: {peak_vram:.1f} MB, reserved: {reserved_vram:.1f} MB")
        print(f"Extrinsics shape: {ext.shape}, Depth shape: {out['depth'].shape}")
