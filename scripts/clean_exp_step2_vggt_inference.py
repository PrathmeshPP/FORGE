"""
F.O.R.G.E. Clean-Slate Experiment — Step 2: Fresh VGGT Inference & Confidence Calibration
Processes 54 adaptive curated frames across 17 sliding windows (size=6, overlap=3).
Maintains RTX 2070 SUPER 8 GB constraint (concurrency strictly 1).
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for _cand in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega"
]:
    if Path(_cand).is_dir() and _cand not in sys.path:
        sys.path.insert(0, _cand)

from src.vggt.inference import load_vggt_model, DEFAULT_CHECKPOINT_PATH
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

EXP_DIR = Path("outputs/indoor/clean_reconstruction_video1_20261030")
FRAMES_DIR = EXP_DIR / "frames"
CACHE_DIR = EXP_DIR / "vggt_cache"
DIAG_DIR = EXP_DIR / "diagnostics"

for d in [CACHE_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — CLEAN-SLATE STEP 2: FRESH VGGT INFERENCE")
print(f"  Target Dir: {EXP_DIR}")
print("=" * 76)

# Load comparison report
comp_p = DIAG_DIR / "frame_selection_comparison.json"
comp = json.loads(comp_p.read_text())
frame_records = comp["frames"]
N_frames = len(frame_records)
image_paths = [Path(r["file_path"]) for r in frame_records]

print(f"[*] Total Curated Keyframes: {N_frames}")

# Setup 17 sliding windows: size=6, overlap=3
window_size = 6
overlap = 3
step = window_size - overlap
windows = []
start = 0
while start + window_size <= N_frames:
    windows.append(list(range(start, start + window_size)))
    start += step
if start < N_frames and windows[-1] != list(range(max(0, N_frames - window_size), N_frames)):
    windows.append(list(range(max(0, N_frames - window_size), N_frames)))

N_windows = len(windows)
print(f"[*] Sliding Windows: {N_windows} windows (size={window_size}, overlap={overlap})")
for w_i, win in enumerate(windows):
    print(f"    Window {w_i:02d}: frames [{win[0]:02d}:{win[-1]:02d}] (t={frame_records[win[0]]['timestamp_sec']:.2f}s to {frame_records[win[-1]]['timestamp_sec']:.2f}s)")

# Run Fresh VGGT Inference
cache_file = CACHE_DIR / f"raw_windows_0_to_{N_windows-1}.npz"
raw_windows_data = []

print(f"\n[*] Initializing VGGT-Omega model on RTX 2070 SUPER...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
t0_load = time.perf_counter()
model = load_vggt_model(DEFAULT_CHECKPOINT_PATH, device)
print(f"[+] VGGT model loaded in {time.perf_counter() - t0_load:.2f}s")

npz_dict = {}
inference_times = []
t0_all_inf = time.perf_counter()

for w_idx, win in enumerate(windows):
    t_w_start = time.perf_counter()
    win_paths = [image_paths[i] for i in win]
    inp = load_and_preprocess_images([str(p) for p in win_paths], image_resolution=320).to(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model(inp)
        ext, intr = encoding_to_camera(out.get("pose_enc"), image_size_hw=inp.shape[-2:])

    ext_np = ext.squeeze(0).cpu().numpy().astype(np.float32)
    intr_np = intr.squeeze(0).cpu().numpy().astype(np.float32)
    depth_np = out["depth"].squeeze(0).cpu().numpy().astype(np.float32)
    if depth_np.ndim == 4:
        depth_np = depth_np.squeeze(-1)
    conf_np = out["depth_conf"].squeeze(0).cpu().numpy().astype(np.float32)
    if conf_np.ndim == 4:
        conf_np = conf_np.squeeze(-1)

    dt = time.perf_counter() - t_w_start
    inference_times.append(dt)

    d = {"ext": ext_np, "intr": intr_np, "depth": depth_np, "conf": conf_np}
    raw_windows_data.append(d)
    npz_dict[f"ext_{w_idx}"] = ext_np
    npz_dict[f"intr_{w_idx}"] = intr_np
    npz_dict[f"depth_{w_idx}"] = depth_np
    npz_dict[f"conf_{w_idx}"] = conf_np

    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0
    print(f"  Window {w_idx:02d}/{N_windows-1}: frames [{win[0]:02d}:{win[-1]:02d}] | time: {dt:.2f}s | VRAM: {vram_mb:.1f} MB")

np.savez_compressed(cache_file, **npz_dict)
total_inf_time = time.perf_counter() - t0_all_inf
print(f"[+] Saved all {N_windows} fresh windows to {cache_file.name} ({cache_file.stat().st_size / (1024**2):.2f} MB)")
print(f"[+] Total fresh inference time: {total_inf_time:.2f}s (Avg {np.mean(inference_times):.2f}s/window)")

# Clean GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Calibrate Depth Confidence Distribution for this clean run
print("\n[*] Calibrating depth confidence distribution across all fresh windows...")
all_confs = []
for d in raw_windows_data:
    c = d["conf"][:, ::3, ::3].ravel()
    all_confs.append(c)
all_confs = np.concatenate(all_confs)

conf_stats = {
    "total_samples": int(len(all_confs)),
    "min": float(np.min(all_confs)),
    "p05": float(np.percentile(all_confs, 5)),
    "p10": float(np.percentile(all_confs, 10)),
    "p15": float(np.percentile(all_confs, 15)),
    "p20": float(np.percentile(all_confs, 20)),
    "p25": float(np.percentile(all_confs, 25)),
    "p30": float(np.percentile(all_confs, 30)),
    "median": float(np.median(all_confs)),
    "mean": float(np.mean(all_confs)),
    "std": float(np.std(all_confs)),
    "p75": float(np.percentile(all_confs, 75)),
    "p90": float(np.percentile(all_confs, 90)),
    "max": float(np.max(all_confs)),
    "recommended_p20_threshold": float(np.percentile(all_confs, 20))
}

print(f"  Confidence Min:    {conf_stats['min']:.4f}")
print(f"  Confidence P10:    {conf_stats['p10']:.4f}")
print(f"  Confidence P20:    {conf_stats['p20']:.4f}  <-- FRESH CALIBRATED THRESHOLD")
print(f"  Confidence Median: {conf_stats['median']:.4f}")
print(f"  Confidence Mean:   {conf_stats['mean']:.4f}")
print(f"  Confidence Max:    {conf_stats['max']:.4f}")

conf_out_p = DIAG_DIR / "confidence_distribution.json"
conf_out_p.write_text(json.dumps(conf_stats, indent=2))
print(f"[+] Saved fresh confidence calibration to {conf_out_p.name}")

