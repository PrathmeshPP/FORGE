"""
F.O.R.G.E. Stage 1 — VGGT Inference & Confidence Calibration for Video 1
Source: WhatsApp Video 2026-10-30 at 17.02.34.mp4 (Hostel Room B 314)
Output: outputs/indoor/vggt_tsdf_video_20261030_170234
"""

import sys
import os
import shutil
import json
import time
import math
from pathlib import Path
import numpy as np
import torch
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for _candidate in [
    r"C:\Users\25beevdt025\Desktop\SIH\third_party\vggt_omega",
    r"C:\Users\25beevdt025\Desktop\coding\third_party\vggt_omega"
]:
    if Path(_candidate).is_dir() and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from src.vggt.inference import load_vggt_model, DEFAULT_CHECKPOINT_PATH
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

OUTPUT_DIR = PROJECT_ROOT / "outputs/indoor/vggt_tsdf_video_20261030_170234"
FRAMES_DIR = OUTPUT_DIR / "frames"
CACHE_DIR = OUTPUT_DIR / "vggt_cache"
DIAG_DIR = OUTPUT_DIR / "diagnostics"

for d in [FRAMES_DIR, CACHE_DIR, DIAG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 76)
print("  F.O.R.G.E. — STAGE 1: VGGT INFERENCE & CONFIDENCE CALIBRATION")
print("  Target: Hostel Room B 314 (WhatsApp Video 2026-10-30 at 17.02.34.mp4)")
print("=" * 76)

# 1. Setup Curated Frames
SRC_FRAMES_DIR = PROJECT_ROOT / "outputs/indoor/video_1_170234/frames"
src_frame_files = sorted(list(SRC_FRAMES_DIR.glob("*.png")))
print(f"[*] Found {len(src_frame_files)} source curated frames.")

dest_image_paths = []
frame_records = []
src_report_p = PROJECT_ROOT / "outputs/indoor/video_1_170234/frame_sampling_report.json"
src_report = json.loads(src_report_p.read_text()) if src_report_p.exists() else {}
src_records_map = {r["rank"]: r for r in src_report.get("selected_frames", [])}

for idx, src_p in enumerate(src_frame_files):
    dst_p = FRAMES_DIR / src_p.name
    if not dst_p.exists():
        shutil.copy2(src_p, dst_p)
    dest_image_paths.append(dst_p)
    
    # Metadata
    rec = src_records_map.get(idx, {})
    frame_records.append({
        "rank": idx,
        "filename": dst_p.name,
        "file_path": str(dst_p.resolve()),
        "frame_index": rec.get("frame_index", idx),
        "timestamp_sec": rec.get("timestamp_sec", idx * 0.64),
        "sharpness": rec.get("laplacian_sharpness", 60.0),
        "luminance": rec.get("mean_luminance", 110.0),
        "contrast": rec.get("contrast_rms", 45.0)
    })

N_frames = len(dest_image_paths)
print(f"[+] Prepared {N_frames} frames in {FRAMES_DIR}")

# Save curation report
curation_out = {
    "video_path": r"C:\Users\25beevdt025\Downloads\WhatsApp Video 2026-10-30 at 17.02.34.mp4",
    "total_frames": N_frames,
    "sampling_density_fps": round(N_frames / 40.24, 3),
    "frames": frame_records
}
(OUTPUT_DIR / "frame_curation_report.json").write_text(json.dumps(curation_out, indent=2))

# 2. Configure Windows: size 6, overlap 3
window_size = 6
overlap = 3
step = window_size - overlap
windows = []
start = 0
while start + window_size <= N_frames:
    windows.append(list(range(start, start + window_size)))
    start += step
if start < N_frames:
    windows.append(list(range(max(0, N_frames - window_size), N_frames)))

N_windows = len(windows)
print(f"[*] Sliding Windows: {N_windows} windows (size={window_size}, overlap={overlap})")

# 3. VGGT Inference
cache_file = CACHE_DIR / f"raw_windows_0_to_{N_windows-1}.npz"
raw_windows_data = []

if cache_file.exists():
    print(f"[*] Loading existing raw windows from {cache_file.name}...")
    cached = np.load(cache_file)
    for w in range(N_windows):
        raw_windows_data.append({
            "ext": cached[f"ext_{w}"],
            "intr": cached[f"intr_{w}"],
            "depth": cached[f"depth_{w}"],
            "conf": cached[f"conf_{w}"]
        })
    print(f"[+] Loaded all {N_windows} windows from cache.")
else:
    print(f"[*] Initializing VGGT-Omega model on RTX 2070 SUPER...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = load_vggt_model(DEFAULT_CHECKPOINT_PATH, device)
    print(f"[+] Model loaded in {time.perf_counter() - t0:.2f}s")
    
    npz_dict = {}
    inference_times = []
    
    for w_idx, win in enumerate(windows):
        t_w_start = time.perf_counter()
        win_paths = [dest_image_paths[i] for i in win]
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
    print(f"[+] Saved all {N_windows} windows to {cache_file.name} ({cache_file.stat().st_size / (1024**2):.2f} MB)")
    print(f"[+] Total inference time: {sum(inference_times):.2f}s (Avg {np.mean(inference_times):.2f}s/window)")

# Clean GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 4. Measure Confidence Distribution & Calibrate Threshold
print("\n[*] Calibrating depth confidence distribution across all windows...")
all_confs = []
for d in raw_windows_data:
    # Subsample 10% to prevent excessive memory
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
    "p40": float(np.percentile(all_confs, 40)),
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
print(f"  Confidence P20:    {conf_stats['p20']:.4f}  <-- CALIBRATED THRESHOLD")
print(f"  Confidence Median: {conf_stats['median']:.4f}")
print(f"  Confidence Mean:   {conf_stats['mean']:.4f}")
print(f"  Confidence Max:    {conf_stats['max']:.4f}")

conf_out_p = DIAG_DIR / "confidence_distribution.json"
conf_out_p.write_text(json.dumps(conf_stats, indent=2))
print(f"[+] Saved confidence distribution to {conf_out_p.name}")

