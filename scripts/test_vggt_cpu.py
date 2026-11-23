r"""
Test VGGT-Omega inference on CPU with 8 curated frames.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\25beevdt025\Desktop\FORGE")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.vggt.inference import VGGTInferencePipeline, DEFAULT_CHECKPOINT_PATH

FRAMES_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/frames"
PREDS_DIR = PROJECT_ROOT / "outputs/new_video_reconstructions/whatsapp_video_2026_10_24/camera"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    frame_paths = sorted([str(p) for p in FRAMES_DIR.glob("*.png")])
    print(f"Found {len(frame_paths)} frames for VGGT inference.")
    print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    print(f"Loading checkpoint: {DEFAULT_CHECKPOINT_PATH}")

    t0 = time.time()
    pipeline = VGGTInferencePipeline(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        device="cpu",
        image_resolution=512
    )
    load_time = pipeline.initialize()
    print(f"Model initialized in {load_time:.2f}s")

    print(f"Starting inference pass on {len(frame_paths)} frames...")
    infer_t0 = time.time()
    results = pipeline.run_inference(frame_paths)
    infer_time = time.time() - infer_t0
    print(f"Inference completed in {infer_time:.2f}s")

    print("Keys in results:", list(results.keys()))
    if "extrinsics" in results:
        print("Extrinsics shape:", results["extrinsics"].shape)
    if "intrinsics" in results:
        print("Intrinsics shape:", results["intrinsics"].shape)
    if "depth" in results:
        print("Depth shape:", results["depth"].shape)

    saved_files = pipeline.save_predictions(results, str(PREDS_DIR))
    print("Saved prediction files:", saved_files)

if __name__ == "__main__":
    main()
