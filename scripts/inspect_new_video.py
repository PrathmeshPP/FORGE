r"""
Inspect the new input video file:
C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

VIDEO_PATH = Path(r"C:\Users\25beevdt025\Desktop\dji_fly_20261024_173224_0168_1792844321671_video.mp4")

def inspect():
    print(f"[*] Checking video file: {VIDEO_PATH}")
    if not VIDEO_PATH.is_file():
        print(f"[-] ERROR: File does not exist: {VIDEO_PATH}")
        sys.exit(1)

    file_size_bytes = VIDEO_PATH.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    file_size_gb = round(file_size_bytes / (1024 * 1024 * 1024), 3)
    print(f"    File size: {file_size_bytes:,} bytes ({file_size_mb} MB / {file_size_gb} GB)")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("[-] ERROR: cv2.VideoCapture failed to open the file!")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    print(f"    Resolution: {width} x {height}")
    print(f"    FPS: {fps:.3f}")
    print(f"    Frame count: {frame_count}")
    print(f"    Duration: {duration_sec:.2f} seconds ({duration_sec / 60:.2f} minutes)")
    print(f"    FourCC codec: {fourcc_str} ({fourcc_int})")

    # Decoding tests: sample at 0%, 25%, 50%, 75%, 99%
    test_fractions = [0.0, 0.25, 0.50, 0.75, 0.99]
    print("\n[*] Testing frame decoding across video timeline:")
    sample_frames = {}
    for frac in test_fractions:
        target_idx = int(frac * (frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"    [-] Frame at index {target_idx} ({frac*100:.0f}%): DECODING FAILED")
        else:
            actual_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            mean_bgr = frame.mean(axis=(0, 1)).round(1)
            sharpness = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            print(f"    [+] Frame at index {actual_idx} ({frac*100:.0f}% | t={actual_idx/fps:.2f}s): "
                  f"Decoded OK, shape={frame.shape}, mean BGR={mean_bgr.tolist()}, sharpness={sharpness:.1f}")
            sample_frames[actual_idx] = frame

    cap.release()
    print("\n[+] Video inspection completed successfully!")

if __name__ == "__main__":
    inspect()
