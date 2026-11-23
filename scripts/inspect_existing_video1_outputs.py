import os
import json
from pathlib import Path

p = Path("outputs/indoor")
matches = []
for f in p.glob("*video*1*"):
    print(f"Dir: {f.name}")
    for sub in f.iterdir():
        if sub.is_file():
            print(f"  File: {sub.name} ({sub.stat().st_size / 1024:.1f} KB)")
        elif sub.is_dir():
            print(f"  Subdir: {sub.name}")
            for ssub in sub.iterdir():
                if ssub.is_file():
                    print(f"    {ssub.name} ({ssub.stat().st_size / 1024:.1f} KB)")

# Also check stage2_texture_video1
p2 = Path("outputs/indoor/stage2_texture_video1")
if p2.exists():
    print(f"\nDir: {p2.name}")
    for sub in p2.iterdir():
        if sub.is_file():
            print(f"  File: {sub.name} ({sub.stat().st_size / 1024:.1f} KB)")
        elif sub.is_dir():
            print(f"  Subdir: {sub.name}")

