import cv2, numpy as np
from pathlib import Path

v12_dir = Path('outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')
dense_dir = v12_dir / 'dense_mvs'
dm_dir = dense_dir / 'stereo' / 'depth_maps'
nm_dir = dense_dir / 'stereo' / 'normal_maps'
im_dir = dense_dir / 'images'

def read_colmap_bin(path):
    with open(path, 'rb') as f:
        header = ''
        amp_count = 0
        while amp_count < 3:
            char = f.read(1).decode()
            if char == '&':
                amp_count += 1
            header += char
        parts = header.strip('&').split('&')
        w, h, c = int(parts[0]), int(parts[1]), int(parts[2])
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((h, w, c))

fname = 'frame_0006_41.391s.png'
geom_dm = read_colmap_bin(dm_dir / f'{fname}.geometric.bin')[:, :, 0]
geom_nm = read_colmap_bin(nm_dir / f'{fname}.geometric.bin')

regions = {
    'Building Roof': (slice(520, 680), slice(550, 1150)),
    'Building Facade': (slice(680, 780), slice(600, 1000)),
    'Courtyard Ground': (slice(750, 880), slice(550, 1100)),
    'Road Surface': (slice(750, 890), slice(1200, 1550)),
    'Vegetation / Trees': (slice(350, 520), slice(100, 450))
}

print(f"=== DETAILED REGIONAL DEPTH & NORMAL AUDIT FOR {fname} ===")
for rname, (ys, xs) in regions.items():
    sub_d = geom_dm[ys, xs]
    sub_n = geom_nm[ys, xs]
    
    total_px = sub_d.size
    valid_mask = sub_d > 0
    valid_px = np.sum(valid_mask)
    coverage = (valid_px / total_px) * 100.0
    
    if valid_px > 0:
        val_d = sub_d[valid_mask]
        d_min, d_max, d_med = np.min(val_d), np.max(val_d), np.median(val_d)
        d_iqr = np.percentile(val_d, 75) - np.percentile(val_d, 25)
        
        # Ripple / High-frequency scanline noise detection:
        # Compute horizontal derivative variance vs vertical derivative variance
        # SGBM produces severe horizontal scanline oscillations
        dx = np.diff(sub_d, axis=1)
        dy = np.diff(sub_d, axis=0)
        # mask where both pixels are valid
        valid_dx = (sub_d[:, :-1] > 0) & (sub_d[:, 1:] > 0)
        valid_dy = (sub_d[:-1, :] > 0) & (sub_d[1:, :] > 0)
        
        dx_val = dx[valid_dx]
        dy_val = dy[valid_dy]
        
        dx_std = np.std(dx_val) if len(dx_val) > 0 else 0.0
        dy_std = np.std(dy_val) if len(dy_val) > 0 else 0.0
        ratio = (dx_std / (dy_std + 1e-6))
        
        # Normals audit
        val_n = sub_n[valid_mask]
        norms_len = np.linalg.norm(val_n, axis=-1)
        valid_norm_mask = norms_len > 0.5
        norm_valid_pct = (np.sum(valid_norm_mask) / valid_px) * 100.0
        
        print(f"\n[{rname}]")
        print(f"  Valid Depth Coverage:  {valid_px}/{total_px} ({coverage:.1f}%)")
        print(f"  Depth (min/med/max):   {d_min:.3f} / {d_med:.3f} / {d_max:.3f} units (IQR={d_iqr:.3f})")
        print(f"  Gradient Continuity:   std(dZ/dx)={dx_std:.4f}, std(dZ/dy)={dy_std:.4f} (Ratio H/V={ratio:.2f})")
        print(f"  Ripple Assessment:     {'SMOOTH / CONTINUOUS (No scanline artifacts)' if 0.5 < ratio < 2.0 else 'CHECK FOR ANISOTROPY'}")
        print(f"  Normal Validity:       {norm_valid_pct:.1f}% valid tangent normals")
    else:
        print(f"\n[{rname}] NO VALID PIXELS")
