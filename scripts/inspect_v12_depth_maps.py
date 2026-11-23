import cv2, numpy as np
from pathlib import Path

v12_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')
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

test_frames = [
    'frame_0006_41.391s.png',
    'frame_0007_42.957s.png',
    'frame_0008_44.522s.png'
]

for fname in test_frames:
    rgb = cv2.imread(str(im_dir / fname))
    h, w = rgb.shape[:2]
    
    geo_dm = read_colmap_bin(dm_dir / f'{fname}.geometric.bin')
    geo_nm = read_colmap_bin(nm_dir / f'{fname}.geometric.bin')
    
    # Roof region crop in 1600x900 coordinates:
    # Roof is around y in [450, 750], x in [400, 1300]
    roof_mask = np.zeros((h, w), dtype=bool)
    roof_mask[480:720, 500:1200] = True
    
    roof_depths = geo_dm[roof_mask]
    valid_roof_depths = roof_depths[roof_depths > 0]
    
    print(f'=== {fname} ===')
    print(f'  Total valid pixels in frame: {np.sum(geo_dm > 0)} / {geo_dm.size} ({np.mean(geo_dm > 0)*100:.1f}%)')
    print(f'  Roof valid depth coverage:   {len(valid_roof_depths)} / {roof_depths.size} ({len(valid_roof_depths)/roof_depths.size*100:.1f}%)')
    print(f'  Roof depth range:            min={valid_roof_depths.min():.3f}, max={valid_roof_depths.max():.3f}, median={np.median(valid_roof_depths):.3f}')
    
    # Measure planarity on roof: fit a plane z = ax + by + c to the 3D points
    y_coords, x_coords = np.where(roof_mask & (geo_dm[:, :, 0] > 0))
    z_vals = geo_dm[y_coords, x_coords, 0]
    
    # RANSAC / Linear fit
    A = np.column_stack([x_coords, y_coords, np.ones_like(x_coords)])
    plane_params, residuals, rank, s = np.linalg.lstsq(A, z_vals, rcond=None)
    fit_z = A @ plane_params
    diff = np.abs(z_vals - fit_z)
    print(f'  Roof planar fit residual:    MAE={np.mean(diff):.4f} units, Std={np.std(diff):.4f} units, Max={np.max(diff):.4f} units')
    
    # Roof normals analysis
    roof_normals = geo_nm[roof_mask]
    valid_norms = roof_normals[np.linalg.norm(roof_normals, axis=-1) > 0.5]
    mean_norm = np.mean(valid_norms, axis=0)
    mean_norm /= np.linalg.norm(mean_norm)
    norm_variations = np.arccos(np.clip(valid_norms @ mean_norm, -1.0, 1.0)) * (180.0 / np.pi)
    print(f'  Roof normal mean:            {mean_norm}')
    print(f'  Roof normal angular dev:     Mean={np.mean(norm_variations):.2f} deg, Std={np.std(norm_variations):.2f} deg')
