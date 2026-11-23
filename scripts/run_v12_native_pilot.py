import os, sys, subprocess, time
from pathlib import Path

v10_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v10')
v12_dir = Path('C:/Users/25beevdt025/Desktop/FORGE/outputs/new_video_reconstructions/whatsapp_video_2026_10_24/best_reconstruction_campaign/final_best_canonical_v12')
pilot_dir = v12_dir / 'pilot_mvs'
pilot_dir.mkdir(parents=True, exist_ok=True)

v10_images = v10_dir / 'pilot_sfm' / 'images'
v10_sparse = v10_dir / 'pilot_sfm' / 'sparse' / '0'

# Step 1: Write pilot_image_list.txt (frames 5 to 11, 7 frames)
pilot_frames = [
    'frame_0005_39.826s.png',
    'frame_0006_41.391s.png',
    'frame_0007_42.957s.png',
    'frame_0008_44.522s.png',
    'frame_0009_46.087s.png',
    'frame_0010_47.652s.png',
    'frame_0011_49.217s.png'
]

list_path = pilot_dir / 'pilot_image_list.txt'
with open(list_path, 'w') as f:
    for name in pilot_frames:
        f.write(name + '\n')

print(f'Wrote {len(pilot_frames)} images to {list_path}')

# Step 2: Run image_undistorter
print('\n=== Running colmap image_undistorter ===')
cmd_undistort = [
    'conda', 'run', '-n', 'colmap_cuda', 'colmap', 'image_undistorter',
    '--image_path', str(v10_images),
    '--input_path', str(v10_sparse),
    '--output_path', str(pilot_dir),
    '--image_list_path', str(list_path),
    '--output_type', 'COLMAP',
    '--max_image_size', '1600'
]
t0 = time.time()
res = subprocess.run(cmd_undistort, capture_output=True, text=True)
print(f'Undistorter finished in {time.time()-t0:.2f}s (code {res.returncode})')
if res.returncode != 0:
    print('Undistorter STDOUT:', res.stdout)
    print('Undistorter STDERR:', res.stderr)
    sys.exit(1)

# Step 3: Run patch_match_stereo with geometric consistency
print('\n=== Running colmap patch_match_stereo (CUDA) ===')
cmd_patch_match = [
    'conda', 'run', '-n', 'colmap_cuda', 'colmap', 'patch_match_stereo',
    '--workspace_path', str(pilot_dir),
    '--workspace_format', 'COLMAP',
    '--PatchMatchStereo.max_image_size', '1600',
    '--PatchMatchStereo.gpu_index', '0',
    '--PatchMatchStereo.geom_consistency', '1'
]
t0 = time.time()
res = subprocess.run(cmd_patch_match, capture_output=True, text=True)
print(f'PatchMatch finished in {time.time()-t0:.2f}s (code {res.returncode})')
if res.returncode != 0:
    print('PatchMatch STDOUT:', res.stdout)
    print('PatchMatch STDERR:', res.stderr)
    sys.exit(1)
else:
    print('PatchMatch STDOUT tail:\n', '\n'.join(res.stdout.splitlines()[-15:]))

# Step 4: Run stereo_fusion
print('\n=== Running colmap stereo_fusion ===')
fused_ply = pilot_dir / 'pilot_fused.ply'
cmd_fuse = [
    'conda', 'run', '-n', 'colmap_cuda', 'colmap', 'stereo_fusion',
    '--workspace_path', str(pilot_dir),
    '--workspace_format', 'COLMAP',
    '--input_type', 'geometric',
    '--output_path', str(fused_ply)
]
t0 = time.time()
res = subprocess.run(cmd_fuse, capture_output=True, text=True)
print(f'Stereo fusion finished in {time.time()-t0:.2f}s (code {res.returncode})')
if res.returncode != 0:
    print('Stereo fusion STDOUT:', res.stdout)
    print('Stereo fusion STDERR:', res.stderr)
    sys.exit(1)
else:
    print('Stereo fusion STDOUT tail:\n', '\n'.join(res.stdout.splitlines()[-15:]))

print('\nPilot dense reconstruction complete! Fused PLY:', fused_ply)
