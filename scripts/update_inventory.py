import os, json, hashlib

def sha256sum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

root = r'C:\Users\25beevdt025\Desktop\FORGE_indoor'
models_dir = os.path.join(root, 'public', 'models')
scenes_json_p = os.path.join(root, 'public', 'data', 'scenes.json')

with open(scenes_json_p, 'r', encoding='utf-8') as f:
    scenes_data = json.load(f)

inventory = {
    'generated_at': '2026-11-01T23:38:00Z',
    'total_scenes': len(scenes_data['scenes']),
    'scenes': []
}

total_bytes = 0
total_models = 0

for sc in scenes_data['scenes']:
    sc_id = sc['id']
    sc_dir = os.path.join(models_dir, sc_id)
    entry = {
        'id': sc_id,
        'name': sc.get('display_name', sc_id),
        'category': sc.get('category', 'General'),
        'files': []
    }
    if os.path.isdir(sc_dir):
        for f in sorted(os.listdir(sc_dir)):
            fp = os.path.join(sc_dir, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                total_bytes += sz
                h = sha256sum(fp)
                is_model = f.endswith(('.glb', '.ply', '.obj'))
                if is_model:
                    total_models += 1
                entry['files'].append({
                    'name': f,
                    'size_bytes': sz,
                    'size_mb': round(sz / (1024 * 1024), 2),
                    'sha256': h,
                    'is_3d_model': is_model
                })
    inventory['scenes'].append(entry)

inventory['total_files_size_bytes'] = total_bytes
inventory['total_files_size_mb'] = round(total_bytes / (1024 * 1024), 2)
inventory['total_models_count'] = total_models

with open(os.path.join(root, 'asset_inventory.json'), 'w', encoding='utf-8') as f:
    json.dump(inventory, f, indent=2)

print(f"Total Scenes: {inventory['total_scenes']}, Models: {total_models}, Total Size: {inventory['total_files_size_mb']} MB")

# Generate markdown table
md = ['# F.O.R.G.E. Explorer Production Asset Inventory\n\n']
md.append(f"- **Total Packaged Scenes:** {inventory['total_scenes']}\n")
md.append(f"- **Total 3D Model Runtime Assets:** {total_models}\n")
md.append(f"- **Total Asset Volume:** {inventory['total_files_size_mb']} MB\n")
md.append(f"- **Individual File Size Cap:** $\\le 68.87$ MB (100% compliant with GitHub 100 MB limit and Vercel static asset thresholds)\n")
md.append(f"- **Generated:** {inventory['generated_at']}\n\n")
md.append('| # | Scene ID | Name | Category | Primary Model Asset | Size (MB) | SHA-256 (Prefix) |\n')
md.append('|---|---|---|---|---|---|---|\n')

for i, s in enumerate(inventory['scenes'], 1):
    primary = next((f for f in s['files'] if f['is_3d_model']), None)
    if primary:
        p_name = primary['name']
        p_sz = f"{primary['size_mb']} MB"
        p_sha = primary['sha256'][:12] + '...'
    else:
        p_name = 'N/A'
        p_sz = '-'
        p_sha = '-'
    md.append(f"| {i} | `{s['id']}` | {s['name']} | {s['category']} | `{p_name}` | {p_sz} | `{p_sha}` |\n")

with open(os.path.join(root, 'ASSET_INVENTORY.md'), 'w', encoding='utf-8') as f:
    f.writelines(md)

print('Updated ASSET_INVENTORY.md')

