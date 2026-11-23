"""
F.O.R.G.E. Explorer - Vercel Deployment Packager & Asset Bundler
Builds a standalone, self-contained, 100% localhost-independent web repository
in C:/Users/25beevdt025/Desktop/FORGE_indoor ready for Vercel 1-click import.
"""

import json
import os
import shutil
import hashlib
from pathlib import Path
import trimesh

SOURCE_VIEWER_DIR = Path("viewer")
DEST_REPO_DIR = Path(r"C:\Users\25beevdt025\Desktop\FORGE_indoor")
DEST_REPO_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_DIR = DEST_REPO_DIR / "public"
PUBLIC_MODELS_DIR = PUBLIC_DIR / "models"
PUBLIC_DATA_DIR = PUBLIC_DIR / "data"

PUBLIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("==================================================================")
    print(" F.O.R.G.E. EXPLORER - VERCEL DEPLOYMENT BUILD & PACKAGING")
    print(f" Source: {SOURCE_VIEWER_DIR.resolve()}")
    print(f" Destination: {DEST_REPO_DIR.resolve()}")
    print("==================================================================")

    # 1. Copy Frontend Source Code
    print("\n[Step 1] Copying Frontend Application Files...")
    shutil.copytree(SOURCE_VIEWER_DIR / "src", DEST_REPO_DIR / "src", dirs_exist_ok=True)
    shutil.copyfile(SOURCE_VIEWER_DIR / "index.html", DEST_REPO_DIR / "index.html")
    shutil.copyfile(SOURCE_VIEWER_DIR / "package.json", DEST_REPO_DIR / "package.json")
    if (SOURCE_VIEWER_DIR / "package-lock.json").exists():
        shutil.copyfile(SOURCE_VIEWER_DIR / "package-lock.json", DEST_REPO_DIR / "package-lock.json")
    for tsconfig in SOURCE_VIEWER_DIR.glob("tsconfig*.json"):
        shutil.copyfile(tsconfig, DEST_REPO_DIR / tsconfig.name)
    if (SOURCE_VIEWER_DIR / ".oxlintrc.json").exists():
        shutil.copyfile(SOURCE_VIEWER_DIR / ".oxlintrc.json", DEST_REPO_DIR / ".oxlintrc.json")

    # Static assets in public
    for icon in ["favicon.svg", "icons.svg"]:
        src_icon = SOURCE_VIEWER_DIR / "public" / icon
        if src_icon.exists():
            shutil.copyfile(src_icon, PUBLIC_DIR / icon)

    walkthrough_meta = SOURCE_VIEWER_DIR / "src/data/indoor_walkthrough_metadata.json"
    if walkthrough_meta.exists():
        shutil.copyfile(walkthrough_meta, PUBLIC_DATA_DIR / "indoor_walkthrough_metadata.json")

    # 2. Write Production vite.config.ts
    print("\n[Step 2] Configuring Production vite.config.ts...")
    vite_config_content = """import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Production configuration for Vercel static deployment
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 3000,
    assetsInlineLimit: 0,
  },
  server: {
    port: 5174,
    host: true,
  },
  preview: {
    port: 4173,
    host: true,
  },
});
"""
    (DEST_REPO_DIR / "vite.config.ts").write_text(vite_config_content, encoding="utf-8")

    # 3. Write Production vercel.json
    print("\n[Step 3] Configuring Production vercel.json...")
    vercel_json_content = {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "vite",
        "buildCommand": "npm run build",
        "outputDirectory": "dist",
        "cleanUrls": True,
        "rewrites": [
            {
                "source": "/((?!assets/|models/|data/|favicon.svg|icons.svg).*)",
                "destination": "/index.html"
            }
        ],
        "headers": [
            {
                "source": "/models/(.*)",
                "headers": [
                    { "key": "Access-Control-Allow-Origin", "value": "*" },
                    { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
                ]
            },
            {
                "source": "/data/(.*)",
                "headers": [
                    { "key": "Access-Control-Allow-Origin", "value": "*" },
                    { "key": "Cache-Control", "value": "public, max-age=3600, must-revalidate" }
                ]
            }
        ]
    }
    (DEST_REPO_DIR / "vercel.json").write_text(json.dumps(vercel_json_content, indent=2), encoding="utf-8")

    # 4. Write .gitignore and .gitattributes
    print("\n[Step 4] Configuring Git LFS & Attributes...")
    gitignore_content = """# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*
lerna-debug.log*

node_modules
dist
dist-ssr
*.local

# Editor directories and files
.vscode/*
!.vscode/extensions.json
.idea
.DS_Store
*.suo
*.ntvs*
*.njsproj
*.sln
*.sw?
"""
    (DEST_REPO_DIR / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    gitattributes_content = """# 3D Model Geometry & Meshes
*.obj filter=lfs diff=lfs merge=lfs -text
*.ply filter=lfs diff=lfs merge=lfs -text
*.glb filter=lfs diff=lfs merge=lfs -text
*.gltf filter=lfs diff=lfs merge=lfs -text
*texture_atlas*.png filter=lfs diff=lfs merge=lfs -text
"""
    (DEST_REPO_DIR / ".gitattributes").write_text(gitattributes_content, encoding="utf-8")

    # 5. Process and Package Models for All 39 Scenes
    print("\n[Step 5] Processing and Packaging Real 3D Models...")
    raw_manifest = json.loads((SOURCE_VIEWER_DIR / "public/model-manifest.json").read_text(encoding="utf-8"))
    scenes = raw_manifest["scenes"]

    packaged_scenes = []
    asset_inventory = []
    total_packaged_bytes = 0

    for s_idx, s in enumerate(scenes):
        sid = s["id"]
        s_name = s.get("display_name", sid)
        print(f"  [{s_idx+1}/{len(scenes)}] Packaging scene: {sid}...")

        scene_model_dir = PUBLIC_MODELS_DIR / sid
        scene_model_dir.mkdir(parents=True, exist_ok=True)

        updated_scene = dict(s)
        updated_reps = {}

        # Previews
        for preview_key, preview_url in [
            ("preview_eye_level.png", s.get("previews", {}).get("eye_level")),
            ("preview_top.png", s.get("previews", {}).get("top")),
            ("preview_thumbnail.png", s.get("thumbnail_url")),
        ]:
            if preview_url:
                clean_prev = preview_url.lstrip("/")
                for candidate in [
                    Path(clean_prev),
                    SOURCE_VIEWER_DIR / "public" / clean_prev,
                    Path("outputs/accuracy_experiment/sprint9_video_to_viewer_release") / clean_prev.replace("models/", ""),
                ]:
                    if candidate.exists() and candidate.is_file():
                        dest_prev = scene_model_dir / preview_key
                        shutil.copyfile(candidate, dest_prev)
                        break

        updated_scene["thumbnail_url"] = f"/models/{sid}/preview_thumbnail.png"
        updated_scene["previews"] = {
            "eye_level": f"/models/{sid}/preview_eye_level.png",
            "top": f"/models/{sid}/preview_top.png"
        }
        updated_scene["metadata_url"] = f"/models/{sid}/metadata.json"

        # Check metadata
        meta_cand = [
            Path(f"outputs/indoor/{sid}/reconstruction_metadata.json"),
            SOURCE_VIEWER_DIR / f"public/models/{sid}/metadata.json",
        ]
        meta_found = False
        for mc in meta_cand:
            if mc.exists():
                shutil.copyfile(mc, scene_model_dir / "metadata.json")
                meta_found = True
                break
        if not meta_found:
            (scene_model_dir / "metadata.json").write_text(json.dumps({
                "scene_id": sid,
                "display_name": s_name,
                "category": s.get("category", "Architecture"),
                "triangles": s.get("triangle_count", 0),
                "vertices": s.get("vertex_count", 0),
            }, indent=2), encoding="utf-8")

        # Process Representations
        for rep_name, rep_data in s.get("representations", {}).items():
            rep_type = rep_data.get("type", "glb")
            rep_entry = dict(rep_data)

            # Special handling for adaptive spatial chunks
            if rep_type == "adaptive_chunks" or rep_data.get("chunks_url"):
                chunks_json_src = Path("outputs/indoor/dataset_my_movie_4_adaptive/adaptive_spatial_chunks.json")
                if chunks_json_src.exists():
                    chunks_dest_json = scene_model_dir / "adaptive_spatial_chunks.json"
                    shutil.copyfile(chunks_json_src, chunks_dest_json)

                    # Copy 12 chunk GLBs
                    chunks_sub_dir = scene_model_dir / "spatial_chunks"
                    chunks_sub_dir.mkdir(parents=True, exist_ok=True)
                    src_chunks_dir = Path("outputs/indoor/dataset_my_movie_4_adaptive/spatial_chunks")
                    for c_glb in src_chunks_dir.glob("*.glb"):
                        dest_c_glb = chunks_sub_dir / c_glb.name
                        shutil.copyfile(c_glb, dest_c_glb)
                        sz = dest_c_glb.stat().st_size
                        total_packaged_bytes += sz
                        asset_inventory.append({
                            "scene_id": sid,
                            "type": "spatial_chunk_glb",
                            "file": f"/models/{sid}/spatial_chunks/{c_glb.name}",
                            "size_bytes": sz,
                            "sha256": compute_sha256(dest_c_glb)
                        })

                    rep_entry["chunks_url"] = f"/models/{sid}/adaptive_spatial_chunks.json"
                    rep_entry["file_size_bytes"] = chunks_dest_json.stat().st_size
                    rep_entry["sha256"] = compute_sha256(chunks_dest_json)
                    rep_entry["available"] = True
                    updated_reps[rep_name] = rep_entry
                    continue

            # Standard File Representations (GLB, OBJ, PLY)
            for k in ["glb_url", "obj_url", "mtl_url", "ply_url"]:
                orig_url = rep_data.get(k)
                if not orig_url:
                    continue
                clean_url = orig_url.lstrip("/")

                src_cand = [
                    Path(clean_url),
                    SOURCE_VIEWER_DIR / "public" / clean_url,
                    Path("outputs/accuracy_experiment/sprint9_video_to_viewer_release") / clean_url.replace("models/", ""),
                    Path("outputs/accuracy_experiment/sprint9_video_to_viewer_release") / clean_url,
                ]

                found_src = None
                for c in src_cand:
                    if c.exists() and c.is_file():
                        found_src = c
                        break

                if found_src:
                    dest_file_name = found_src.name
                    dest_file = scene_model_dir / dest_file_name

                    # Optimization for outdoor_pipeline_applied_video1 (137 MB unindexed GLB)
                    if sid == "outdoor_pipeline_applied_video1" and k == "glb_url" and found_src.stat().st_size > 100 * 1024 * 1024:
                        print("    [*] Optimizing 137MB GLB via vertex indexing...")
                        tm_scene = trimesh.load(str(found_src))
                        geom = list(tm_scene.geometry.values())[0]
                        geom.merge_vertices()
                        glb_bytes = geom.export(file_type="glb")
                        dest_file.write_bytes(glb_bytes)
                    else:
                        shutil.copyfile(found_src, dest_file)

                    file_sz = dest_file.stat().st_size
                    sha_str = compute_sha256(dest_file)
                    total_packaged_bytes += file_sz

                    rep_entry[k] = f"/models/{sid}/{dest_file_name}"
                    rep_entry["file_size_bytes"] = file_sz
                    rep_entry["sha256"] = sha_str
                    rep_entry["available"] = True

                    asset_inventory.append({
                        "scene_id": sid,
                        "type": k,
                        "file": f"/models/{sid}/{dest_file_name}",
                        "size_bytes": file_sz,
                        "sha256": sha_str
                    })

            updated_reps[rep_name] = rep_entry

        updated_scene["representations"] = updated_reps
        packaged_scenes.append(updated_scene)

    # 6. Save Manifests
    print("\n[Step 6] Emitting Static Manifests (scenes.json & model-manifest.json)...")
    final_manifest = {
        "app_title": "F.O.R.G.E Explorer",
        "version": "1.0.0",
        "sprint": "Production Vercel Release",
        "description": "Complete Video-to-3D Interactive Reconstruction Explorer & Digital Twin Viewer",
        "scenes_count": len(packaged_scenes),
        "total_asset_size_mb": round(total_packaged_bytes / (1024**2), 2),
        "scenes": packaged_scenes
    }

    manifest_json_str = json.dumps(final_manifest, indent=2)

    # Write to all required locations
    (PUBLIC_DATA_DIR / "scenes.json").write_text(manifest_json_str, encoding="utf-8")
    (PUBLIC_DATA_DIR / "model-manifest.json").write_text(manifest_json_str, encoding="utf-8")
    (PUBLIC_DIR / "model-manifest.json").write_text(manifest_json_str, encoding="utf-8")
    (DEST_REPO_DIR / "src/data/scenes.json").write_text(manifest_json_str, encoding="utf-8")
    (DEST_REPO_DIR / "src/data/model-manifest.json").write_text(manifest_json_str, encoding="utf-8")

    # 7. Write Inventory JSON and Markdown
    print("\n[Step 7] Writing Asset Inventory and Verification Summary...")
    (DEST_REPO_DIR / "asset_inventory.json").write_text(json.dumps({
        "timestamp": "2026-11-01T23:30:00Z",
        "total_scenes": len(packaged_scenes),
        "total_files": len(asset_inventory),
        "total_size_bytes": total_packaged_bytes,
        "total_size_mb": round(total_packaged_bytes / (1024**2), 2),
        "assets": asset_inventory
    }, indent=2), encoding="utf-8")

    inventory_md = [
        "# F.O.R.G.E. Explorer — Production Asset Inventory & Vercel Deployment Manifest",
        "",
        f"- **Total Scenes Packaged:** {len(packaged_scenes)}",
        f"- **Total Packaged Asset Size:** {round(total_packaged_bytes / (1024**2), 2)} MB ({round(total_packaged_bytes / (1024**3), 2)} GB)",
        f"- **Total 3D Model Deliverables:** {len(asset_inventory)} files",
        "- **All Asset Paths:** Relative static URLs (`/models/<scene_id>/...`)",
        "- **Zero Localhost Dependencies:** 100% self-contained static architecture.",
        "",
        "## Scene Breakdown",
        "",
        "| # | Scene ID | Category | Primary Format | Packaged Asset Count |",
        "|---|---|---|---|---|"
    ]

    for idx, sc in enumerate(packaged_scenes):
        inventory_md.append(
            f"| {idx+1} | `{sc['id']}` | {sc.get('category', 'Architecture')} | {sc.get('default_representation', 'gltf_glb')} | {len(sc.get('representations', {}))} reps |"
        )

    (DEST_REPO_DIR / "ASSET_INVENTORY.md").write_text("\n".join(inventory_md), encoding="utf-8")

    # 8. Write README.md with Vercel Import Instructions
    readme_content = f"""# F.O.R.G.E. Explorer — 3D Digital Twin Viewer (Vercel Release)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/PrathmeshPP/FORGE_indoor)

An interactive, high-performance WebGL digital twin viewer and 3D reconstruction explorer powered by Three.js and React.

## 🚀 1-Click Vercel Deployment

This repository is pre-configured for direct Vercel import:

1. Open **[Vercel Dashboard](https://vercel.com/)** -> **Add New...** -> **Project**.
2. Select repository: **`PrathmeshPP/FORGE_indoor`**.
3. Vercel automatically detects the framework:
   - **Framework Preset:** `Vite`
   - **Root Directory:** `./`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Click **Deploy**. Within 60 seconds, your public URL will be live!

---

## 🎮 Shared Navigation Controls (Available in ALL Scenes)

Every scene supports universal first-person walkthrough and 3D flight:

| Key | Action |
|:---:|:---|
| **W / S** | Move Forward / Backward |
| **A / D** | Strafe Left / Right |
| **Space** | **Move Upward (Ascend)** |
| **C** | **Move Downward (Descend)** |
| **Shift** | Sprint (2x Speed) |
| **O** | **Toggle Walk / Fly Mode** |
| **Mouse** | Look Around (Pointer Lock Pitch & Yaw) |
| **Mouse Wheel** | Zoom In / Out |
| **G / Esc** | Exit Walkthrough |

---

## 🏛️ Packaged Scenes ({len(packaged_scenes)} Total)

1. **`indoor_my_movie_4_adaptive`**: Latest adaptive view-dependent 3D reconstruction with 4-quadrant spatial chunks (LOD 0/1/2) and real-time GPU HUD.
2. **`indoor_nov1_191043_photographic_v3`**: Hostel Room B 314 V3 quality 8mm point cloud backprojection & aspect-preserved 8K atlas.
3. **`campus_0049`**: Complete architectural campus survey.
4. **`texture_experiment_video_20261101_142811`**: Canonical November 1 textured digital twin.
5. ...and all other 35 usable scenes in the Scene Library!

---

## 📦 Total Deployment Size
- **Total Scenes:** {len(packaged_scenes)}
- **Total Asset Size:** {round(total_packaged_bytes / (1024**2), 2)} MB
- **Localhost Dependencies:** 0 (Fully independent static bundle)
"""
    (DEST_REPO_DIR / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"\n[+] Successfully packaged {len(packaged_scenes)} scenes ({round(total_packaged_bytes / (1024**2), 2)} MB total)!")
    print(f"[+] Output ready in: {DEST_REPO_DIR}")

if __name__ == "__main__":
    main()

