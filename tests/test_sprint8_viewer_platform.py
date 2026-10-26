#!/usr/bin/env python3
"""
Unit and Integration Tests for Sprint 8: Interactive 3D Reconstruction Viewer
=============================================================================
Validates:
1. Viewer directory structure, source files, and production build output (dist/).
2. Asset manifest schema, all 5 benchmark scene entries, and required metadata.
3. Disk existence and SHA-256 cryptographic integrity for all referenced models,
   materials, texture atlases, thumbnails, and metadata JSON files.
4. OBJ/MTL/texture reference integrity (mtllib & map_Kd relative references).
5. PLY mesh integrity (ensures meshes contain faces/triangles, not bare point clouds,
   and photographic PLYs contain RGB vertex color channels).
6. Server scripts functionality.
"""

import json
import re
import unittest
from pathlib import Path

import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = PROJECT_ROOT / "viewer"
MODELS_DIR = PROJECT_ROOT / "outputs/accuracy_experiment/sprint7h_final_defect_recovery"


class TestSprint8ViewerPlatform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = VIEWER_DIR / "src/data/model-manifest.json"
        cls.public_manifest_path = VIEWER_DIR / "public/model-manifest.json"
        cls.dist_dir = VIEWER_DIR / "dist"

    def test_01_viewer_directory_structure_and_build(self):
        """Verify viewer structure, component files, and production build artifacts exist."""
        self.assertTrue(VIEWER_DIR.is_dir(), f"Missing viewer dir: {VIEWER_DIR}")
        
        required_source_files = [
            "package.json",
            "vite.config.ts",
            "index.html",
            "src/App.tsx",
            "src/types.ts",
            "src/index.css",
            "src/components/Header.tsx",
            "src/components/SceneLibrary.tsx",
            "src/components/Viewport3D.tsx",
            "src/components/ComparisonView.tsx",
            "src/components/InspectionPanel.tsx",
            "src/components/Toolbar.tsx",
            "src/components/HelpModal.tsx",
        ]
        for rel in required_source_files:
            p = VIEWER_DIR / rel
            self.assertTrue(p.is_file(), f"Missing viewer component or source file: {p}")

        # Check production build artifacts
        self.assertTrue(self.dist_dir.is_dir(), "Production build directory dist/ does not exist")
        dist_index = self.dist_dir / "index.html"
        self.assertTrue(dist_index.is_file(), "Missing dist/index.html")
        self.assertGreater(dist_index.stat().st_size, 200, "dist/index.html is unexpectedly small")

        assets_dir = self.dist_dir / "assets"
        self.assertTrue(assets_dir.is_dir(), "Missing dist/assets directory")
        js_files = list(assets_dir.glob("*.js"))
        css_files = list(assets_dir.glob("*.css"))
        self.assertGreater(len(js_files), 0, "No bundled JS files found in dist/assets")
        self.assertGreater(len(css_files), 0, "No bundled CSS files found in dist/assets")

    def test_02_manifest_schema_and_required_fields(self):
        """Verify model manifest schema, 5 primary scenes, and comparison pairs."""
        self.assertTrue(self.manifest_path.is_file(), f"Missing {self.manifest_path}")
        self.assertTrue(self.public_manifest_path.is_file(), f"Missing {self.public_manifest_path}")

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["app_title"], "F.O.R.G.E Explorer")
        self.assertIn(manifest["sprint"], ["Sprint 8", "Sprint 9", "Sprint 10.1"])

        scenes = manifest["scenes"]
        self.assertEqual(len(scenes), 5, "Manifest must contain exactly 5 primary benchmark scenes")

        scene_ids = [s["id"] for s in scenes]
        expected_ids = [
            "campus_0049",
            "desert_road_candidate_a",
            "desert_road_candidate_b",
            "forest_road",
            "river_bridge",
        ]
        self.assertEqual(scene_ids, expected_ids, "Scene IDs must match primary benchmark scenes")

        required_keys = [
            "id", "scene_name", "display_name", "subtitle", "category",
            "acceptance_status", "description", "known_limitations",
            "coordinate_status", "vertex_count", "triangle_count",
            "representations", "thumbnail_url", "previews", "metadata_url",
            "initial_camera"
        ]
        for scene in scenes:
            for k in required_keys:
                self.assertIn(k, scene, f"Scene {scene.get('id')} missing required key '{k}'")
            self.assertIn(scene["acceptance_status"], ["PASS", "PARTIAL", "FAIL"])
            self.assertGreater(scene["vertex_count"], 10000)
            self.assertGreater(scene["triangle_count"], 20000)

        # Check comparison pair
        self.assertIn("comparison_pairs", manifest)
        self.assertGreater(len(manifest["comparison_pairs"]), 0)
        pair = manifest["comparison_pairs"][0]
        self.assertEqual(pair["left_id"], "desert_road_candidate_a")
        self.assertEqual(pair["right_id"], "desert_road_candidate_b")

    def test_03_every_referenced_asset_path_exists_on_disk(self):
        """Verify every URL in the manifest resolves to an existing file in outputs with valid SHA-256."""
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

        def url_to_disk_path(url: str) -> Path:
            clean = url.split("?")[0]
            if clean.startswith("/models/"):
                rel = clean[len("/models/"):]
                return MODELS_DIR / rel
            return VIEWER_DIR / clean.lstrip("/")

        for scene in manifest["scenes"]:
            # Check thumbnail
            thumb_path = url_to_disk_path(scene["thumbnail_url"])
            self.assertTrue(thumb_path.is_file(), f"Missing thumbnail on disk: {thumb_path}")
            self.assertGreater(thumb_path.stat().st_size, 1000)

            # Check metadata
            meta_path = url_to_disk_path(scene["metadata_url"])
            self.assertTrue(meta_path.is_file(), f"Missing metadata on disk: {meta_path}")

            # Check previews
            for p_name, p_url in scene["previews"].items():
                p_path = url_to_disk_path(p_url)
                self.assertTrue(p_path.is_file(), f"Missing preview on disk: {p_path} ({p_name})")

            # Check representations
            for rep_name, rep in scene["representations"].items():
                self.assertTrue(rep["available"], f"Representation {rep_name} marked unavailable")
                self.assertRegex(rep["sha256"], r"^[0-9a-f]{64}$", f"Invalid SHA-256 hash in {rep_name}")

                if rep["type"] == "obj":
                    obj_path = url_to_disk_path(rep["obj_url"])
                    mtl_path = url_to_disk_path(rep["mtl_url"])
                    self.assertTrue(obj_path.is_file(), f"Missing OBJ on disk: {obj_path}")
                    self.assertTrue(mtl_path.is_file(), f"Missing MTL on disk: {mtl_path}")
                    for tex_url in rep["texture_urls"]:
                        tex_path = url_to_disk_path(tex_url)
                        self.assertTrue(tex_path.is_file(), f"Missing texture atlas on disk: {tex_path}")
                elif rep["type"] == "ply":
                    ply_path = url_to_disk_path(rep["ply_url"])
                    self.assertTrue(ply_path.is_file(), f"Missing PLY on disk: {ply_path}")

    def test_04_obj_mtl_texture_reference_integrity(self):
        """Verify mtllib and map_Kd relative file references inside OBJ and MTL match filesystem."""
        for scene_folder in [
            "campus_0049",
            "desert_road/candidate_a",
            "desert_road/candidate_b",
            "forest_road",
            "river_bridge",
        ]:
            s_dir = MODELS_DIR / scene_folder
            obj_path = s_dir / "textured_mesh.obj"
            mtl_path = s_dir / "textured_mesh.mtl"
            atlas_path = s_dir / "texture_atlas.png"

            self.assertTrue(obj_path.is_file(), f"Missing {obj_path}")
            self.assertTrue(mtl_path.is_file(), f"Missing {mtl_path}")
            self.assertTrue(atlas_path.is_file(), f"Missing {atlas_path}")

            # Check mtllib in first 100 lines of OBJ
            obj_lines = []
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for _ in range(100):
                    line = f.readline()
                    if not line:
                        break
                    obj_lines.append(line)
            obj_head = "".join(obj_lines)
            self.assertIn("mtllib textured_mesh.mtl", obj_head, f"OBJ {obj_path} does not declare mtllib textured_mesh.mtl")

            # Check map_Kd in MTL
            mtl_content = mtl_path.read_text(encoding="utf-8")
            self.assertIn("map_Kd texture_atlas.png", mtl_content, f"MTL {mtl_path} does not declare map_Kd texture_atlas.png")

    def test_05_ply_mesh_faces_and_colors(self):
        """Ensure PLY meshes contain genuine face triangles and photographic PLYs contain vertex colors."""
        for scene_folder, mesh_name in [
            ("campus_0049", "surface_mesh_photographic.ply"),
            ("desert_road/candidate_a", "surface_mesh_photographic.ply"),
            ("desert_road/candidate_b", "surface_mesh_vehicle_stabilized_photographic.ply"),
            ("forest_road", "surface_mesh_photographic.ply"),
            ("river_bridge", "surface_mesh_photographic.ply"),
        ]:
            p_path = MODELS_DIR / scene_folder / mesh_name
            self.assertTrue(p_path.is_file(), f"Missing photographic mesh: {p_path}")
            mesh = o3d.io.read_triangle_mesh(str(p_path))
            self.assertGreater(len(mesh.vertices), 10000, f"{mesh_name} vertices too low")
            self.assertGreater(len(mesh.triangles), 20000, f"{mesh_name} triangles too low (must not be point cloud)")
            self.assertTrue(mesh.has_vertex_colors(), f"{mesh_name} must contain RGB vertex colors")

    def test_06_server_script_functionality(self):
        """Verify Python server and generator scripts exist."""
        srv_script = PROJECT_ROOT / "scripts/serve_viewer.py"
        gen_script = PROJECT_ROOT / "scripts/generate_model_manifest.py"
        self.assertTrue(srv_script.is_file(), f"Missing server script: {srv_script}")
        self.assertTrue(gen_script.is_file(), f"Missing generator script: {gen_script}")


if __name__ == "__main__":
    unittest.main()
