#!/usr/bin/env python3
"""
Dual Asset and Production Server for F.O.R.G.E Explorer.
Serves built frontend assets (viewer/dist or viewer) and routes /models/*
directly to outputs/accuracy_experiment/sprint7h_final_defect_recovery/
without copying or duplicating large 3D model files.
"""

import argparse
import http.server
import mimetypes
import os
from pathlib import Path
import socketserver
import urllib.parse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs/accuracy_experiment/sprint9_video_to_viewer_release"
DIST_DIR = PROJECT_ROOT / "viewer/dist"
DEV_DIR = PROJECT_ROOT / "viewer"

mimetypes.add_type("text/plain", ".obj")
mimetypes.add_type("text/plain", ".mtl")
mimetypes.add_type("application/octet-stream", ".ply")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("application/json", ".json")


class ForgeExplorerHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def translate_path(self, path):
        clean_path = urllib.parse.unquote(path.split("?")[0])
        if clean_path.startswith("/models/"):
            rel = clean_path[len("/models/"):]
            target = (MODELS_DIR / rel).resolve()
            # Prevent directory traversal
            if target.is_file() and str(target).startswith(str(MODELS_DIR.resolve())):
                return str(target)
            indoor_target = (PROJECT_ROOT / "outputs/indoor/room_001/final_release" / Path(rel).name).resolve()
            if indoor_target.is_file():
                return str(indoor_target)
            dist_target = (DIST_DIR / "models" / rel).resolve()
            if dist_target.is_file():
                return str(dist_target)
            return str(target)

        if clean_path.startswith("/outputs/"):
            out_target = (PROJECT_ROOT / clean_path.lstrip("/")).resolve()
            if out_target.is_file() and str(out_target).startswith(str((PROJECT_ROOT / "outputs").resolve())):
                return str(out_target)

        # Default web root: use DIST_DIR if built, otherwise DEV_DIR
        base = DIST_DIR if DIST_DIR.is_dir() else DEV_DIR
        rel = clean_path.lstrip("/")
        target = (base / rel).resolve()
        if not target.exists() and (base / "index.html").is_file():
            # SPA fallback
            return str(base / "index.html")
        return str(target)


def main():
    parser = argparse.ArgumentParser(description="F.O.R.G.E Explorer Web & Asset Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    # Change to root or dist
    os.chdir(str(PROJECT_ROOT))
    handler = ForgeExplorerHandler

    print(f"================================================================")
    print(f"  F.O.R.G.E Explorer Platform Server")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Web Root: {DIST_DIR if DIST_DIR.is_dir() else DEV_DIR}")
    print(f"================================================================")

    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
