#!/usr/bin/env python3
"""
Lightweight QA Server for in-browser WebGL GLB verification.
Serves repository root on port 8089 with proper CORS and MIME types.
"""

import http.server
import mimetypes
from pathlib import Path
import socketserver
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/json", ".json")

class QAServerHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    port = 8089
    handler = QAServerHandler
    print(f"Starting QA Server on http://127.0.0.1:{port} from {PROJECT_ROOT}...")
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    main()
