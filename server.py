#!/usr/bin/env python3
"""Serve the web UI and /api/compile for mock or future Gemma inference (stdlib only)."""
from __future__ import annotations

import errno
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from inference import parse_thought_and_sequence, run_mock_inference

WEB_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".woff2": "font/woff2",
}


def _compile(prompt: str) -> dict:
    """Run compiler backend.

    Replace ``run_mock_inference`` with a Gemma (or vLLM / OpenAI) call that returns
    the same training format::

        <|channel>thought
        ...reasoning...
        <channel|>
        DNA...

    then keep ``parse_thought_and_sequence`` unchanged.
    """
    raw = run_mock_inference(prompt)
    thought, sequence = parse_thought_and_sequence(raw)
    return {
        "thought": thought,
        "sequence": sequence,
        "raw": raw,
        "model": "mock",
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "DGeneCompiler/1.0"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/compile":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON body"}, 400)
            return
        prompt = str(payload.get("prompt", ""))
        try:
            result = _compile(prompt)
        except Exception as exc:
            self._json({"error": str(exc)}, 500)
            return
        self._json(result)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", ""):
            path = "/index.html"
        rel = path.lstrip("/")
        fs_path = os.path.normpath(os.path.join(WEB_ROOT, rel))
        web_norm = os.path.normpath(WEB_ROOT)
        if not fs_path.startswith(web_norm) or not os.path.isfile(fs_path):
            self.send_error(404)
            return
        ext = os.path.splitext(fs_path)[1]
        ctype = MIME.get(ext, "application/octet-stream")
        with open(fs_path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"{self.address_string()} — {fmt % args}\n")


def main() -> None:
    base_port = int(os.environ.get("PORT", "8765"))
    if not os.path.isdir(WEB_ROOT):
        print(f"Missing web root: {WEB_ROOT}", file=sys.stderr)
        sys.exit(1)

    httpd = None
    port = base_port
    for offset in range(32):
        candidate = base_port + offset
        try:
            httpd = HTTPServer(("", candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", -1)):
                raise
            if offset == 31:
                print(
                    f"No free port in {base_port}–{base_port + 31}. "
                    "Set PORT=... or stop the process using that range.",
                    file=sys.stderr,
                )
                sys.exit(1)

    if port != base_port:
        print(
            f"Port {base_port} in use; listening on {port} instead.",
            file=sys.stderr,
        )
    print(f"DGene compiler UI: http://127.0.0.1:{port}/")
    print("API: POST /api/compile  JSON {{\"prompt\": \"...\"}}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
