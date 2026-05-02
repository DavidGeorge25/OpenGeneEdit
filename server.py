#!/usr/bin/env python3
"""DGene compiler server — UI + /api/compile orchestration.

Pipeline per /api/compile request:

  1. inference backend → N candidate (thought, sequence) pairs
  2. iGEM RAG          → optional substitution from ``igem_dataset.jsonl`` via ChromaDB
                         (see ``igem_rag.py``); candidates gain a ``rag`` field
  3. compiler passes   → per-candidate diagnostics + score-bearing metrics
  4. ranker            → composite score + Pareto front
  5. response          → candidates[] sorted by composite, with `is_pareto`

The backend is resolved once inside ``inference.get_backend()`` (lazy singleton):

  • Set ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` (+ optional ``DGENE_GEMINI_MODEL`` such as
    ``gemma-4-31b-it``) for hosted Gemma 4 via the Generative Language API.

  • Or ``DGENE_GGUF_PATH=/path/to/model.gguf`` for local quantized Gemma (``llama-cpp-python``).

  • See ``inference.py`` for ``DGENE_INFERENCE=auto|gemini|gguf``.

This process uses ``ThreadingHTTPServer`` so a long-running ``/api/compile``
(e.g. several Gemma API round-trips) does not block other tabs or reloads.

**Live progress (UI):** ``POST /api/compile`` with ``"progress": true`` returns
``202`` and ``{"job_id": "..."}``. Poll ``GET /api/compile/status?job_id=...``
until ``done``; each response includes a growing ``lines`` trace from Gemma,
passes, and ranking.

**stderr:** High-frequency poll requests are hidden from the default access log (see ``DGENE_HTTP_LOG`` in ``.env.example``).
Async jobs always print ``[dgene/server] job <id> · …`` when the worker starts, finishes, or fails.
"""
from __future__ import annotations

import errno
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from inference import (  # noqa: F401
    compile_progress,
    get_backend,
    infer_debug_log,
    parse_thought_and_sequence,
    run_inference,
    set_compile_progress_hook,
    set_compile_stream_hook,
)
from passes import passes_to_dicts, run_passes
from ranker import rank, score_candidate, scores_to_dict


def _health_dict() -> dict:
    bk = get_backend()
    data = {
        "status": "ok",
        "model": getattr(bk, "name", "unknown"),
        "backend_kind": getattr(bk, "backend_kind", "unknown"),
    }
    mid = getattr(bk, "model_id", None)
    if getattr(bk, "backend_kind", None) == "hosted" and mid:
        data["api_model_id"] = mid
    gf = getattr(bk, "gguf_filename", None)
    if getattr(bk, "backend_kind", None) == "fine_tuned" and gf:
        data["gguf_file"] = gf
    return data


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


# ---------------------------------------------------------------------------
# Compile pipeline
# ---------------------------------------------------------------------------


def _apply_rag_substitution(thought: str, sequence: str, *, candidate_id: str = ""):
    """Lazy-import ``igem_rag`` so the server boots without ML deps installed."""

    try:
        from igem_rag import apply_rag_substitution as rag_fn
        from igem_rag import extract_part_map_slots

        final_seq, rag_detail = rag_fn(
            thought,
            sequence,
            progress_cb=compile_progress,
            log_context=candidate_id,
        )
        rag_detail = dict(rag_detail)
        map_slots = extract_part_map_slots(thought)
        parts_list = rag_detail.get("parts")
        if isinstance(parts_list, list):
            for idx, slot in enumerate(map_slots):
                if idx < len(parts_list):
                    p = parts_list[idx]
                    if isinstance(p, dict):
                        slot["verified"] = bool(p.get("verified"))
                        src = p.get("sequence_source")
                        if isinstance(src, str) and src:
                            slot["sequence_source"] = src
        rag_detail["map_slots"] = map_slots
        return final_seq, rag_detail
    except Exception as exc:
        flat = "".join((sequence or "").upper().split())
        try:
            from igem_rag import extract_part_map_slots

            return flat, {
                "enabled": False,
                "error": str(exc),
                "map_slots": extract_part_map_slots(thought),
            }
        except Exception:
            return flat, {"enabled": False, "error": str(exc)}


def _compile(prompt: str, n: int = 4) -> dict:
    compile_progress("compile · resolving backend…")
    backend = get_backend()
    bk = getattr(backend, "name", "?")
    mid = getattr(backend, "model_id", None) or getattr(backend, "gguf_filename", None)
    if mid:
        compile_progress(f"compile · backend={bk} · {mid}")
    else:
        compile_progress(f"compile · backend={bk}")
    infer_debug_log(
        f"/api/compile start backend={getattr(backend, 'name', '?')} n={n} "
        f"prompt_chars={len(prompt)}"
    )
    t0 = time.perf_counter()
    compile_progress(f"compile · inference ({n} candidates)…")
    candidates = backend.generate(prompt, n=n)
    infer_debug_log(
        f"/api/compile inference done in {(time.perf_counter() - t0):.1f}s "
        f"candidates={len(candidates)}"
    )
    compile_progress(
        f"compile · inference finished in {(time.perf_counter() - t0):.1f}s · "
        f"static passes ({len(candidates)} seqs)…"
    )

    try:
        from igem_rag import rag_debug_log as _rag_resp_log
    except ImportError:

        def _rag_resp_log(_msg: str) -> None:
            return None

    out = []
    for idx, cand in enumerate(candidates):
        compile_progress(f"compile · iGEM RAG · candidate {idx + 1}/{len(candidates)}…")
        final_seq, rag_detail = _apply_rag_substitution(
            cand.thought, cand.sequence, candidate_id=cand.candidate_id
        )
        compile_progress(f"compile · passes · candidate {idx + 1}/{len(candidates)}…")
        passes = run_passes(final_seq)
        scores = score_candidate(passes, len(final_seq))
        out.append({
            "id": cand.candidate_id,
            "thought": cand.thought,
            "sequence": final_seq,
            "strategy": cand.strategy,
            "strategy_name": cand.strategy_name,
            "passes": passes_to_dicts(passes),
            "scores": scores_to_dict(scores),
            "rag": rag_detail,
        })
        _rag_resp_log(
            f"server: candidate {cand.candidate_id} API payload `sequence` len={len(final_seq)} bp "
            f"(after RAG substitution — same string the frontend renders)"
        )

    compile_progress("compile · ranking · Pareto front…")
    ranked = rank(out)
    best_id = ranked[0]["id"] if ranked else None
    compile_progress("compile · done")

    return {
        "candidates": ranked,
        "best_id": best_id,
        "model": getattr(backend, "name", "unknown"),
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# Async compile jobs (live progress for the UI)
# ---------------------------------------------------------------------------

_JOBS_LOCK = threading.Lock()
_COMPILE_JOBS: dict[str, dict] = {}
_MAX_JOB_LINES = 500
_STREAM_PREVIEW_CHARS = 65536


def _job_append_line(job_id: str, line: str) -> None:
    with _JOBS_LOCK:
        job = _COMPILE_JOBS.get(job_id)
        if not job:
            return
        job["lines"].append(line)
        if len(job["lines"]) > _MAX_JOB_LINES:
            job["lines"] = job["lines"][-(_MAX_JOB_LINES - 50) :]


def _job_set_stream(job_id: str, stream_id: str, text: str) -> None:
    if len(text) > _STREAM_PREVIEW_CHARS:
        text = "…(truncated for live panel)\n" + text[-_STREAM_PREVIEW_CHARS:]
    with _JOBS_LOCK:
        job = _COMPILE_JOBS.get(job_id)
        if not job:
            return
        job.setdefault("streams", {})[stream_id] = text


def _server_debug(msg: str) -> None:
    """Verbose compile/job tracing when DGENE_SERVER_DEBUG is set."""
    if (os.environ.get("DGENE_SERVER_DEBUG") or "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[dgene/server {ts}] {msg}\n")
    sys.stderr.flush()


def _job_lifecycle(job_id: str, msg: str) -> None:
    """Always-on high-signal line so long jobs and failures are visible in the terminal."""
    sys.stderr.write(f"[dgene/server] job {job_id} · {msg}\n")
    sys.stderr.flush()


def _run_compile_job(job_id: str, prompt: str, n: int) -> None:
    def push(msg: str) -> None:
        _job_append_line(job_id, f"{time.strftime('%H:%M:%S')}  {msg}")

    def push_stream(stream_id: str, text: str) -> None:
        _job_set_stream(job_id, stream_id, text)

    set_compile_progress_hook(push)
    set_compile_stream_hook(push_stream)
    t_job = time.perf_counter()
    try:
        try:
            from igem_rag import set_rag_debug_mirror

            set_rag_debug_mirror(
                lambda ln: _job_append_line(
                    job_id, f"{time.strftime('%H:%M:%S')}  [rag] {ln}"
                )
            )
        except ImportError:
            pass
        push("job · started")
        _job_lifecycle(job_id, "thread started (async compile)")
        _server_debug(
            f"job {job_id} prompt_len={len(prompt)} n={n} — entering inference+RAG+passes"
        )
        result = _compile(prompt, n=n)
        elapsed = time.perf_counter() - t_job
        nc = len(result.get("candidates") or [])
        _job_lifecycle(
            job_id,
            f"OK · {elapsed:.1f}s · {nc} candidate(s) — poll will return result",
        )
        _server_debug(
            f"job {job_id} _compile returned in {elapsed:.1f}s best_id={result.get('best_id')!r}"
        )
        with _JOBS_LOCK:
            job = _COMPILE_JOBS.get(job_id)
            if job:
                job["result"] = result
                job["done"] = True
            else:
                _job_lifecycle(job_id, "WARN: job dict missing after compile (race?)")
    except Exception as exc:
        elapsed = time.perf_counter() - t_job
        _job_lifecycle(
            job_id,
            f"FAILED after {elapsed:.1f}s · {type(exc).__name__}: {exc}",
        )
        traceback.print_exc(file=sys.stderr)
        with _JOBS_LOCK:
            job = _COMPILE_JOBS.get(job_id)
            if job:
                job["error"] = str(exc)
                job["done"] = True
            else:
                _job_lifecycle(job_id, "WARN: job dict missing; error not stored in job")
    finally:
        set_compile_progress_hook(None)
        set_compile_stream_hook(None)
        try:
            from igem_rag import set_rag_debug_mirror

            set_rag_debug_mirror(None)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "DGeneCompiler/2.0"

    def log_message(self, format: str, *args) -> None:
        """Suppress poll/health spam; set DGENE_HTTP_LOG=all to log every request."""
        try:
            rendered = format % args if args else format
        except Exception:
            rendered = str(format)
        mode = (os.environ.get("DGENE_HTTP_LOG") or "").strip().lower()
        if mode not in ("all", "verbose", "1", "true", "yes"):
            if "/api/compile/status" in rendered or "/api/health" in rendered:
                return
        super().log_message(format, *args)

    def _write_body(self, body: bytes) -> None:
        """Write response body; ignore client disconnect (avoids secondary tracebacks)."""
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

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
        self._write_body(body)

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
        n = int(payload.get("n", 4))
        n = max(1, min(8, n))
        want_progress = bool(payload.get("progress"))
        if want_progress:
            job_id = uuid.uuid4().hex
            with _JOBS_LOCK:
                _COMPILE_JOBS[job_id] = {
                    "lines": [],
                    "streams": {},
                    "done": False,
                    "result": None,
                    "error": None,
                }
            thread = threading.Thread(
                target=_run_compile_job,
                args=(job_id, prompt, n),
                daemon=True,
            )
            thread.start()
            self._json({"job_id": job_id}, 202)
            return
        try:
            result = _compile(prompt, n=n)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            self._json({"error": str(exc)}, 500)
            return
        self._json(result)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/compile/status":
            qs = parse_qs(urlparse(self.path).query)
            job_id = (qs.get("job_id") or [""])[0].strip()
            if not job_id:
                self._json({"error": "missing job_id"}, 400)
                return
            with _JOBS_LOCK:
                job = _COMPILE_JOBS.get(job_id)
            if not job:
                self._json({"error": "unknown job_id"}, 404)
                return
            payload: dict = {
                "done": job["done"],
                "lines": list(job["lines"]),
            }
            streams = job.get("streams") or {}
            if streams:
                payload["streams"] = dict(streams)
            err = job.get("error")
            if err:
                payload["error"] = err
            res = job.get("result")
            if res is not None:
                payload["result"] = res
            self._json(payload)
            if job["done"]:
                with _JOBS_LOCK:
                    _COMPILE_JOBS.pop(job_id, None)
            return
        if path == "/api/health":
            self._json(_health_dict())
            return
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
        self._write_body(data)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def main() -> None:
    base_port = int(os.environ.get("PORT", "8765"))
    if not os.path.isdir(WEB_ROOT):
        print(f"Missing web root: {WEB_ROOT}", file=sys.stderr)
        sys.exit(1)

    httpd = None
    port = base_port
    for offset in range(32):
        cand = base_port + offset
        try:
            httpd = ThreadingHTTPServer(("", cand), Handler)
            port = cand
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
        print(f"Port {base_port} in use; listening on {port} instead.", file=sys.stderr)

    # Pre-warm the backend so first compile isn't slow on cold start.
    try:
        b = get_backend()
        print(f"[server] Backend ready: {getattr(b, 'name', 'unknown')}", file=sys.stderr)
    except Exception as exc:
        print(f"[server] Backend init failed at boot: {exc}", file=sys.stderr)

    print(f"DGene compiler UI: http://127.0.0.1:{port}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
