"""Inference layer for DGene — **Gemma 4 only.**

Backends:

  • **Hosted Gemma 4**: ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` / ``DGENE_GOOGLE_API_KEY``
    plus ``DGENE_GEMINI_MODEL`` (e.g. ``gemma-4-31b-it``). Calls the Google
    Generative Language ``generateContent`` API (stdlib ``urllib`` only).

  • **Local Gemma GGUF**: ``DGENE_GGUF_PATH`` pointing at a ``.gguf`` file plus
    ``llama-cpp-python``.

``DGENE_INFERENCE``:

  • ``auto`` — use API keys if set, otherwise GGUF file if valid; missing both
    is a startup error (**no fallback**).

  • ``gemini`` / ``hosted`` — API only.

  • ``gguf`` / ``local`` — GGUF only.

Malformed model output fails the request (**no skipping** failed candidates).

``.env`` in this package directory is loaded on import (existing environment
variables are not overwritten).

Set ``DGENE_DEBUG=1`` or ``DGENE_GEMINI_DEBUG=1`` for stderr traces (HTTP timings,
retries, candidate threads). Restart the server after changing ``.env``.
"""
from __future__ import annotations

import concurrent.futures
import errno
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# Compile hooks must be process-global: parallel Gemma calls run on ThreadPoolExecutor
# workers — threading.local() hooks would be invisible there (no live stream / progress).
_HOOK_LOCK = threading.Lock()
_COMPILE_PROGRESS_CB: Optional[Callable[[str], None]] = None
_COMPILE_STREAM_CB: Optional[Callable[[str, str], None]] = None


def set_compile_progress_hook(cb: Optional[Callable[[str], None]]) -> None:
    """Register callback for compile progress lines (cleared with None)."""
    global _COMPILE_PROGRESS_CB
    with _HOOK_LOCK:
        _COMPILE_PROGRESS_CB = cb


def compile_progress(msg: str) -> None:
    """Emit one progress line if a hook is installed (no-op otherwise)."""
    with _HOOK_LOCK:
        fn = _COMPILE_PROGRESS_CB
    if not fn:
        return
    try:
        fn(msg)
    except Exception:
        pass


def set_compile_stream_hook(cb: Optional[Callable[[str, str], None]]) -> None:
    """Callback(stream_id, full_text_so_far) — server stores per-candidate streams."""
    global _COMPILE_STREAM_CB
    with _HOOK_LOCK:
        _COMPILE_STREAM_CB = cb


def compile_stream_update(stream_id: str, full_text: str) -> None:
    with _HOOK_LOCK:
        fn = _COMPILE_STREAM_CB
    if not fn:
        return
    try:
        fn(stream_id, full_text)
    except Exception:
        pass


def _load_dotenv() -> None:
    """Minimal .env reader; does not override existing os.environ."""
    path = os.path.join(_MODULE_DIR, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_dotenv()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: str
    thought: str
    sequence: str
    strategy: str = ""
    strategy_name: str = ""
    raw: str = ""


# ---------------------------------------------------------------------------
# Helpers shared by all backends
# ---------------------------------------------------------------------------


def _canonical_raw(thought: str, sequence: str) -> str:
    return "<|channel>thought\n{}\n<channel|>\n{}".format(
        thought.strip(),
        "".join(sequence.strip().upper().split()),
    )


def _strip_markdown_fences(text: str) -> str:
    """Trim optional ``` wrapper often added despite instructions."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    while lines and lines[-1].strip() in ("```", ""):
        lines.pop()
    return "\n".join(lines).strip()


def _min_parse_dna_length() -> int:
    raw = os.environ.get("DGENE_MIN_PARSE_DNA_LEN", "").strip()
    if raw:
        try:
            return max(6, min(5000, int(raw)))
        except ValueError:
            pass
    return 12


def _extract_dna_after_marker(rest: str, *, min_len: int) -> str:
    """Take the longest DNA substring after `<channel|>`; allows trailing prose."""
    if not rest or not rest.strip():
        return ""
    flat = re.sub(r"\s+", "", rest)
    best = ""
    for m in re.finditer(r"[ACGTNacgtn]+", flat):
        seg = m.group(0).upper()
        if len(seg) >= min_len and len(seg) > len(best):
            best = seg
    if len(best) >= min_len:
        return best
    letters = "".join(c for c in rest.upper() if c in "ACGTN")
    return letters if len(letters) >= min_len else ""


def parse_thought_and_sequence(model_output: str) -> Tuple[str, str]:
    """Extract thought + DNA from the canonical training tag format.

    Format::

        <|channel>thought
        ...reasoning...
        <channel|>
        DNA...

    Tolerates leading preamble, markdown fences, and non-DNA text after the sequence.
    """
    raw_in = model_output.strip()
    idx = raw_in.find("<|channel>thought")
    if idx > 0:
        raw_in = raw_in[idx:]
    raw = _strip_markdown_fences(raw_in)
    min_dna = _min_parse_dna_length()

    channel_pat = re.compile(
        r"<\|channel\>thought\s*(.*?)\s*<channel\|>",
        re.DOTALL,
    )
    match = channel_pat.search(raw)
    if match:
        thought = match.group(1).strip()
        tail = raw[match.end() :].strip()
        sequence = _extract_dna_after_marker(tail, min_len=min_dna)
        if thought and sequence:
            return thought, sequence

    strict = re.compile(
        r"<\|channel\>thought\s*(.*?)\s*<channel\|>\s*([ACGTNacgtn\s]+)\s*$",
        re.DOTALL,
    )
    m2 = strict.search(raw)
    if m2:
        thought = m2.group(1).strip()
        sequence = re.sub(r"\s+", "", m2.group(2)).upper()
        if thought and sequence:
            return thought, sequence

    if "<|channel>thought" in raw and "<channel|>" in raw:
        thought_part, seq_part = raw.split("<channel|>", 1)
        thought = thought_part.replace("<|channel>thought", "", 1).strip()
        sequence = _extract_dna_after_marker(seq_part, min_len=min_dna)
        if thought and sequence:
            return thought, sequence

    raise ValueError("Could not parse thought and DNA sequence from model output.")


_FORMAT_RETRY_SUFFIX = (
    "\n\n**Format correction (required):** Your reply must start with `<|channel>thought` "
    "as the very first characters—no title, no markdown fence, no preamble. "
    "End reasoning with `<channel|>` on its own line. "
    "After that line output only A/C/G/T nucleotides (≥12 bp), then stop."
)


def _seed_for(prompt: str, idx: int) -> int:
    h = 0
    for ch in prompt:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h ^ (idx * 2654435761)) & 0xFFFFFFFF


class InferenceConfigurationError(RuntimeError):
    """Missing GEMINI_* key, GGUF path, or contradictory DGENE_INFERENCE."""


# ---------------------------------------------------------------------------
# Gemini API — hosted Gemma 4 via Generative Language REST
# ---------------------------------------------------------------------------

_GEMINI_API_BASE = (
    os.environ.get("DGENE_GEMINI_API_BASE", "").strip()
    or "https://generativelanguage.googleapis.com/v1beta"
)


def _pick_google_api_key() -> Optional[str]:
    """First non-empty key from GEMINI_API_KEY, GOOGLE_API_KEY, or DGENE_GOOGLE_API_KEY."""
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "DGENE_GOOGLE_API_KEY"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def _gemini_prompt_template(user_design_brief: str) -> str:
    brief = user_design_brief.strip()
    return (
        "You are DGene, a synthetic-biology DNA compiler. Read the user's circuit brief and "
        "output one DNA construct solution.\n\n"
        "**User brief**\n"
        f"{brief}\n\n"
        "**Output format (required)**\n"
        "- Start immediately with `<|channel>thought` on its own opening line.\n"
        "- After one paragraph of reasoning with concrete parts and trade-offs, emit the closing "
        "`<channel|>` marker on its own line.\n"
        "- Below that marker, emit one continuous nucleotide DNA string using ONLY letters "
        "`A`, `C`, `G`, and `T` (no FASTA headers, numbering, whitespace, or line breaks inside "
        "the DNA string).\n\n"
        "The first characters of your reply must be `<|channel>thought`. Nothing before them.\n"
    )


def _gemini_completion_text(payload: dict) -> str:
    cands = payload.get("candidates") or []
    if not cands:
        fb = payload.get("promptFeedback")
        raise RuntimeError(f"Gemini returned no candidates — promptFeedback: {fb!r}")

    parts = (cands[0].get("content") or {}).get("parts") or ()
    blobs: List[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("text") is not None:
            blobs.append(str(p["text"]))
    joined = "".join(blobs).strip("\n")
    if not joined:
        raise RuntimeError("Gemini returned empty text in candidate parts.")
    return joined


def _gemini_http_timeout_seconds() -> float:
    """Per-request socket timeout (Gemini can stream/thinking for a long time)."""
    raw = os.environ.get("DGENE_GEMINI_HTTP_TIMEOUT", "").strip()
    if raw:
        try:
            return max(30.0, float(raw))
        except ValueError:
            pass
    return 600.0


def _gemini_env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def infer_debug_enabled() -> bool:
    """True when ``DGENE_DEBUG`` or ``DGENE_GEMINI_DEBUG`` is set (stderr traces)."""
    return _gemini_env_bool("DGENE_DEBUG", False) or _gemini_env_bool("DGENE_GEMINI_DEBUG", False)


def infer_debug_log(line: str) -> None:
    """Log one line to stderr when infer_debug_enabled()."""
    if not infer_debug_enabled():
        return
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[dgene/infer {ts}] {line}\n")
    sys.stderr.flush()


def _gemini_error_is_transient(msg: str) -> bool:
    """429/rate limits, timeouts, and typical flaky HTTP/network failures."""
    m = msg.lower()
    if "429" in m or "resource exhausted" in m or "rate" in m:
        return True
    if "500" in m and ("internal" in m or "server" in m):
        return True
    if "502" in m or "503" in m or "504" in m:
        return True
    if "timed out" in m or "timeout" in m:
        return True
    if "temporarily unavailable" in m or "try again" in m:
        return True
    if "connection reset" in m or "broken pipe" in m:
        return True
    if "network is unreachable" in m or "name or service not known" in m:
        return True
    return False


def _gemini_post(
    api_key: str, model_id: str, body: dict, *, debug_ctx: str = ""
) -> dict:
    """POST generateContent and return decoded JSON."""
    base = _GEMINI_API_BASE.rstrip("/")
    qs = urllib.parse.urlencode({"key": api_key})
    url = f"{base}/models/{model_id}:generateContent?{qs}"
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    timeout = _gemini_http_timeout_seconds()
    tag = f"{debug_ctx} " if debug_ctx else ""
    if infer_debug_enabled():
        gen = (body.get("generationConfig") or {}) if isinstance(body, dict) else {}
        mo = gen.get("maxOutputTokens")
        infer_debug_log(
            f"{tag}generateContent → model={model_id!r} api_base={base!r} "
            f"body_bytes={len(encoded)} timeout_s={timeout:.0f} max_out_tokens={mo!r}"
        )
    t0 = time.perf_counter()
    hb_sec_raw = os.environ.get("DGENE_GEMINI_HTTP_HEARTBEAT_SEC", "15").strip()
    try:
        hb_interval = float(hb_sec_raw)
    except ValueError:
        hb_interval = 15.0
    stop_heartbeat = threading.Event()

    def _http_heartbeat() -> None:
        if hb_interval <= 0:
            return
        n = 0
        while not stop_heartbeat.wait(timeout=hb_interval):
            n += 1
            elapsed = time.perf_counter() - t0
            label = debug_ctx or "http"
            compile_progress(
                f"gemma · generateContent still waiting… {elapsed:.0f}s so far "
                f"(per-attempt cap {timeout:.0f}s · {label} · pulse {n})"
            )

    hb_thread = (
        threading.Thread(target=_http_heartbeat, daemon=True)
        if hb_interval > 0
        else None
    )
    if hb_thread is not None:
        hb_thread.start()
    try:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            msg = err_body or str(exc.reason)
            if err_body:
                try:
                    err_json = json.loads(err_body)
                    err_obj = err_json.get("error")
                    if isinstance(err_obj, dict) and err_obj.get("message"):
                        msg = str(err_obj["message"])
                except json.JSONDecodeError:
                    pass
            infer_debug_log(
                f"{tag}HTTP {exc.code} after {elapsed_ms:.0f}ms — {msg[:500]}"
                + ("…" if len(msg) > 500 else "")
            )
            ml = msg.lower()
            hint = ""
            if exc.code in (400, 401, 403) and (
                "api key" in ml
                or "permission" in ml
                or "authentication" in ml
                or "request had invalid authentication" in ml
            ):
                hint = (
                    " — Fix: set GEMINI_API_KEY (or GOOGLE_API_KEY) to a key from "
                    "Google AI Studio (https://aistudio.google.com/apikey), restart this server, "
                    "and ensure `.env` has no stray quotes or spaces around the key."
                )
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {msg}{hint}") from None
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            infer_debug_log(
                f"{tag}{type(exc).__name__} after {elapsed_ms:.0f}ms: {exc!s}"
            )
            raise
    finally:
        stop_heartbeat.set()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    infer_debug_log(
        f"{tag}OK {elapsed_ms:.0f}ms response_chars={len(raw)} "
        f"thread={threading.current_thread().name!r}"
    )
    return json.loads(raw)


def _gemini_chunk_text(payload: dict) -> str:
    """Text delta from one streamed GenerateContentResponse JSON object."""
    cands = payload.get("candidates") or []
    if not cands:
        return ""
    parts = (cands[0].get("content") or {}).get("parts") or ()
    blobs: List[str] = []
    for p in parts:
        if isinstance(p, dict) and p.get("text") is not None:
            blobs.append(str(p["text"]))
    return "".join(blobs)


def _gemini_stream_collect(
    api_key: str,
    model_id: str,
    body: dict,
    *,
    debug_ctx: str = "",
) -> str:
    """POST streamGenerateContent (SSE), accumulate text, emit compile_stream_update."""
    base = _GEMINI_API_BASE.rstrip("/")
    qs = urllib.parse.urlencode({"key": api_key, "alt": "sse"})
    url = f"{base}/models/{model_id}:streamGenerateContent?{qs}"
    encoded = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        method="POST",
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "text/event-stream",
        },
    )
    timeout = _gemini_http_timeout_seconds()
    tag = f"{debug_ctx} " if debug_ctx else ""
    if infer_debug_enabled():
        infer_debug_log(
            f"{tag}streamGenerateContent → model={model_id!r} timeout_s={timeout:.0f}"
        )
    t0 = time.perf_counter()
    accumulated = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            line_buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                line_buf += chunk
                while b"\n" in line_buf:
                    raw_line, line_buf = line_buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", errors="replace").strip().replace("\r", "")
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload_raw = line[5:].strip()
                    if payload_raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload_raw)
                    except json.JSONDecodeError:
                        continue
                    err_obj = obj.get("error")
                    if isinstance(err_obj, dict):
                        msg = str(err_obj.get("message") or err_obj)
                        code = err_obj.get("code") or err_obj.get("status")
                        raise RuntimeError(f"Gemini stream error {code}: {msg}")
                    piece = _gemini_chunk_text(obj)
                    if piece:
                        accumulated += piece
                        compile_stream_update(debug_ctx or "stream", accumulated)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = err_body or str(exc.reason)
        if err_body:
            try:
                err_json = json.loads(err_body)
                err_o = err_json.get("error")
                if isinstance(err_o, dict) and err_o.get("message"):
                    msg = str(err_o["message"])
            except json.JSONDecodeError:
                pass
        infer_debug_log(f"{tag}stream HTTP {exc.code} after {elapsed_ms:.0f}ms — {msg[:400]}")
        raise RuntimeError(f"Gemini stream HTTP {exc.code}: {msg}") from None

    elapsed_ms = (time.perf_counter() - t0) * 1000
    infer_debug_log(
        f"{tag}stream OK {elapsed_ms:.0f}ms chars={len(accumulated)} "
        f"thread={threading.current_thread().name!r}"
    )
    if not accumulated.strip():
        raise RuntimeError(
            "Gemini stream ended with no text — model blocked or empty candidates."
        )
    return accumulated


def _apply_optional_thinking_config(gen_cfg: dict) -> None:
    lvl = os.environ.get("DGENE_GEMINI_THINKING_LEVEL", "").strip()
    if lvl:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": lvl}


def _gemini_generate_text_with_retries(
    api_key: str, model_id: str, body: dict, *, debug_ctx: str = ""
) -> str:
    delay = 1.0
    last_err: Optional[RuntimeError] = None
    tag = f"{debug_ctx} " if debug_ctx else ""
    with _HOOK_LOCK:
        _have_stream = _COMPILE_STREAM_CB is not None
    use_sse = _gemini_env_bool("DGENE_GEMINI_STREAM", True) and _have_stream
    for attempt in range(8):
        infer_debug_log(f"{tag}retry attempt {attempt + 1}/8")
        if attempt > 0:
            compile_progress(f"gemma · HTTP retry {attempt + 1}/8 (backoff)…")
        try:
            if use_sse:
                text = _gemini_stream_collect(
                    api_key, model_id, body, debug_ctx=debug_ctx
                )
            else:
                payload = _gemini_post(api_key, model_id, body, debug_ctx=debug_ctx)
                text = _gemini_completion_text(payload)
            cap = 200
            tail = "…" if len(text) > cap else ""
            infer_debug_log(
                f"{tag}completion chars={len(text)} head={text[:cap]!r}{tail}"
            )
            return text
        except (RuntimeError, OSError, urllib.error.URLError) as exc:
            last_err = exc if isinstance(exc, RuntimeError) else RuntimeError(str(exc))
            msg = str(last_err)
            transient = _gemini_error_is_transient(msg)
            if isinstance(exc, OSError) and exc.errno in (
                errno.ETIMEDOUT,
                errno.ECONNRESET,
                errno.EPIPE,
                errno.ECONNREFUSED,
            ):
                transient = True
            infer_debug_log(
                f"{tag}caught {type(exc).__name__}: transient={transient} msg={msg[:300]}"
            )
            if attempt < 7 and transient:
                jitter = random.uniform(0.0, 0.5)
                sleep_s = delay + jitter
                infer_debug_log(f"{tag}backing off {sleep_s:.2f}s (delay was {delay:.1f}s)")
                time.sleep(sleep_s)
                delay = min(delay * 2, 120.0)
                continue
            raise last_err from None

    raise last_err or RuntimeError("Gemma API request failed.")  # pragma: no cover


def _gemini_generate_single(
    api_key: str,
    model_id: str,
    prompt_text: str,
    temperature: float,
    max_out: int,
    *,
    debug_ctx: str = "",
) -> str:
    gen_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_out}
    _apply_optional_thinking_config(gen_cfg)
    body: dict = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": gen_cfg,
    }
    return _gemini_generate_text_with_retries(
        api_key, model_id, body, debug_ctx=debug_ctx
    )


def generate_text_gemma4(
    user_message: str,
    *,
    system_message: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> str:
    """Single text completion via hosted Gemma 4 (configured API key + DGENE_GEMINI_MODEL)."""

    key = _pick_google_api_key()
    if not key:
        raise InferenceConfigurationError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY to call hosted Gemma 4."
        )
    mid = _default_gemma_hosted_model_id()
    gen_cfg: dict = {
        "temperature": temperature,
        "maxOutputTokens": (
            max_output_tokens
            if max_output_tokens is not None
            else int(os.environ.get("DGENE_GEMINI_MAX_OUTPUT", "8192"))
        ),
    }
    _apply_optional_thinking_config(gen_cfg)
    body: dict = {
        "contents": [{"parts": [{"text": user_message}]}],
        "generationConfig": gen_cfg,
    }
    if system_message:
        body["systemInstruction"] = {"parts": [{"text": system_message}]}

    text = _gemini_generate_text_with_retries(
        key, mid, body, debug_ctx="generate_text_gemma4"
    ).strip()
    if not text:
        raise RuntimeError("Gemma 4 returned empty text.")
    return text


class GeminiBackend:
    """Hosted Gemma via Google Generative Language API (Gemini-compatible endpoint)."""

    backend_kind = "hosted"

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
        self.name = "Gemma-4"

    def generate(self, prompt: str, n: int = 4) -> List[Candidate]:
        template = _gemini_prompt_template(prompt)
        max_out = int(os.environ.get("DGENE_GEMINI_MAX_OUTPUT", "8192"))
        temps = [0.4, 0.55, 0.7, 0.85, 1.0, 1.1]

        def one(i: int) -> Candidate:
            ctx = f"cand_{i}"
            temperature = temps[i % len(temps)]
            compile_progress(
                f"gemma · candidate {i + 1}/{n} · generateContent · T={temperature:g} · "
                f"model={self.model_id!r}"
            )
            infer_debug_log(
                f"{ctx} start T={temps[i % len(temps)]} thread={threading.current_thread().name!r}"
            )
            tmpl_retry = template + _FORMAT_RETRY_SUFFIX
            thought = ""
            sequence = ""
            text = ""
            for attempt in range(2):
                use_template = template if attempt == 0 else tmpl_retry
                if attempt > 0:
                    compile_progress(
                        f"gemma · candidate {i + 1}/{n} · retry · stricter format reminder…"
                    )
                    infer_debug_log(f"{ctx} parse retry attempt 2")
                t_c0 = time.perf_counter()
                text = _gemini_generate_single(
                    self.api_key,
                    self.model_id,
                    use_template,
                    temperature,
                    max_out,
                    debug_ctx=ctx if attempt == 0 else f"{ctx}_retry",
                )
                infer_debug_log(
                    f"{ctx} raw received in {(time.perf_counter() - t_c0):.1f}s, parsing…"
                )
                compile_progress(
                    f"gemma · candidate {i + 1}/{n} · HTTP OK in "
                    f"{(time.perf_counter() - t_c0):.1f}s · parsing…"
                )
                try:
                    thought, sequence = parse_thought_and_sequence(text)
                    break
                except ValueError as exc:
                    if attempt == 0:
                        infer_debug_log(f"{ctx} parse failed, retrying once: {exc!s}")
                        continue
                    infer_debug_log(f"{ctx} parse FAILED: {exc!s}")
                    raise RuntimeError(
                        f"Hosted Gemma output for candidate {i} is not parseable "
                        "(expected `<|channel>thought … <channel|>` then a DNA string of A/C/G/T). "
                        "Try again, or set DGENE_MIN_PARSE_DNA_LEN=8 if the design is very short."
                    ) from exc
            infer_debug_log(
                f"{ctx} done seq_len={len(sequence)} thought_chars={len(thought)}"
            )
            return Candidate(
                candidate_id=f"cand_{i}",
                thought=thought,
                sequence=sequence,
                strategy=f"T{temperature:g}",
                strategy_name=f"Gemma hosted (T={temperature:g})",
                raw=_canonical_raw(thought, sequence),
            )

        parallel = _gemini_env_bool("DGENE_GEMINI_PARALLEL", True) and n > 1
        compile_progress(
            f"gemma · hosted · {n} candidates · parallel={parallel} · "
            f"wall≈slowest call · max_out_tokens={max_out}"
        )
        infer_debug_log(
            f"GeminiBackend.generate n={n} parallel={parallel} model={self.model_id!r}"
        )
        if not parallel:
            return [one(i) for i in range(n)]

        try:
            max_workers = int(os.environ.get("DGENE_GEMINI_MAX_WORKERS", "4").strip() or "4")
        except ValueError:
            max_workers = 4
        max_workers = max(1, min(max_workers, n))
        infer_debug_log(f"ThreadPoolExecutor max_workers={max_workers}")
        g0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            out = list(pool.map(one, range(n)))
        infer_debug_log(f"GeminiBackend.generate finished in {(time.perf_counter() - g0):.1f}s")
        return out


class GGUFBackend:
    """Wraps a quantized Gemma 4 fine-tune via llama-cpp-python.

    Activated automatically when ``DGENE_GGUF_PATH`` env var points at a
    valid .gguf file. N candidates are generated by re-sampling at different
    temperatures + seeds — same prompt, different decodings.
    """

    backend_kind = "fine_tuned"

    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "GGUFBackend requires llama-cpp-python. Install with:\n"
                "  python3 -m pip install --upgrade llama-cpp-python\n"
                "Then restart the server."
            ) from exc

        self.model_path = os.path.abspath(model_path)
        self.gguf_filename = os.path.basename(self.model_path)
        self.name = "Gemma-4 FT"
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=int(os.environ.get("DGENE_GGUF_CTX", "4096")),
            n_gpu_layers=int(os.environ.get("DGENE_GGUF_GPU_LAYERS", "-1")),
            verbose=False,
        )

    def _format_prompt(self, user_prompt: str) -> str:
        # Mirrors the training format in gemma_train.jsonl: instruction + thought channel.
        return (
            "<|user|>\n"
            f"{user_prompt}\n"
            "<|assistant|>\n"
            "<|channel>thought\n"
        )

    def generate(self, prompt: str, n: int = 4) -> List[Candidate]:
        formatted = self._format_prompt(prompt)
        out: List[Candidate] = []
        # Temperature ladder for diversity across candidates.
        temps = [0.4, 0.7, 0.9, 1.1]
        compile_progress(f"gguf · local · {n} candidates · {self.gguf_filename}")
        for i in range(n):
            compile_progress(
                f"gguf · candidate {i + 1}/{n} · sample T={temps[i % len(temps)]}…"
            )
            res = self._llm(
                formatted,
                max_tokens=int(os.environ.get("DGENE_GGUF_MAX_TOKENS", "1024")),
                temperature=temps[i % len(temps)],
                top_p=0.95,
                top_k=40,
                seed=_seed_for(prompt, i),
                stop=["</s>", "<|user|>"],
            )
            text = res["choices"][0]["text"]
            full = formatted + text
            try:
                thought, sequence = parse_thought_and_sequence(full)
            except ValueError as exc:
                raise RuntimeError(
                    f"Local GGUF Gemma candidate {i} is not parseable "
                    "(expected `<|channel>thought … <channel|>` then ATCG)."
                ) from exc
            out.append(Candidate(
                candidate_id=f"cand_{i}",
                thought=thought,
                sequence=sequence,
                strategy=f"sample_T{temps[i % len(temps)]}",
                strategy_name=f"Gemma sample (T={temps[i % len(temps)]})",
                raw=full,
            ))
        return out


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

_INFER_BACKEND: Optional[object] = None
_BACKEND_LOCK = threading.Lock()
_BACKEND_FAILURE: Optional[BaseException] = None


_GEMINI_MODES = frozenset({"gemini", "api", "google", "hosted"})
_GGUF_MODES = frozenset({"gguf", "local", "finetuned"})


def _default_gemma_hosted_model_id() -> str:
    raw = os.environ.get("DGENE_GEMINI_MODEL", "").strip()
    mid = raw or "gemma-4-31b-it"
    if mid.lower().startswith("gemini"):
        print(
            f"[inference] WARNING: DGENE_GEMINI_MODEL={mid!r} looks like Gemini, "
            "not Gemma — set to a Gemma 4 id such as gemma-4-31b-it or gemma-4-26b-a4b-it.",
            file=sys.stderr,
        )
    return mid


def _require_gguf_file(path_raw: str) -> str:
    if not path_raw.strip():
        raise InferenceConfigurationError("DGENE_GGUF_PATH must be set to a valid .gguf file.")
    p = os.path.abspath(path_raw.strip())
    if not os.path.isfile(p):
        raise InferenceConfigurationError(f"DGENE_GGUF_PATH is not a file: {path_raw!r}")
    return p


def _create_inference_backend() -> object:
    mode = os.environ.get("DGENE_INFERENCE", "auto").strip().lower() or "auto"
    gguf_raw = os.environ.get("DGENE_GGUF_PATH", "").strip()
    api_key = _pick_google_api_key()

    if mode in _GEMINI_MODES:
        if not api_key:
            raise InferenceConfigurationError(
                "DGENE_INFERENCE requests hosted Gemma but GEMINI_API_KEY / GOOGLE_API_KEY is unset."
            )
        model_id = _default_gemma_hosted_model_id()
        print(f"[inference] GeminiBackend model={model_id}", file=sys.stderr)
        return GeminiBackend(api_key, model_id)

    if mode in _GGUF_MODES:
        gguf_path = _require_gguf_file(gguf_raw)
        print(f"[inference] GGUFBackend {gguf_path}", file=sys.stderr)
        return GGUFBackend(gguf_path)

    # auto
    if api_key:
        model_id = _default_gemma_hosted_model_id()
        print(f"[inference] GeminiBackend (auto) model={model_id}", file=sys.stderr)
        return GeminiBackend(api_key, model_id)
    if gguf_raw:
        gguf_path = _require_gguf_file(gguf_raw)
        print(f"[inference] GGUFBackend (auto) {gguf_path}", file=sys.stderr)
        return GGUFBackend(gguf_path)

    raise InferenceConfigurationError(
        "DGene requires hosted Gemma 4 (GEMINI_API_KEY / GOOGLE_API_KEY + DGENE_GEMINI_MODEL) "
        "or local Gemma GGUF (DGENE_GGUF_PATH). No mock/offline inference is compiled in."
    )


def get_backend() -> object:
    """Return the process-wide inference backend singleton."""

    global _INFER_BACKEND, _BACKEND_FAILURE

    if _INFER_BACKEND is not None:
        return _INFER_BACKEND
    if _BACKEND_FAILURE is not None:
        raise _BACKEND_FAILURE

    with _BACKEND_LOCK:
        if _INFER_BACKEND is not None:
            return _INFER_BACKEND
        if _BACKEND_FAILURE is not None:
            raise _BACKEND_FAILURE
        try:
            _INFER_BACKEND = _create_inference_backend()
        except Exception as exc:
            _BACKEND_FAILURE = exc
            raise
        return _INFER_BACKEND


# ---------------------------------------------------------------------------
# Single-shot compiler output (Streamlit)
# ---------------------------------------------------------------------------


def run_inference(prompt: str) -> str:
    cands = get_backend().generate(prompt, n=1)
    return cands[0].raw
