#!/usr/bin/env python3
"""OpenGeneEdit compiler server — UI + /api/compile orchestration.

Pipeline per /api/compile request (default ``DGENE_COMPILE_MODE=circuit_synth``):

  **Circuit synthesis (default):** extract boolean intent (Gemma) → deterministic topology +
  curated iGEM parts → **truth-table verification** (:mod:`circuit_pipeline`); remaining candidate
  slots use RAG-first. Post-hoc slot RAG substitution is skipped for verified circuit rows.

  **RAG-first:** extract biological intent (Gemma) → Chroma retrieval from
  ``data/igem_dataset.jsonl`` (top-k per query) → optional deterministic **slot template** cassette
  (``slot_template_compile.py``: AND/OR/BUF promoter+RBS+CDS+terminator from per-type retrieval when
  the intent JSON parses ``gate`` / ``input_analytes``) → remaining slots use menu-constrained compiler (Gemma) → concatenate
  registry DNA only (``circuit_rag_first.py``). Post-hoc RAG substitution is skipped.

  **Legacy** (``DGENE_COMPILE_MODE=legacy``):

  1. inference backend → N candidate (thought, sequence) pairs
  2. iGEM RAG          → optional substitution from ``data/igem_dataset.jsonl`` via ChromaDB
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
until ``done``. While inference runs, poll responses may include a ``result``
object with ``"partial": true``, ``variants_ready``, and ``variants_total`` so
the UI can show the best variant among finished candidates before the job
finishes. When ``done`` is true, ``result`` is the final ranked payload
(``snapshot_id`` may be attached).

**Targeted fix:** ``POST /api/fix`` with ``original_prompt``, ``current_sequence``,
``fix_type`` (``repeats`` | ``type_iis`` | ``cai`` | ``rbs`` | ``repeats_type_iis``),
and ``candidates`` (current workspace list) runs a single-slot recompile with an
injected constraint, merges into the list, re-ranks, and returns ``fix`` metadata
(``new_candidate_id``, ``new_candidate_index``, ``still_flagged``).

**stderr:** High-frequency poll requests are hidden from the default access log (see ``DGENE_HTTP_LOG`` in ``.env.example``).
Async jobs always print ``[oge/server] job <id> · …`` when the worker starts, finishes, or fails.
"""
from __future__ import annotations

import errno
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from inference import (  # noqa: F401
    Candidate,
    compile_progress,
    get_backend,
    infer_debug_log,
    parse_thought_and_sequence,
    run_inference,
    set_compile_progress_hook,
    set_compile_stream_hook,
)
from passes import passes_to_dicts, run_passes
from ranker import attach_fidelity_scores, rank, score_candidate, scores_to_dict


# Constraint text for targeted /api/fix recompiles (issue-specific prompt injection).
FIX_PROMPTS: dict[str, str] = {
    "repeats": (
        "Minimize direct repeat sequences ≥ 18bp. Use alternative RBS variants "
        "(e.g. BBa_B0032 or BBa_B0031 instead of repeated BBa_B0034) and alternative "
        "terminators to avoid repetitive sequence."
    ),
    "type_iis": (
        "Remove all internal Type IIS restriction sites (BsaI, BbsI, BsmBI, SapI) from "
        "the sequence by applying synonymous codon substitution in CDS regions. "
        "Do not change protein sequence."
    ),
    "cai": (
        "Optimize all CDS sequences for E. coli K12 codon usage. Target CAI > 0.85. "
        "Use the most frequent codon for each amino acid."
    ),
    "rbs": (
        "Select RBS sequences with stronger Shine-Dalgarno complementarity. "
        "Target SD complementarity 5/6 or 6/6. Consider BBa_B0034 or stronger RBS variants."
    ),
    "repeats_type_iis": (
        "Fix both: minimize direct repeats ≥ 18bp AND remove all internal Type IIS sites "
        "via synonymous substitution."
    ),
}


def _assemble_fix_prompt(original_prompt: str, current_sequence: str, fix_type: str) -> str:
    constraint = FIX_PROMPTS[fix_type]
    op = original_prompt.rstrip()
    return (
        f"{op}\n\nCONSTRAINT: {constraint}\n\n"
        f"Current sequence for reference: {current_sequence}"
    )


def _pass_still_flagged(pass_dicts: list, fix_type: str) -> bool:
    by_id = {p.get("pass_id"): p for p in pass_dicts if isinstance(p, dict)}
    if fix_type == "repeats_type_iis":
        for pid in ("repeats", "type_iis"):
            p = by_id.get(pid)
            if p and p.get("status") in ("warn", "error"):
                return True
        return False
    p = by_id.get(fix_type)
    return bool(p and p.get("status") in ("warn", "error"))


_STATUS_RANK = {"error": 0, "warn": 1, "ok": 2}


def _pass_lookup(passes: list, pass_id: str) -> Optional[dict]:
    for p in passes or []:
        if isinstance(p, dict) and p.get("pass_id") == pass_id:
            return p
    return None


def _status_rank(status: Optional[str]) -> int:
    if not status:
        return -1
    return int(_STATUS_RANK.get(status, -1))


def _warn_diag_count(p: Optional[dict]) -> int:
    if not p:
        return 0
    d = p.get("diagnostics") or []
    return sum(
        1
        for x in d
        if isinstance(x, dict) and x.get("severity") in ("warn", "error")
    )


def _repeat_issue_count(p: Optional[dict]) -> Optional[float]:
    if not p:
        return None
    if p.get("status") == "ok":
        return 0.0
    s = p.get("summary") or ""
    m = re.search(r"(\d+)\s+direct repeat", s)
    if m:
        return float(m.group(1))
    n = _warn_diag_count(p)
    return float(n) if n else None


def _type_iis_issue_count(p: Optional[dict]) -> Optional[float]:
    if not p:
        return None
    if p.get("status") == "ok":
        return 0.0
    s = p.get("summary") or ""
    m = re.search(r"(\d+)\s+Type IIS", s)
    if m:
        return float(m.group(1))
    n = _warn_diag_count(p)
    return float(n) if n else None


def _metric_float(p: Optional[dict]) -> Optional[float]:
    if not p:
        return None
    m = p.get("metric")
    if m is None:
        return None
    try:
        return float(m)
    except (TypeError, ValueError):
        return None


def _fix_pass_improved(old_passes: list, new_passes: list, pass_id: str) -> bool:
    """True if the rerun is clearly better on this pass than the candidate user clicked Fix on."""

    old_p = _pass_lookup(old_passes, pass_id)
    new_p = _pass_lookup(new_passes, pass_id)
    if not new_p:
        return False
    if old_p is None:
        return _status_rank(new_p.get("status")) >= 2

    ro, rn = _status_rank(old_p.get("status")), _status_rank(new_p.get("status"))
    if rn > ro:
        return True
    if rn < ro:
        return False
    if rn >= 2:
        return True

    if pass_id in ("cai", "rbs"):
        om, nm = _metric_float(old_p), _metric_float(new_p)
        return bool(
            om is not None and nm is not None and nm > om + 1e-9
        )

    if pass_id == "repeats":
        o_c, n_c = _repeat_issue_count(old_p), _repeat_issue_count(new_p)
        return bool(
            o_c is not None and n_c is not None and n_c < o_c - 1e-9
        )

    if pass_id == "type_iis":
        o_c, n_c = _type_iis_issue_count(old_p), _type_iis_issue_count(new_p)
        return bool(
            o_c is not None and n_c is not None and n_c < o_c - 1e-9
        )

    return False


def _fix_improved_for_type(old_passes: list, new_passes: list, fix_type: str) -> bool:
    if fix_type == "repeats_type_iis":
        if not _pass_still_flagged(new_passes, fix_type):
            return True
        return _fix_pass_improved(old_passes, new_passes, "repeats") or _fix_pass_improved(
            old_passes, new_passes, "type_iis"
        )
    return _fix_pass_improved(old_passes, new_passes, fix_type)


def _find_source_passes(
    existing_rows: list,
    *,
    source_candidate_id: Optional[str],
    current_sequence: str,
) -> list:
    if source_candidate_id and str(source_candidate_id).strip():
        sid = str(source_candidate_id).strip()
        for r in existing_rows:
            if isinstance(r, dict) and str(r.get("id", "")).strip() == sid:
                ps = r.get("passes")
                return list(ps) if isinstance(ps, list) else []
    cs = "".join((current_sequence or "").upper().split())
    if cs:
        for r in existing_rows:
            if not isinstance(r, dict):
                continue
            rs = "".join(str(r.get("sequence") or "").upper().split())
            if rs == cs:
                ps = r.get("passes")
                return list(ps) if isinstance(ps, list) else []
    return []


def _fix_user_still_flagged(
    old_passes: list, new_passes: list, fix_type: str
) -> tuple[bool, bool]:
    """Whether the UI should show the harsh 'still flagged' message, and if the pass is literally ok.

    Returns (still_flagged, pass_cleared). ``pass_cleared`` means the check is no longer warn/error.
    A one-shot fix often improves CAI/repeats without reaching the compiler's ``ok`` band; that is
    *not* treated as a failed fix when ``old_passes`` shows measurable improvement.
    """

    pass_cleared = not _pass_still_flagged(new_passes, fix_type)
    if pass_cleared:
        return False, True
    improved = _fix_improved_for_type(old_passes, new_passes, fix_type) if old_passes else False
    still = not improved
    return still, False


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
_SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".design_snapshots")
_SNAPSHOT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _snapshots_enabled() -> bool:
    v = os.environ.get("DGENE_SNAPSHOTS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _save_compile_snapshot(payload: dict) -> str:
    """Persist compile JSON to disk; return new snapshot id (32 hex chars)."""

    os.makedirs(_SNAPSHOT_DIR, exist_ok=True)
    sid = uuid.uuid4().hex
    path = os.path.join(_SNAPSHOT_DIR, f"{sid}.json")
    clean = dict(payload)
    clean.pop("snapshot_id", None)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, separators=(",", ":"), ensure_ascii=False)
    return sid


def _load_compile_snapshot(sid: str) -> Optional[dict]:
    if not _SNAPSHOT_ID_RE.match(sid or ""):
        return None
    path = os.path.join(_SNAPSHOT_DIR, f"{sid}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _finalize_compile_result(result: dict) -> dict:
    """Attach ``snapshot_id`` when disk snapshots are enabled (gitignored ``.design_snapshots/``)."""

    out = dict(result)
    if not _snapshots_enabled():
        return out
    try:
        sid = _save_compile_snapshot(out)
        out["snapshot_id"] = sid
        sys.stderr.write(f"[oge/server] snapshot saved id={sid} → {_SNAPSHOT_DIR!r}\n")
    except Exception as exc:
        sys.stderr.write(f"[oge/server] snapshot save failed: {exc}\n")
    return out


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
        map_slots = rag_detail.get("map_slots")
        if not isinstance(map_slots, list):
            map_slots = extract_part_map_slots(thought)
        parts_list = rag_detail.get("parts")
        if isinstance(parts_list, list):
            pos = 1
            for idx, slot in enumerate(map_slots):
                if idx < len(parts_list):
                    p = parts_list[idx]
                    if isinstance(p, dict):
                        slot["verified"] = bool(p.get("verified"))
                        src = p.get("sequence_source")
                        if isinstance(src, str) and src:
                            slot["sequence_source"] = src
                        frag = p.get("sequence")
                        if isinstance(frag, str) and isinstance(slot, dict):
                            ln = len("".join(frag.upper().split()))
                            if ln > 0:
                                slot["start_bp"] = pos
                                slot["end_bp"] = pos + ln - 1
                                pos += ln
            tot = len("".join(final_seq.upper().split())) if final_seq else 0
            if (
                tot > 0
                and map_slots
                and isinstance(map_slots[-1], dict)
                and map_slots[-1].get("end_bp") is not None
                and map_slots[-1]["end_bp"] != tot
            ):
                map_slots[-1]["end_bp"] = tot
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


def _extract_ordered_part_names(rag: dict) -> list:
    o = rag.get("ordered_part_names")
    if isinstance(o, list) and o:
        return [str(x).strip() for x in o if str(x).strip()]
    trace = rag.get("assembly_trace")
    if isinstance(trace, list):
        names = []
        for t in trace:
            if not isinstance(t, dict):
                continue
            if t.get("ok") is False:
                continue
            n = t.get("normalized_name") or t.get("part_name")
            if n:
                names.append(str(n).strip())
        if names:
            return names
    parts = rag.get("parts")
    if isinstance(parts, list):
        return [
            str(p.get("part_name") or "").strip()
            for p in parts
            if isinstance(p, dict) and p.get("part_name")
        ]
    return []


def _attach_design_qa(row: dict, *, user_prompt: str) -> None:
    """Regulatory consistency lint + optional Gemma reviewer (``DGENE_EXPERT_REVIEW``)."""

    rag = row.get("rag")
    if not isinstance(rag, dict):
        return
    names = _extract_ordered_part_names(rag)
    if not names:
        return
    try:
        from design_expert_lint import lint_ordered_construct

        rag["expert_lint"] = lint_ordered_construct(names)
    except Exception as exc:
        rag["expert_lint"] = {"error": str(exc), "grade": "unclear"}
    try:
        from expert_review import expert_gemma_review

        rev = expert_gemma_review(user_prompt or "", names)
        if rev is not None:
            rag["expert_review"] = rev
    except Exception as exc:
        rag["expert_review"] = {"error": str(exc)}


def _ranked_row_from_candidate(
    cand: Candidate,
    idx_1: int,
    n_total: int,
    *,
    user_prompt: str = "",
) -> dict:
    """Single candidate through RAG + static passes (+ stderr mirror)."""

    try:
        from igem_rag import rag_debug_log as _rag_resp_log
    except ImportError:

        def _rag_resp_log(_msg: str) -> None:
            return None

    pre = getattr(cand, "rag_first_detail", None)
    pipe = pre.get("pipeline") if isinstance(pre, dict) else None
    if pipe in ("rag_first", "circuit_synth", "slot_template"):
        label = (
            "topology-verified linear plasmid (ori+CmR+MCS+cassettes; no post substitution)"
            if pipe == "circuit_synth"
            else "slot-template vector (cassette in ori+CmR+MCS scaffold; no post substitution)"
            if pipe == "slot_template"
            else "RAG-first assembly (no post substitution)"
        )
        compile_progress(
            f"compile · {label} · candidate {idx_1}/{n_total} ({cand.candidate_id})…"
        )
        final_seq = cand.sequence
        rag_detail = dict(pre)
    else:
        compile_progress(
            f"compile · iGEM RAG · candidate {idx_1}/{n_total} ({cand.candidate_id})…"
        )
        final_seq, rag_detail = _apply_rag_substitution(
            cand.thought, cand.sequence, candidate_id=cand.candidate_id
        )
    compile_progress(f"compile · passes · candidate {idx_1}/{n_total} ({cand.candidate_id})…")
    passes = run_passes(final_seq)
    scores = score_candidate(passes, len(final_seq))
    _rag_resp_log(
        f"server: candidate {cand.candidate_id} API payload `sequence` len={len(final_seq)} bp "
        f"(after RAG substitution — same string the frontend renders)"
    )
    row = {
        "id": cand.candidate_id,
        "thought": cand.thought,
        "sequence": final_seq,
        "strategy": cand.strategy,
        "strategy_name": cand.strategy_name,
        "passes": passes_to_dicts(passes),
        "scores": scores_to_dict(scores),
        "rag": rag_detail,
    }
    attach_fidelity_scores(row["scores"], prompt=user_prompt, row=row)
    _attach_design_qa(row, user_prompt=user_prompt)
    return row


def _compile(
    prompt: str, n: int = 4, *, user_prompt_for_alignment: Optional[str] = None
) -> dict:
    from circuit_pipeline import compile_hybrid_variants
    from circuit_rag_first import compile_mode, rag_first_configured, run_rag_first_variants

    align_prompt = (
        user_prompt_for_alignment if user_prompt_for_alignment is not None else prompt
    )
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
    mode = compile_mode()
    if mode == "circuit_synth" and rag_first_configured():
        compile_progress(
            f"compile · circuit_synth (verified when applicable) + RAG-first padding · {n} slot(s)…"
        )
        candidates = compile_hybrid_variants(prompt, n, progress_cb=compile_progress)
    elif mode == "rag_first" and rag_first_configured():
        compile_progress(
            f"compile · RAG-first pipeline · {n} variants (shared intent + menu)…"
        )
        candidates = run_rag_first_variants(prompt, n, progress_cb=compile_progress)
    elif mode in ("circuit_synth", "rag_first") and not rag_first_configured():
        sys.stderr.write(
            "[oge/server] "
            f"DGENE_COMPILE_MODE={mode} but hybrid LLM unavailable "
            "(need GEMINI_API_KEY / GOOGLE_API_KEY or local GGUF via DGENE_GGUF_PATH) — "
            "falling back to legacy channel compile.\n"
        )
        compile_progress(
            "compile · WARN · hosted Gemma API key required for circuit_synth / RAG-first — "
            "using legacy inference+RAG…"
        )
        compile_progress(f"compile · inference ({n} candidates)…")
        candidates = backend.generate(prompt, n=n)
    else:
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

    out = [
        _ranked_row_from_candidate(
            c, i + 1, len(candidates), user_prompt=align_prompt
        )
        for i, c in enumerate(candidates)
    ]

    compile_progress("compile · ranking · Pareto front…")
    ranked = rank(out)
    best_id = ranked[0]["id"] if ranked else None
    compile_progress("compile · done")

    return _finalize_compile_result({
        "candidates": ranked,
        "best_id": best_id,
        "model": getattr(backend, "name", "unknown"),
        "prompt": prompt,
    })


def _run_fix_compile(
    original_prompt: str,
    current_sequence: str,
    fix_type: str,
    existing_rows: list,
    *,
    source_candidate_id: Optional[str] = None,
) -> dict:
    """Single-candidate recompile with a constraint, merged into existing candidates."""

    if fix_type not in FIX_PROMPTS:
        raise ValueError(f"unsupported fix_type: {fix_type!r}")
    op = original_prompt.strip()
    if not op:
        raise ValueError("original_prompt is required")
    seq = "".join((current_sequence or "").upper().split())
    if not seq:
        raise ValueError("current_sequence is required")
    if not isinstance(existing_rows, list):
        raise ValueError("candidates must be a list")

    assembled = _assemble_fix_prompt(op, seq, fix_type)
    compile_out = _compile(assembled, n=1, user_prompt_for_alignment=op)
    new_list = compile_out.get("candidates") or []
    if not new_list:
        raise RuntimeError("Fix compile produced no candidates")
    new_cand = dict(new_list[0])
    new_cand["fix_badge"] = "Fixed"

    merged: list = [dict(r) for r in existing_rows] + [new_cand]
    ranked = rank(merged)
    best_id = ranked[0]["id"] if ranked else None
    new_id = new_cand["id"]
    new_idx = next((i + 1 for i, c in enumerate(ranked) if c.get("id") == new_id), 0)
    old_passes = _find_source_passes(
        existing_rows,
        source_candidate_id=source_candidate_id,
        current_sequence=seq,
    )
    still, pass_cleared = _fix_user_still_flagged(
        old_passes, new_cand.get("passes") or [], fix_type
    )

    return _finalize_compile_result({
        "candidates": ranked,
        "best_id": best_id,
        "model": compile_out.get("model"),
        "prompt": op,
        "fix": {
            "fix_type": fix_type,
            "new_candidate_id": new_id,
            "new_candidate_index": new_idx,
            "still_flagged": still,
            "pass_cleared": pass_cleared,
        },
    })


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
    sys.stderr.write(f"[oge/server {ts}] {msg}\n")
    sys.stderr.flush()


def _job_lifecycle(job_id: str, msg: str) -> None:
    """Always-on high-signal line so long jobs and failures are visible in the terminal."""
    sys.stderr.write(f"[oge/server] job {job_id} · {msg}\n")
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
        compile_progress("compile · resolving backend…")
        backend = get_backend()
        bk = getattr(backend, "name", "?")
        mid = getattr(backend, "model_id", None) or getattr(backend, "gguf_filename", None)
        if mid:
            compile_progress(f"compile · backend={bk} · {mid}")
        else:
            compile_progress(f"compile · backend={bk}")
        infer_debug_log(
            f"/api/compile job start backend={getattr(backend, 'name', '?')} n={n} "
            f"prompt_chars={len(prompt)}"
        )
        t_inf = time.perf_counter()
        from circuit_pipeline import compile_hybrid_variants_iter
        from circuit_rag_first import (
            compile_mode as _compile_mode_job,
            rag_first_configured,
            run_rag_first_variants_iter,
        )

        rows: list = []
        job_mode = _compile_mode_job()
        if job_mode == "circuit_synth" and rag_first_configured():
            compile_progress(
                f"compile · circuit_synth + RAG-first streaming · {n} slot(s)…"
            )
            cand_iter = compile_hybrid_variants_iter(
                prompt, n, progress_cb=compile_progress
            )
        elif job_mode == "rag_first" and rag_first_configured():
            compile_progress(
                f"compile · RAG-first pipeline · {n} variants (shared intent + menu)…"
            )
            cand_iter = run_rag_first_variants_iter(
                prompt, n, progress_cb=compile_progress
            )
        elif job_mode in ("circuit_synth", "rag_first") and not rag_first_configured():
            sys.stderr.write(
                f"[oge/server] DGENE_COMPILE_MODE={job_mode} but hybrid LLM unavailable "
                "(need GEMINI_API_KEY / GOOGLE_API_KEY or local GGUF via DGENE_GGUF_PATH) — "
                "falling back to legacy channel compile.\n"
            )
            compile_progress(
                "compile · WARN · hosted Gemma API key required — using legacy inference+RAG…"
            )
            compile_progress(f"compile · inference ({n} candidates)…")
            gen_fn = getattr(backend, "generate_iter", None)
            if callable(gen_fn):
                cand_iter = gen_fn(prompt, n=n)
            else:
                cand_iter = iter(backend.generate(prompt, n=n))
        else:
            compile_progress(f"compile · inference ({n} candidates)…")
            gen_fn = getattr(backend, "generate_iter", None)
            if callable(gen_fn):
                cand_iter = gen_fn(prompt, n=n)
            else:
                cand_iter = iter(backend.generate(prompt, n=n))

        result = None
        model_name = getattr(backend, "name", "unknown")
        for k, cand in enumerate(cand_iter, start=1):
            rows.append(_ranked_row_from_candidate(cand, k, n, user_prompt=prompt))
            ranked = rank(rows)
            best_id = ranked[0]["id"] if ranked else None
            if k < n:
                compile_progress(
                    f"compile · partial result · {k}/{n} variants · best so far `{best_id}` "
                    "(more generating…)"
                )
                infer_debug_log(
                    f"/api/compile job partial {k}/{n} best_id={best_id!r} "
                    f"wall={(time.perf_counter() - t_inf):.1f}s"
                )
                payload = {
                    "candidates": ranked,
                    "best_id": best_id,
                    "model": model_name,
                    "prompt": prompt,
                    "partial": True,
                    "variants_ready": k,
                    "variants_total": n,
                }
                with _JOBS_LOCK:
                    job_partial = _COMPILE_JOBS.get(job_id)
                    if job_partial:
                        job_partial["result"] = payload
            else:
                infer_debug_log(
                    f"/api/compile job inference+RAG complete in {(time.perf_counter() - t_inf):.1f}s "
                    f"candidates={len(rows)}"
                )
                compile_progress(
                    f"compile · inference finished in {(time.perf_counter() - t_inf):.1f}s · "
                    f"ranking ({len(rows)} seqs)…"
                )
                compile_progress("compile · ranking · Pareto front…")
                compile_progress("compile · done")
                result = _finalize_compile_result({
                    "candidates": ranked,
                    "best_id": best_id,
                    "model": model_name,
                    "prompt": prompt,
                })

        if result is None:
            if not rows:
                raise RuntimeError("Compile produced no candidates")
            ranked = rank(rows)
            best_id = ranked[0]["id"] if ranked else None
            infer_debug_log(
                f"/api/compile job finished with {len(rows)}/{n} candidate(s) "
                f"(iterator exhausted before requested count)"
            )
            compile_progress(
                f"compile · inference finished · {len(rows)}/{n} candidate(s) · ranking…"
            )
            compile_progress("compile · ranking · Pareto front…")
            compile_progress("compile · done")
            result = _finalize_compile_result({
                "candidates": ranked,
                "best_id": best_id,
                "model": model_name,
                "prompt": prompt,
            })

        elapsed = time.perf_counter() - t_job
        nc = len(result.get("candidates") or [])
        _job_lifecycle(
            job_id,
            f"OK · {elapsed:.1f}s · {nc} candidate(s) — poll will return result",
        )
        _server_debug(
            f"job {job_id} incremental compile finished in {elapsed:.1f}s "
            f"best_id={result.get('best_id')!r}"
        )
        with _JOBS_LOCK:
            job_done = _COMPILE_JOBS.get(job_id)
            if job_done:
                job_done["result"] = result
                job_done["done"] = True
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
    server_version = "OpenGeneEditCompiler/2.0"

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
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON body"}, 400)
            return

        if path == "/api/fix":
            original_prompt = str(payload.get("original_prompt", ""))
            current_sequence = str(payload.get("current_sequence", ""))
            fix_type = str(payload.get("fix_type", "")).strip()
            existing = payload.get("candidates")
            if not isinstance(existing, list):
                existing = []
            source_candidate_id = str(payload.get("source_candidate_id", "")).strip() or None
            try:
                result = _run_fix_compile(
                    original_prompt,
                    current_sequence,
                    fix_type,
                    existing,
                    source_candidate_id=source_candidate_id,
                )
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                self._json({"error": str(exc)}, 500)
                return
            self._json(result)
            return

        if path != "/api/compile":
            self.send_error(404)
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
        if path == "/api/snapshot":
            qs = parse_qs(urlparse(self.path).query)
            sid = (qs.get("id") or [""])[0].strip().lower()
            if not sid:
                self._json({"error": "missing id"}, 400)
                return
            if not _snapshots_enabled():
                self._json({"error": "snapshots_disabled"}, 503)
                return
            data = _load_compile_snapshot(sid)
            if data is None:
                self._json({"error": "not_found"}, 404)
                return
            payload = dict(data)
            payload["snapshot_id"] = sid
            self._json(payload)
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
    if not os.path.isdir(WEB_ROOT):
        print(f"Missing web root: {WEB_ROOT}", file=sys.stderr)
        sys.exit(1)

    # Railway / Render / Fly inject PORT — bind exactly that port (router never follows +1…+31).
    # Omit PORT locally to keep legacy behaviour: try 8765 then bump until free.
    port_raw = os.environ.get("PORT")
    httpd = None
    port = 8765
    if port_raw is not None and str(port_raw).strip() != "":
        port = int(port_raw)
        try:
            httpd = ThreadingHTTPServer(("", port), Handler)
        except OSError as exc:
            print(
                f"[server] Fatal: could not bind PORT={port} (all interfaces): {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        base_port = 8765
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

    public = (os.environ.get("RAILWAY_PUBLIC_DOMAIN") or "").strip()
    if public:
        print(f"OpenGeneEdit listening on 0.0.0.0:{port} · https://{public}/", file=sys.stderr)
    else:
        print(
            f"OpenGeneEdit listening on 0.0.0.0:{port}/ (local: http://127.0.0.1:{port}/)",
            file=sys.stderr,
        )

    httpd.serve_forever()


if __name__ == "__main__":
    main()
