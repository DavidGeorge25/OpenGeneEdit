"""iGEM parts RAG: JSONL → ChromaDB + sentence-transformers retrieval.

Dataset: ``igem_dataset.jsonl`` in the project root (``part_id``, ``part_name``,
``part_type``, ``short_desc``, ``sequence``).

Environment:

  • ``DGENE_IGEM_JSONL`` — override path to JSONL (default: beside this package).
  • ``DGENE_CHROMA_PATH`` — persistent Chroma directory (default: ``.chroma_igem``).
  • ``DGENE_RAG`` — set ``0`` / ``false`` to disable substitution in the compile pipeline.
  • ``DGENE_RAG_MIN_SIM`` — minimum cosine similarity to **substitute** registry DNA for a
    slot (default ``0.6``). Below threshold the model's DNA slice for that slot is kept.

  • ``DGENE_RAG_MIN_SIM_PROMOTER`` — stricter floor for **Promoter** slots only (default
    ``0.80``). Reduces false Chroma hits where names share a short prefix (e.g. ``pL*``).

  • **NCBI Gene fallback** (``ncbi_gene.py``) — when iGEM does not verify a CDS-like slot,
    Entrez Gene + RefSeq nucleotide is queried for the symbol (e.g. ``PhzR``). Set
    ``NCBI_API_KEY`` or ``DGENE_NCBI_API_KEY`` for 10 req/s (free at NCBI). ``DGENE_NCBI=0``
    disables. See ``.env.example``.

Logging (stderr): By default every compile prints chroma query strings, alias hits,
per-slot retrieval lines, etc. (**``[oge/rag]``**). Set **``DGENE_LOG_REASONING_ONLY=1``**
to suppress those stderr lines while still running RAG — use with **``DGENE_DEBUG=1``**
(or **``DGENE_COMPILE_PROGRESS_STDERR=1``**) in ``inference`` to print **``[oge/progress]``**
and **``[oge/gemma]``** reasoning instead. Alternatively set **``DGENE_RAG_STDERR=0``**
without reasoning-only mode. **``DGENE_RAG_PROGRESS_MIRROR=1``** forces RAG lines into the
async job / UI trace even when **``DGENE_LOG_REASONING_ONLY=1``** (stderr stays clean).
Legacy channel compile still prints parsed thought under **``[oge/thought]``**.
Extra hit lists when RAG stderr is on: **``DGENE_RAG_DEBUG=1``** or **``DGENE_DEBUG=1``**.

Alias table (subset): ``luxR``→BBa_C0062, ``luxI``→BBa_C0061, ``plux``→BBa_R0062,
``mcherry``→BBa_E1010, ``j23100``→BBa_K4233030 (dataset snapshot has no BBa_J23100),
``b0034``→BBa_K812053 (no BBa_B0034), ``b0015``→BBa_B0015.

**LLM tool bridge:** ``search_igem_registry_for_llm_tool`` wraps ``retrieve_parts`` for Gemini
``functionResponse`` payloads (sequence previews to the model; full rows merged server-side).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ncbi_gene import fetch_gene_cds, gene_symbol_eligible_for_ncbi

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSONL = os.path.join(_MODULE_DIR, "igem_dataset.jsonl")
_DEFAULT_CHROMA = os.path.join(_MODULE_DIR, ".chroma_igem")
_EMBED_MODEL = "all-MiniLM-L6-v2"
_COLLECTION = "igem_parts"

# Word-boundary aliases → exact ``part_name`` in ``igem_dataset.jsonl``.
# BBa_J23100 / BBa_B0034 are not in this snapshot; we use closest documented equivalents.
_PART_ALIASES: Dict[str, str] = {
    "luxr": "BBa_C0062",
    "luxi": "BBa_C0061",
    "plux": "BBa_R0062",
    "mcherry": "BBa_E1010",
    "j23100": "BBa_K4233030",
    "b0034": "BBa_K812053",
    "b0015": "BBa_B0015",
}

_ALIAS_VERIFY_ONCE = False
_LOCK = threading.Lock()
_CLIENT = None
_COLLECTION_HANDLE = None
_MODEL = None

# Registry-derived vocabulary (built once from igem_dataset.jsonl, cached on disk).
# Key = token.lower(), value = {"display": "amilCP", "part_type": "CDS",
# "occurrences": N}. Lets `extract_part_names` recognize any iGEM-registered
# gene/protein/promoter token without us having to hard-code it in
# `_GENERIC_PARTS`. See `_ensure_registry_token_index`.
_REGISTRY_TOKEN_INDEX: Optional[Dict[str, Dict[str, Any]]] = None
_REGISTRY_TOKEN_LOCK = threading.Lock()
_REGISTRY_TOKEN_CACHE_VERSION = 2


def _jsonl_path() -> str:
    return os.environ.get("DGENE_IGEM_JSONL", _DEFAULT_JSONL).strip() or _DEFAULT_JSONL


def _chroma_path() -> str:
    return os.environ.get("DGENE_CHROMA_PATH", _DEFAULT_CHROMA).strip() or _DEFAULT_CHROMA


def rag_enabled_env() -> bool:
    v = os.environ.get("DGENE_RAG", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def min_similarity() -> float:
    raw = os.environ.get("DGENE_RAG_MIN_SIM", "0.6").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.6


def min_similarity_for_slot(part_type_filter: Optional[str]) -> float:
    """Per-slot cosine threshold (promoters need higher precision than 0.6).

    Prevents vague Chroma hits (e.g. ``pL*`` name collision) from substituting
    unrelated registry promoters when the brief names a specific sensor."""
    base = min_similarity()
    th = (part_type_filter or "").strip().lower()
    if th != "promoter":
        return base
    raw = os.environ.get("DGENE_RAG_MIN_SIM_PROMOTER", "0.80").strip()
    try:
        prom = max(0.0, min(1.0, float(raw)))
    except ValueError:
        prom = 0.80
    return max(base, prom)


def _rag_env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def rag_debug_enabled() -> bool:
    return _rag_env_bool("DGENE_RAG_DEBUG", False) or _rag_env_bool("DGENE_DEBUG", False)


def _rag_stderr_enabled() -> bool:
    """``[oge/rag]`` on stderr unless silenced."""

    if _rag_env_bool("DGENE_LOG_REASONING_ONLY", False):
        return False
    v = (os.environ.get("DGENE_RAG_STDERR") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _rag_mirror_enabled() -> bool:
    """Copy RAG lines into async compile-job progress (``[rag]`` in UI). Disabled in reasoning-only."""

    o = (os.environ.get("DGENE_RAG_PROGRESS_MIRROR") or "").strip().lower()
    if o in ("1", "true", "yes", "on"):
        return True
    if o in ("0", "false", "no", "off"):
        return False
    if _rag_env_bool("DGENE_LOG_REASONING_ONLY", False):
        return False
    return True


_RAG_DEBUG_MIRROR: Any = None


def set_rag_debug_mirror(cb: Optional[Any]) -> None:
    """Optional callback (e.g. async compile job line) — mirrors :func:`rag_debug_log` / :func:`rag_always_log` when enabled."""

    global _RAG_DEBUG_MIRROR
    _RAG_DEBUG_MIRROR = cb


def rag_debug_log(line: str) -> None:
    if not rag_debug_enabled():
        return
    ts = time.strftime("%H:%M:%S")
    if _rag_stderr_enabled():
        sys.stderr.write(f"[oge/rag {ts}] {line}\n")
        sys.stderr.flush()
    if _RAG_DEBUG_MIRROR is not None and _rag_mirror_enabled():
        try:
            _RAG_DEBUG_MIRROR(line)
        except Exception:
            pass


def rag_always_log(line: str) -> None:
    """Always-on RAG log line (no ``DGENE_RAG_DEBUG`` gate).

    Used to mark high-signal events (``RAG called for: ...``, ``apply_rag_substitution``
    invocations) so we can verify retrieval is actually running rather than being silently
    skipped.
    """
    ts = time.strftime("%H:%M:%S")
    if _rag_stderr_enabled():
        sys.stderr.write(f"[oge/rag {ts}] {line}\n")
        sys.stderr.flush()
    if _RAG_DEBUG_MIRROR is not None and _rag_mirror_enabled():
        try:
            _RAG_DEBUG_MIRROR(line)
        except Exception:
            pass


def reasoning_trace_log(line: str) -> None:
    """Legacy path: dump parsed model thought chunk to stderr (**``[oge/thought]``**).

    Not filtered by reasoning-only mode (unlike ``[oge/rag]``) so trimming RAG stderr
    still shows channel-thought prose during ``apply_rag_substitution``."""
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[oge/thought {ts}] {line}\n")
    sys.stderr.flush()


def _lazy_clients():
    global _CLIENT, _COLLECTION_HANDLE, _MODEL
    with _LOCK:
        if _CLIENT is not None and _COLLECTION_HANDLE is not None and _MODEL is not None:
            return _CLIENT, _COLLECTION_HANDLE, _MODEL
        try:
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "iGEM RAG requires chromadb and sentence-transformers. "
                "Install with: pip install chromadb sentence-transformers"
            ) from exc

        _MODEL = SentenceTransformer(_EMBED_MODEL)
        _CLIENT = chromadb.PersistentClient(
            path=os.path.abspath(_chroma_path()),
            settings=Settings(anonymized_telemetry=False),
        )
        _COLLECTION_HANDLE = _CLIENT.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        return _CLIENT, _COLLECTION_HANDLE, _MODEL


def _load_jsonl_rows(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _embed_document(row: dict) -> str:
    return (
        f"{row.get('part_name', '')} {row.get('part_type', '')} "
        f"{row.get('short_desc', '')}"
    ).strip()


def ensure_indexed(*, progress_cb: Optional[Any] = None) -> int:
    """Build the Chroma collection if empty. Returns document count."""

    def prog(msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    _, collection, model = _lazy_clients()
    n = collection.count()
    if n > 0:
        return n

    path = _jsonl_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"iGEM dataset not found: {path}")

    prog(f"rag · indexing iGEM parts from {path!r} (one-time)…")
    rows = _load_jsonl_rows(path)
    batch = 256
    total = len(rows)
    for start in range(0, total, batch):
        chunk = rows[start : start + batch]
        ids = [str(r.get("part_id", start + i)) for i, r in enumerate(chunk)]
        docs = [_embed_document(r) for r in chunk]
        embeddings = model.encode(
            docs,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        metadatas = []
        for r in chunk:
            metadatas.append(
                {
                    "part_id": str(r.get("part_id", "")),
                    "part_name": str(r.get("part_name", ""))[:512],
                    "part_type": str(r.get("part_type", ""))[:128],
                    "short_desc": str(r.get("short_desc", ""))[:2048],
                    "sequence": str(r.get("sequence", "")),
                }
            )
        collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            documents=docs,
            metadatas=metadatas,
        )
        prog(
            f"rag · indexed {min(start + batch, total)}/{total} parts…"
        )

    prog(f"rag · index complete · {collection.count()} vectors")
    return collection.count()


def _verify_alias_targets_once() -> None:
    """Log once per process that alias targets exist in JSONL with bp counts."""

    global _ALIAS_VERIFY_ONCE
    if _ALIAS_VERIFY_ONCE:
        return
    _ALIAS_VERIFY_ONCE = True
    path = _jsonl_path()
    if not os.path.isfile(path):
        rag_always_log(f"RAG alias verify skipped — dataset not found: {path}")
        return
    by_name: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pn = str(r.get("part_name", ""))
            if pn:
                by_name[pn] = len(str(r.get("sequence", "")))
    rag_always_log("RAG alias registry — targets in JSONL (by part_name):")
    for pn in sorted(set(_PART_ALIASES.values())):
        L = by_name.get(pn)
        rag_always_log(f"  {pn}: {'OK ' + str(L) + ' bp' if L is not None else 'MISSING'}")
    rag_always_log("RAG alias keys (word-boundary match in query):")
    for key in sorted(_PART_ALIASES.keys()):
        pn = _PART_ALIASES[key]
        L = by_name.get(pn)
        rag_always_log(
            f"  ALIAS {key!r} → {pn} ({L} bp)" if L is not None else f"  ALIAS {key!r} → {pn} (MISSING)"
        )


def _resolve_registry_lookup(query: str) -> Tuple[Optional[str], str]:
    """Return ``(part_name, note)`` for direct Chroma lookup.

    ``note`` is ``explicit-BBa_…``, ``alias:token``, or ``none`` (use semantic search).
    """

    q = (query or "").strip()
    if not q:
        return None, "none"
    m = _BBA_RE.search(q)
    if m:
        return m.group(0), f"explicit-{m.group(0)}"
    qfold = q.casefold()
    for key in sorted(_PART_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", qfold):
            return _PART_ALIASES[key], f"alias:{key}"
    return None, "none"


@dataclass
class RetrievedPart:
    part_name: str
    part_type: str
    short_desc: str
    sequence: str
    similarity: float
    part_id: str = ""
    match_kind: str = "semantic"  # exact-alias | exact-bba | semantic


def _lookup_part_by_exact_name(part_name: str) -> Optional[RetrievedPart]:
    """Direct metadata lookup in Chroma (same rows as JSONL index)."""

    _, collection, _ = _lazy_clients()
    raw = collection.get(
        where={"part_name": part_name},
        include=["metadatas"],
        limit=1,
    )
    ids = raw.get("ids") or []
    metas = raw.get("metadatas") or []
    if not ids or not metas or not metas[0]:
        return None
    meta = metas[0]
    return RetrievedPart(
        part_id=str(meta.get("part_id", ids[0])),
        part_name=str(meta.get("part_name", "")),
        part_type=str(meta.get("part_type", "")),
        short_desc=str(meta.get("short_desc", "")),
        sequence=str(meta.get("sequence", "")),
        similarity=1.0,
    )


def search_igem_registry_for_llm_tool(args: dict) -> dict:
    """Execute ``search_igem_registry`` tool args → structured hits for Gemini ``functionResponse``.

    Returns ``{"parts": [...]}`` for the API (sequence previews only). Full rows for assembly /
    UI merge are appended via ``inference.extend_igem_tool_merge_rows`` (thread-local buffer).
    """

    qraw = args.get("query")
    query = str(qraw if qraw is not None else "").strip()
    pt_raw = args.get("part_type")
    part_type_filter: Optional[str] = None
    if pt_raw is not None:
        p = str(pt_raw).strip()
        if p and p.casefold() not in ("any", "none", "unknown"):
            if p in ("Promoter", "RBS", "CDS", "Terminator"):
                part_type_filter = p
    tk = args.get("top_k", 10)
    try:
        top_k = int(float(tk))
    except (TypeError, ValueError):
        top_k = 10
    top_k = max(1, min(25, top_k))

    hits = retrieve_parts(query, part_type_filter=part_type_filter, top_k=top_k)
    preview_cap = 160
    llm_parts: List[dict] = []
    full_rows: List[dict] = []
    for h in hits:
        seq = str(h.sequence or "")
        llm_parts.append(
            {
                "part_name": h.part_name,
                "part_type": h.part_type,
                "short_desc": (h.short_desc or "")[:380],
                "sequence_bp": len(seq),
                "sequence_preview": seq[:preview_cap] + ("…" if len(seq) > preview_cap else ""),
                "similarity": round(float(h.similarity), 4),
                "match_kind": getattr(h, "match_kind", "semantic"),
            }
        )
        full_rows.append(
            {
                "part_name": h.part_name,
                "part_type": h.part_type,
                "short_desc": h.short_desc,
                "sequence": seq,
                "similarity": float(h.similarity),
                "match_kind": getattr(h, "match_kind", "semantic"),
                "retrieval_query": query,
            }
        )

    try:
        from inference import extend_igem_tool_merge_rows

        extend_igem_tool_merge_rows(full_rows)
    except Exception:
        pass

    return {"query": query, "part_type_filter": part_type_filter, "parts": llm_parts}


def retrieve_parts(
    query: str,
    part_type_filter: Optional[str] = None,
    top_k: int = 3,
) -> List[RetrievedPart]:
    """Keyword query → matching registry parts (exact BBa / alias table, else embedding search)."""

    q = (query or "").strip()
    filt_suffix = (
        f" (part_type_filter={part_type_filter!r})" if part_type_filter else ""
    )
    rag_always_log(f"RAG chroma query string: {q!r}{filt_suffix} top_k={top_k}")
    if not q:
        rag_always_log("RAG called with empty query — returning [] (no retrieval performed)")
        return []

    ensure_indexed()
    lookup_name, res_note = _resolve_registry_lookup(q)
    from_alias = res_note.startswith("alias:")
    alias_token = res_note.split(":", 1)[1] if from_alias else ""

    if lookup_name:
        hit = _lookup_part_by_exact_name(lookup_name)
        if hit is not None:
            type_mismatch = bool(part_type_filter and hit.part_type != part_type_filter)
            if type_mismatch and from_alias:
                rag_always_log(
                    f"type filter {part_type_filter!r} overridden for alias hit "
                    f"(registry part is {hit.part_type!r})"
                )
                type_mismatch = False
            if not type_mismatch:
                if from_alias:
                    rag_always_log(
                        f"ALIAS HIT: {alias_token} -> {hit.part_name} "
                        f"({len(hit.sequence)} bp, part_id={hit.part_id})"
                    )
                else:
                    rag_always_log(
                        f"EXACT BBa lookup: {lookup_name!r} part_id={hit.part_id} "
                        f"type={hit.part_type!r} ({len(hit.sequence)} bp) sim=1.0000"
                    )
                rag_debug_log(
                    f"retrieve exact path={res_note!r} part_name={hit.part_name!r} "
                    f"part_id={hit.part_id!r} seq_bp={len(hit.sequence)}"
                )
                return [
                    RetrievedPart(
                        part_name=hit.part_name,
                        part_type=hit.part_type,
                        short_desc=hit.short_desc,
                        sequence=hit.sequence,
                        similarity=1.0,
                        part_id=hit.part_id,
                        match_kind=("exact-alias" if from_alias else "exact-bba"),
                    )
                ]
            rag_debug_log(
                f"exact part_name={lookup_name!r} type {hit.part_type!r} != "
                f"filter {part_type_filter!r} — falling back to semantic search"
            )
        else:
            rag_always_log(
                f"exact lookup MISS part_name={lookup_name!r} (note={res_note}) — "
                "semantic search"
            )

    _, collection, model = _lazy_clients()
    emb = model.encode(q, normalize_embeddings=True)
    kwargs: dict = {
        "query_embeddings": [emb.tolist()],
        "n_results": max(1, min(50, top_k)),
        "include": ["distances", "metadatas", "documents"],
    }
    if part_type_filter:
        kwargs["where"] = {"part_type": part_type_filter}

    raw = collection.query(**kwargs)
    ids_out = (raw.get("ids") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]

    rag_always_log(
        f"RAG SEMANTIC SEARCH query={q!r} part_type_filter={part_type_filter!r} "
        f"n_results={kwargs['n_results']}"
    )
    rag_debug_log(
        f"retrieve semantic query={q!r} part_type_filter={part_type_filter!r} top_k={kwargs['n_results']}"
    )

    out: List[RetrievedPart] = []
    for i, mid in enumerate(ids_out):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        d = float(dists[i]) if i < len(dists) else 1.0
        sim = max(0.0, min(1.0, 1.0 - d))
        out.append(
            RetrievedPart(
                part_id=str(meta.get("part_id", mid)),
                part_name=str(meta.get("part_name", "")),
                part_type=str(meta.get("part_type", "")),
                short_desc=str(meta.get("short_desc", "")),
                sequence=str(meta.get("sequence", "")),
                similarity=sim,
            )
        )
    for i, rp in enumerate(out):
        desc = (rp.short_desc or "").replace("\n", " ")
        if len(desc) > 120:
            desc = desc[:117] + "…"
        rag_debug_log(
            f"retrieve semantic hit #{i + 1} sim={rp.similarity:.4f} part_id={rp.part_id!r} "
            f"name={rp.part_name!r} type={rp.part_type!r} desc={desc!r} seq_bp={len(rp.sequence)}"
        )
    if out:
        top = out[0]
        rag_always_log(
            f"RAG semantic best: {top.part_name!r} part_id={top.part_id} "
            f"sim={top.similarity:.4f} bp={len(top.sequence)}"
        )
    else:
        rag_always_log("RAG semantic: no hits returned")
    return out


_SECTION_HEAD = re.compile(
    r"(?im)^\s*(parts?\s*used|components|construct\s*outline|part\s*list|modules?|design\s*modules?)\s*[:.\s]*\s*$"
)
_BULLET = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s+(.+)$")

# Lines like ``Line 2: Design reasoning.``, ``Line k+1: \``, lone backticks/quotes —
# these are prompt-skeleton echoes from Gemma copying our template wording back to us
# and must never reach the registry as a query.
_SKELETON_GARBAGE = re.compile(
    r"^\s*line\s*\d+\b|"
    r"^\s*line\s*k\s*[+\-]\s*\d+|"
    r"^\s*lines?\s+\d+\s*\.\.\s*k\b|"
    r"^\s*reasoning\s*:?\s*$|"
    r"^\s*[`'\"\\]+\s*$|"
    r"^\s*<[a-z|/][^>]*>\s*$",
    re.IGNORECASE,
)


def extract_part_descriptions(thought: str) -> List[str]:
    """Heuristic extraction of per-part lines from free-form reasoning text."""

    text = (thought or "").strip()
    if not text:
        return []

    lines = text.splitlines()
    collected: List[str] = []
    in_section = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if _SECTION_HEAD.match(line):
            in_section = True
            continue

        m = _BULLET.match(line)
        if m:
            piece = m.group(1).strip()
            piece = re.sub(r"\*\*([^*]+)\*\*", r"\1", piece)
            if len(piece) >= 8:
                collected.append(piece)
            continue

        if in_section:
            if ":" in line[:48]:
                _, _, rest = line.partition(":")
                rest = rest.strip()
                if len(rest) >= 8:
                    collected.append(rest)
            elif len(line) >= 12:
                collected.append(line)

    if not collected:
        for raw in lines:
            m = _BULLET.match(raw.strip())
            if m:
                piece = m.group(1).strip()
                if len(piece) >= 12:
                    collected.append(piece)

    seen = set()
    out: List[str] = []
    for c in collected:
        if _SKELETON_GARBAGE.search(c):
            continue
        key = c.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out[:48]


_TYPE_PREFIX_CANON = {
    "promoter": "Promoter",
    "rbs": "RBS",
    "cds": "CDS",
    "terminator": "Terminator",
}

# --- Per-part identifier extraction (drives high-precision RAG queries) ---------------
# These regexes pull crisp part names out of free-form reasoning so each registry lookup
# is "J23100 promoter" rather than a whole sentence — similarity scores jump from
# ~0.25 (junk) into ~0.85+ (verified) for canonical iGEM parts.

_BBA_RE = re.compile(r"\bBBa_[A-Z]\d{4,5}[a-zA-Z]?\b")
_J_PROMOTER_RE = re.compile(r"\bJ\d{5}\b")
_B_PART_RE = re.compile(r"\bB\d{4}\b")

# (canonical_name, default_type_hint). Default applies when surrounding context doesn't
# already yield a stronger type hint via :func:`_scan_window_for_type`.
_GENERIC_PARTS: Tuple[Tuple[str, Optional[str]], ...] = (
    # --- Fluorescent / chromoproteins (iGEM standards) -----------------------
    ("sfGFP", "CDS"),
    ("eGFP", "CDS"),
    ("mGFP", "CDS"),
    ("GFP", "CDS"),
    ("mRFP1", "CDS"),
    ("mRFP", "CDS"),
    ("RFP", "CDS"),
    ("mCherry", "CDS"),
    ("mScarlet", "CDS"),
    ("mOrange", "CDS"),
    ("mTurquoise", "CDS"),
    ("mPlum", "CDS"),
    ("mTagBFP2", "CDS"),
    ("mTagBFP", "CDS"),
    ("dsRed", "CDS"),
    ("dTomato", "CDS"),
    ("YFP", "CDS"),
    ("CFP", "CDS"),
    ("BFP", "CDS"),
    ("Citrine", "CDS"),
    ("Venus", "CDS"),
    ("Cerulean", "CDS"),
    ("Sapphire", "CDS"),
    ("amilCP", "CDS"),
    ("amilGFP", "CDS"),
    ("eforRed", "CDS"),
    ("eforBlue", "CDS"),
    ("fwYellow", "CDS"),
    ("aeBlue", "CDS"),
    ("cjBlue", "CDS"),
    ("amajLime", "CDS"),
    ("scOrange", "CDS"),
    ("spisPink", "CDS"),
    ("asPink", "CDS"),
    ("mRojoA", "CDS"),
    # --- Bioluminescence ---------------------------------------------------
    ("luxAB", "CDS"),
    ("luxCDABE", "CDS"),
    # --- Quorum sensing transcription factors / synthases ------------------
    ("LuxR", "CDS"),
    ("LuxI", "CDS"),
    ("LasR", "CDS"),
    ("LasI", "CDS"),
    ("RhlR", "CDS"),
    ("RhlI", "CDS"),
    ("AhlR", "CDS"),
    ("AhlI", "CDS"),
    ("CinR", "CDS"),
    ("CinI", "CDS"),
    ("EsaR", "CDS"),
    ("EsaI", "CDS"),
    ("TraR", "CDS"),
    ("TraI", "CDS"),
    # --- Two-component / metabolite sensor regulators ----------------------
    ("LldR", "CDS"),
    ("LldP", "CDS"),
    ("PhzR", "CDS"),
    ("PhzI", "CDS"),
    ("OhhR", "CDS"),
    ("PbrR", "CDS"),
    ("MerR", "CDS"),
    ("ArsR", "CDS"),
    ("CueR", "CDS"),
    ("ZntR", "CDS"),
    ("CadC", "CDS"),
    ("OmpR", "CDS"),
    ("EnvZ", "CDS"),
    ("ToxR", "CDS"),
    # --- Common regulators -------------------------------------------------
    ("lacI", "CDS"),
    ("tetR", "CDS"),
    ("araC", "CDS"),
    ("cI", "CDS"),
    # --- Operators (no type hint -> "operator" subcategory in map) ---------
    ("lacO", None),
    ("tetO", None),
    # --- Promoters ---------------------------------------------------------
    ("Plux", "Promoter"),
    ("PluxR", "Promoter"),
    ("PlasR", "Promoter"),
    ("PlasI", "Promoter"),
    ("PrhlR", "Promoter"),
    ("PcinR", "Promoter"),
    ("PtraR", "Promoter"),
    ("PlldR", "Promoter"),
    ("Plld", "Promoter"),
    ("PphzR", "Promoter"),
    ("Pphz", "Promoter"),
    ("Phyb", "Promoter"),
    ("PbrA", "Promoter"),
    ("pBAD", "Promoter"),
    ("pTet", "Promoter"),
    ("pLac", "Promoter"),
    ("Plac", "Promoter"),
    ("pTrc", "Promoter"),
    ("Ptrc", "Promoter"),
    ("pT7", "Promoter"),
    ("T7", "Promoter"),
)


# Camel/mixed-case identifiers in iGEM thoughts that aren't in `_GENERIC_PARTS`
# (novel sensor regulators, custom hybrid promoters, etc.). We only accept matches
# when a part-type keyword sits within ±60 chars to keep false positives off
# (English words rarely sit next to "promoter"/"CDS"/"RBS"/"terminator").
_CAMEL_GENE_RE = re.compile(
    r"\b("
    r"[A-Z][a-z]{1,5}[A-Z][A-Za-z0-9]{0,6}"      # PascalCase + internal cap (PhzR, LldR, ToxR, OmpA)
    r"|[a-z]{2,6}[A-Z]{2,5}[a-z0-9]?"            # lowercase prefix + caps (amilCP, eforRed, dsRed, sfGFP)
    r"|[a-z][A-Z][A-Za-z0-9]{2,8}"               # leading-lower + cap (mCherry, mScarlet, dTomato)
    r")\b"
)

# Tokens that match the camelCase shape but are NEVER a gene/part — keeps the
# fallback from polluting slots with English/biology jargon or restriction sites.
_CAMEL_GENE_BLOCKLIST = frozenset({
    "BBa", "iGEM", "DNA", "RNA", "mRNA", "tRNA", "rRNA", "ATP", "ADP", "GTP",
    "PCR", "RBS", "CDS", "ORF", "UTR", "TSS", "kDa", "AND", "NOR", "NAND", "XOR",
    "EcoRI", "XbaI", "SpeI", "PstI", "NotI", "BamHI", "HindIII", "PvuI", "PvuII",
    "BsaI", "SapI", "BsmBI", "AarI",
    "OK", "API", "URL", "JSON",
})

# Part-type words within ±60 chars qualify a camelCase token as a real slot.
_CAMEL_TYPE_CONTEXT_RE = re.compile(
    r"\b(promoter|promotor|terminator|rbs|cds|coding\s+sequence|coding\s+region|"
    r"open\s+reading\s+frame|operator|expression|expressed|drives?|driving|"
    r"encodes?|encoding|chromoprotein|fluorescent\s+protein|reporter|sensor|"
    r"transcription\s+factor|repressor|activator|protein|gene)\b",
    re.IGNORECASE,
)


_TYPE_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("promoter", "Promoter"),
    ("promotor", "Promoter"),
    ("terminator", "Terminator"),
    ("ribosome binding", "RBS"),
    (" rbs ", "RBS"),
    (" rbs.", "RBS"),
    (" rbs,", "RBS"),
    (" rbs:", "RBS"),
    (" rbs)", "RBS"),
    ("coding sequence", "CDS"),
    ("coding region", "CDS"),
    ("open reading frame", "CDS"),
    (" cds ", "CDS"),
    (" cds.", "CDS"),
    (" cds,", "CDS"),
    (" cds:", "CDS"),
    (" cds)", "CDS"),
)


def _classify_window(window: str, *, max_offset: Optional[int] = None) -> Optional[str]:
    """Return the part type whose keyword appears at the smallest offset in ``window``.

    Picking the nearest keyword (rather than a fixed priority order) keeps each part name
    bound to *its* type word — so in ``"B0034 RBS, sfGFP coding sequence, B0015 terminator"``
    each name resolves to the right type even though all three keywords are in range.

    ``max_offset`` (optional) rejects matches that sit further than that many chars
    from the start of the window. This handles cases like
    ``"the terminator B0015 follows the RBS B0034"`` where the cut after-window
    for ``B0015`` is ``" follows the RBS "`` — the ``RBS`` keyword is 13 chars in,
    really belongs to ``B0034``, and should not bind to ``B0015``.
    """

    padded = f" {window} "
    best: Optional[Tuple[int, str]] = None
    for needle, label in _TYPE_KEYWORDS:
        idx = padded.find(needle)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, label)
    if best is None:
        return None
    if max_offset is not None and best[0] - 1 > max_offset:
        return None
    return best[1]


_PART_TOKEN_BOUNDARY_RE = re.compile(
    r"\b(?:BBa_[A-Z]\d{4,5}[a-zA-Z]?|J\d{5}|B\d{4})\b"
)


def _scan_window_for_type(
    text: str,
    start: int,
    end: int,
    *,
    span_after: int = 40,
    span_before: int = 12,
) -> Optional[str]:
    """Look around a regex match for a part-type keyword.

    iGEM convention is ``<part_name> <type_word>`` (``B0034 RBS``, ``B0015 terminator``,
    ``J23100 promoter``). The after-window is searched first and **cut at the next
    BBa_/B####/J##### identifier** so a type word like ``RBS`` in
    ``"amilCP expression via a B0034 RBS"`` binds to ``B0034`` rather than leaking
    back to ``amilCP``.

    The before-window catches the rare reversed-order convention
    (``"terminator B0015"``, ``"RBS B0034"``) and is rejected if a sentence
    boundary (``.`` / ``;`` / ``!``) or any other part identifier sits inside
    it — otherwise ``"B0015 terminator. lacI represses…"`` would mis-type
    ``lacI`` as a terminator and ``"T7 promoter drives sfGFP"`` would mis-type
    ``sfGFP`` as a promoter.
    """

    after_full = text[end : end + span_after]
    next_part = _PART_TOKEN_BOUNDARY_RE.search(after_full)
    after = (after_full if not next_part else after_full[: next_part.start()]).lower()
    # When the after-window is followed by another part identifier, only
    # accept type words that sit close to the *current* part (within 8 chars).
    # Otherwise the type word is in the immediate-precedence zone of the next
    # part and belongs to it (reverse-order convention: ``"... follows the RBS B0034"``).
    after_max_offset = 8 if next_part is not None else None
    hit_after = _classify_window(after, max_offset=after_max_offset)
    if hit_after is not None:
        return hit_after
    before_full = text[max(0, start - span_before) : start]
    if _PART_TOKEN_BOUNDARY_RE.search(before_full):
        return None
    if any(ch in before_full for ch in ".;!"):
        return None
    return _classify_window(before_full.lower())


def _b_part_type_prior(name: str) -> Optional[str]:
    """Canonical iGEM B-series ranges: B0010-B0019 are Terminators, B0030-B0039 are RBSs."""

    try:
        n = int(name[1:])
    except ValueError:
        return None
    if 10 <= n <= 19:
        return "Terminator"
    if 30 <= n <= 39:
        return "RBS"
    return None


# --- Registry-derived vocabulary (built from full igem_dataset.jsonl, cached) -------
# Drives `extract_part_names` so any iGEM-registered protein/promoter/operator name
# appearing in the model's reasoning becomes its own slot — rather than relying on a
# tiny hardcoded list. Without this, slots like `amilCP` / `lacZ` / `LldR` would only
# match if we'd manually added them to `_GENERIC_PARTS`, and any other 30k registered
# parts would silently be lost from the construct map.

# Token shapes that look like iGEM gene/protein/promoter names. Loose on purpose —
# the registry membership check below is the actual filter, so false-positive
# *shapes* are fine as long as they aren't real parts.
_REGISTRY_TOKEN_RE = re.compile(
    r"\b("
    r"[A-Z]{2,5}\d{0,2}"                          # GFP, RFP, T7, T1
    r"|[a-z]{1,6}[A-Z][A-Za-z0-9]{0,8}"           # lacI, tetR, mCherry, amilCP, sfGFP, dTomato
    r"|[A-Z][a-z]{1,6}[A-Z][A-Za-z0-9]{0,6}"      # LldR, PhzR, ToxR, OmpA, AraC
    r")\b"
)

# Tokens that LOOK gene-shaped but should never be slots even if a part description
# happens to mention them (English jargon, methods, units, restriction enzymes,
# chemical inducers / small-molecule ligands).
_REGISTRY_TOKEN_STOP = frozenset(s.lower() for s in (
    "DNA", "RNA", "mRNA", "tRNA", "rRNA", "ATP", "ADP", "GTP", "NADH", "NADPH",
    "PCR", "BBa", "iGEM", "RBS", "CDS", "ORF", "UTR", "TSS", "kDa", "pKa",
    "EcoRI", "XbaI", "SpeI", "PstI", "NotI", "BamHI", "HindIII", "PvuI", "PvuII",
    "BsaI", "SapI", "BsmBI", "AarI",
    "AND", "NOR", "NAND", "XOR", "OR", "NOT", "OK", "API", "URL", "JSON", "FAQ",
    "USA", "USSR", "UK", "EU", "Bp", "Kb", "Mb", "Da", "uL", "mL", "uM", "mM",
    "From", "With", "For", "And", "The", "Of", "An", "In", "On", "At", "To",
    "When", "Then", "If", "Else", "Use", "Uses", "Used", "Add", "Set",
    "TE",  # "TE from coliphage T7" — terminator filler word, not a part name
    # Chemical inducers / small-molecule ligands frequently named in part
    # descriptions ("induced by IPTG", "AHL-responsive") but never themselves
    # iGEM parts.
    "IPTG", "aTc", "AHL", "ATc", "OHHL", "OC6HSL", "OC12HSL", "AI2",
    "HSL", "AhL", "PoPS", "RiPS",
))

# Minimum number of registry rows a token must appear in before it counts as a real
# part. Cuts one-off mentions / typos / rare incidental uses of English-ish words
# that slipped past the stoplist.
_REGISTRY_TOKEN_MIN_OCCURRENCES = 2


def _registry_token_cache_path() -> str:
    base = _chroma_path()
    return os.path.join(os.path.abspath(base), "registry_tokens.json")


def _build_registry_token_index() -> Dict[str, Dict[str, Any]]:
    """One-pass scan of ``igem_dataset.jsonl`` building a token vocabulary.

    For each row we extract gene-name-shaped tokens from ``short_desc`` and tally
    their part_types. Result: ``{token_lower: {"display": "amilCP",
    "part_type": "CDS", "occurrences": N}}``. Tokens that appear in only a
    handful of rows are pruned by the caller (see ``_REGISTRY_TOKEN_MIN_OCCURRENCES``).
    """

    path = _jsonl_path()
    if not os.path.isfile(path):
        rag_always_log(
            f"registry token index: dataset missing at {path!r} — "
            f"falling back to hardcoded `_GENERIC_PARTS` only"
        )
        return {}

    type_counts: Dict[str, Dict[str, int]] = {}
    canonical_case: Dict[str, str] = {}
    rows_seen = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_seen += 1
            short_desc = str(row.get("short_desc", ""))
            part_type = str(row.get("part_type", ""))
            if not short_desc or not part_type:
                continue
            for m in _REGISTRY_TOKEN_RE.finditer(short_desc):
                tok = m.group(1)
                if len(tok) < 2 or len(tok) > 16:
                    continue
                key = tok.lower()
                if key in _REGISTRY_TOKEN_STOP:
                    continue
                bucket = type_counts.setdefault(key, {})
                bucket[part_type] = bucket.get(part_type, 0) + 1
                if key not in canonical_case:
                    canonical_case[key] = tok

    out: Dict[str, Dict[str, Any]] = {}
    for key, counts in type_counts.items():
        occ = sum(counts.values())
        if occ < _REGISTRY_TOKEN_MIN_OCCURRENCES:
            continue
        best_type = max(counts.items(), key=lambda kv: kv[1])[0]
        out[key] = {
            "display": canonical_case[key],
            "part_type": best_type,
            "occurrences": occ,
        }

    rag_always_log(
        f"registry token index built: {len(out)} tokens kept "
        f"(min_occ={_REGISTRY_TOKEN_MIN_OCCURRENCES}) from {rows_seen} JSONL rows"
    )
    return out


def _ensure_registry_token_index() -> Dict[str, Dict[str, Any]]:
    """Lazy, thread-safe loader with on-disk cache keyed by JSONL mtime.

    Cache file lives next to the Chroma directory so it's already in
    ``.gitignore`` and gets blown away when the dataset changes.
    """

    global _REGISTRY_TOKEN_INDEX
    if _REGISTRY_TOKEN_INDEX is not None:
        return _REGISTRY_TOKEN_INDEX
    with _REGISTRY_TOKEN_LOCK:
        if _REGISTRY_TOKEN_INDEX is not None:
            return _REGISTRY_TOKEN_INDEX

        cache_path = _registry_token_cache_path()
        try:
            jsonl_mtime = os.path.getmtime(_jsonl_path())
        except OSError:
            jsonl_mtime = 0.0

        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if (
                    isinstance(data, dict)
                    and data.get("version") == _REGISTRY_TOKEN_CACHE_VERSION
                    and float(data.get("jsonl_mtime", 0.0)) == jsonl_mtime
                    and isinstance(data.get("tokens"), dict)
                ):
                    _REGISTRY_TOKEN_INDEX = data["tokens"]
                    rag_always_log(
                        f"registry token index loaded from cache "
                        f"({len(_REGISTRY_TOKEN_INDEX)} tokens, {cache_path!r})"
                    )
                    return _REGISTRY_TOKEN_INDEX
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                rag_debug_log(f"registry token cache unreadable ({exc!s}) — rebuilding")

        _REGISTRY_TOKEN_INDEX = _build_registry_token_index()

        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "version": _REGISTRY_TOKEN_CACHE_VERSION,
                        "jsonl_mtime": jsonl_mtime,
                        "tokens": _REGISTRY_TOKEN_INDEX,
                    },
                    fh,
                )
            rag_debug_log(
                f"registry token cache written ({len(_REGISTRY_TOKEN_INDEX)} tokens) "
                f"to {cache_path!r}"
            )
        except OSError as exc:
            rag_debug_log(f"registry token cache write failed: {exc!s}")

        return _REGISTRY_TOKEN_INDEX


def extract_part_names(thought: str) -> List[Tuple[Optional[str], str]]:
    """Scan reasoning for explicit iGEM identifiers + canonical generic names.

    Returns ``[(type_hint, query_text), ...]`` in document order, deduped. The
    ``query_text`` is enriched with the type word so the embedding model gets a
    stronger signal (e.g. ``"B0034 RBS"`` rather than bare ``"B0034"``).
    """

    text = thought or ""
    if not text.strip():
        return []

    events: List[Tuple[int, Optional[str], str]] = []
    seen_names: set = set()

    def add(start: int, type_hint: Optional[str], name: str) -> None:
        # Dedupe by casefolded *name only* (not by (type, name)) so the same
        # gene/protein appears at most once per construct, even when multiple
        # passes (BBa, _GENERIC_PARTS, registry-vocab, camelCase) hit it with
        # subtly different type guesses or different-case display strings.
        # First-write wins, so the most-trusted pass (run earliest) sets the
        # final type_hint.
        key = name.casefold()
        if key in seen_names:
            return
        seen_names.add(key)
        q = f"{name} {type_hint}".strip() if type_hint else name
        events.append((start, type_hint, q))

    for m in _BBA_RE.finditer(text):
        add(m.start(), _scan_window_for_type(text, m.start(), m.end()), m.group(0))

    for m in _J_PROMOTER_RE.finditer(text):
        add(m.start(), "Promoter", m.group(0))

    for m in _B_PART_RE.finditer(text):
        name = m.group(0)
        type_hint = _scan_window_for_type(text, m.start(), m.end()) or _b_part_type_prior(name)
        add(m.start(), type_hint, name)

    for canonical, default_type in _GENERIC_PARTS:
        for m in re.finditer(rf"\b{re.escape(canonical)}\b", text, flags=re.IGNORECASE):
            type_hint = _scan_window_for_type(text, m.start(), m.end()) or default_type
            add(m.start(), type_hint, canonical)

    # Registry-vocabulary pass: scan the thought for any token that exists in
    # the full igem_dataset.jsonl vocabulary. This is the workhorse — it lets
    # the system recognize ANY iGEM-registered gene/protein (amilCP, lacZ,
    # cI857, dCas9, …) without us hard-coding every one in `_GENERIC_PARTS`.
    # Failure to load the index degrades gracefully to the hardcoded list.
    try:
        registry_index = _ensure_registry_token_index()
    except Exception as exc:  # noqa: BLE001 — never let vocab loading break extraction
        rag_debug_log(f"registry token index load failed ({exc!s}) — skipping pass")
        registry_index = {}

    if registry_index:
        for m in _REGISTRY_TOKEN_RE.finditer(text):
            tok = m.group(1)
            if not tok or len(tok) < 2 or len(tok) > 16:
                continue
            key = tok.lower()
            if key in _REGISTRY_TOKEN_STOP:
                continue
            info = registry_index.get(key)
            if not info:
                continue  # token not in any iGEM short_desc — not a real part
            display = info.get("display") or tok
            default_type = info.get("part_type")
            type_hint = _scan_window_for_type(text, m.start(), m.end()) or default_type
            add(m.start(), type_hint, display)

    # Context-gated camelCase fallback: catches truly novel proteins / custom
    # promoters that aren't in the registry at all (e.g. PhzR/PhzI from
    # Pyocyanin sensors, custom hybrid promoters). Without this, anything the
    # model invents would be silently lost when the sequence is chunk-split
    # during substitution.
    canonical_lower = {c.lower() for c, _ in _GENERIC_PARTS}
    for m in _CAMEL_GENE_RE.finditer(text):
        token = m.group(1)
        if not token or len(token) < 3 or len(token) > 14:
            continue
        if token in _CAMEL_GENE_BLOCKLIST:
            continue
        tlow = token.lower()
        if tlow in canonical_lower:
            continue  # already added via _GENERIC_PARTS pass
        if registry_index and tlow in registry_index:
            continue  # already added via registry-vocabulary pass
        # Skip if this token is part of a BBa_/B####/J##### already captured
        # (e.g. "B0034" would not match the camel regex anyway, but PhzR could
        # appear inside "PhzR/PhzI" — we still want both as separate hits).
        ctx_lo = max(0, m.start() - 60)
        ctx_hi = min(len(text), m.end() + 60)
        ctx = text[ctx_lo:ctx_hi]
        if not _CAMEL_TYPE_CONTEXT_RE.search(ctx):
            continue
        type_hint = _scan_window_for_type(text, m.start(), m.end())
        add(m.start(), type_hint, token)

    events.sort(key=lambda t: t[0])
    return [(th, q) for _, th, q in events]


def _cassette_tier(part_type_filter: Optional[str]) -> int:
    """Sort key: 5′→3′ cassette order (promoters first, terminator last).

    Extraction order follows model prose and can list CDS before RBS; we reassemble
    registry segments in this order before returning DNA to the UI."""
    th = (part_type_filter or "").strip().lower()
    if th == "promoter":
        return 10
    if th in ("operator", "insulator"):
        return 25
    if th == "rbs":
        return 40
    if th in ("cds", "reporter"):
        return 55
    if th == "terminator":
        return 85
    return 50


def extract_part_queries(thought: str) -> Tuple[List[Tuple[Optional[str], str]], str]:
    """Build (type_hint, query_text) pairs for RAG retrieval.

    Prefers explicit part identifiers (``BBa_*``, ``J#####``, ``B####``, ``sfGFP``,
    ``lacO``, …) so the registry sees crisp queries. Falls back to per-line extraction
    when no canonical names are present, with prompt-skeleton echoes filtered out.

    Returns ``(queries, parsing_strategy)`` where ``parsing_strategy`` is
    ``\"named-parts\"`` or ``\"free-text-lines\"``.
    """

    named = extract_part_names(thought)
    if named:
        return named, "named-parts"

    out: List[Tuple[Optional[str], str]] = []
    for line in extract_part_descriptions(thought):
        if _SKELETON_GARBAGE.search(line):
            continue
        m = re.match(
            r"(?i)^\s*(Promoter|RBS|CDS|Terminator)\s*[:.\-–]\s*(.+)$",
            line.strip(),
        )
        if m:
            type_hint = _TYPE_PREFIX_CANON.get(m.group(1).lower(), m.group(1))
            query_text = m.group(2).strip()
        else:
            type_hint = None
            query_text = line.strip()
        out.append((type_hint, query_text))
    return out, "free-text-lines"


def _part_display_label(type_hint: Optional[str], query_text: str) -> str:
    """Short label for UI (plasmid map) — strip trailing type word from enriched queries."""

    q = (query_text or "").strip()
    if not q:
        return "part"
    th = (type_hint or "").strip()
    if th:
        suffix = f" {th}"
        if len(q) > len(suffix) and q.endswith(suffix):
            core = q[: -len(suffix)].strip()
            if core:
                return core
    return q if len(q) <= 36 else q[:33] + "…"


def _part_map_subcategory(type_hint: Optional[str], query_text: str) -> str:
    """Normalize feature class for map coloring (matches inferFeatures ``sub`` semantics)."""

    th = (type_hint or "").strip().lower()
    if th in ("promoter", "rbs", "cds", "terminator"):
        return th
    qlow = (query_text or "").lower()
    if "operator" in qlow:
        return "operator"
    head = qlow.split()[0] if qlow.split() else ""
    if head in ("laco", "teto"):
        return "operator"
    return "feature"


def extract_part_map_slots(thought: str) -> List[Dict[str, str]]:
    """Ordered part labels for the circular map — same discovery order as :func:`extract_part_queries`."""

    queries, _strategy = extract_part_queries(thought)
    slots: List[Dict[str, str]] = []
    for type_hint, query_text in queries:
        slots.append(
            {
                "label": _part_display_label(type_hint, query_text),
                "sub": _part_map_subcategory(type_hint, query_text),
            }
        )
    return slots


def _split_sequence_chunks(seq: str, n: int) -> List[str]:
    if n <= 0:
        return []
    seq = "".join(seq.upper().split())
    L = len(seq)
    if L == 0:
        return [""] * n
    base = L // n
    rem = L % n
    chunks: List[str] = []
    pos = 0
    for i in range(n):
        take = base + (1 if i < rem else 0)
        chunks.append(seq[pos : pos + take])
        pos += take
    return chunks


def apply_rag_substitution(
    thought: str,
    model_sequence: str,
    *,
    progress_cb: Optional[Any] = None,
    log_context: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Merge registry DNA only when retrieval similarity ≥ :func:`min_similarity`.

    Each parsed part maps to one contiguous slice of the model sequence (equal-length
    chunks). Registry substitution applies **only** to verified hits; otherwise that
    slot keeps the model-derived DNA so no part is dropped from the construct.

    Returns ``(final_sequence, rag_detail_dict)``.
    """

    ctx = (log_context or "").strip() or "candidate"

    rag_always_log(
        f"apply_rag_substitution invoked for {ctx} "
        f"(model_seq_bp={len((model_sequence or '').strip())} thought_chars={len((thought or '').strip())})"
    )

    if not rag_enabled_env():
        seq = "".join((model_sequence or "").upper().split())
        rag_always_log(f"{ctx}: RAG disabled (DGENE_RAG) — using model sequence as-is")
        return seq, {"enabled": False, "reason": "DGENE_RAG disabled"}

    try:
        ensure_indexed(progress_cb=progress_cb)
    except Exception as exc:
        seq = "".join((model_sequence or "").upper().split())
        rag_always_log(f"{ctx}: RAG index/load FAILED — {exc!s}; using model sequence as-is")
        return seq, {"enabled": False, "error": str(exc)}

    _verify_alias_targets_once()

    seq_clean = "".join((model_sequence or "").upper().split())
    rag_always_log(f"{ctx}: SEQUENCE before RAG: {len(seq_clean)} bp")

    queries, parsing_strategy = extract_part_queries(thought)
    rag_always_log(
        f"{ctx}: PART QUERY EXTRACTION strategy={parsing_strategy!r} count={len(queries)}"
    )
    if queries and len(queries) <= 2 and len(seq_clean) >= 1000:
        rag_always_log(
            f"{ctx}: WARNING — only {len(queries)} slot(s) identified for a "
            f"{len(seq_clean)} bp model sequence. Verified registry hits will "
            f"replace large equal-chunk regions and any unrecognized parts "
            f"(custom promoters, novel sensors, CDSs missing from "
            f"`_GENERIC_PARTS`) will be lost. Add their canonical names to "
            f"`_GENERIC_PARTS` in igem_rag.py so they register as their own "
            f"slot and the model's DNA for that region is preserved."
        )
    th_prev = (thought or "").strip()
    if th_prev:
        cap = 4000
        if len(th_prev) > cap:
            reasoning_trace_log(
                f"{ctx}: MODEL REASONING ({len(th_prev)} chars, first {cap} shown):\n"
                f"{th_prev[:cap]}\n… [truncated]"
            )
        else:
            reasoning_trace_log(f"{ctx}: MODEL REASONING ({len(th_prev)} chars):\n{th_prev}")
    else:
        reasoning_trace_log(f"{ctx}: MODEL REASONING: (empty)")
    for qi, (type_hint, query_text) in enumerate(queries):
        rag_always_log(
            f"{ctx}:   extracted[{qi}] type_hint={type_hint!r} query={query_text!r}"
        )

    if not queries:
        rag_always_log(
            f"{ctx}: no part descriptions extracted from thought — "
            "skipping retrieval, keeping model sequence (sequences may be hallucinated)"
        )
        return seq_clean, {
            "enabled": True,
            "applied": False,
            "reason": "no_part_descriptions_in_thought",
            "parts": [],
            "assembly_audit": {
                "parsing_strategy": parsing_strategy,
                "identified_queries": [],
                "slot_count": 0,
                "slots_in_final_sequence": [],
                "final_sequence_bp": len(seq_clean),
                "parsed_vs_assembled_slot_match": True,
            },
        }

    n = len(queries)
    chunks = _split_sequence_chunks(seq_clean, n)
    merged: List[str] = []
    parts_out: List[Dict[str, Any]] = []
    retrieve_k = max(1, min(10, 5 if rag_debug_enabled() else 1))

    for i, (type_hint, query_text) in enumerate(queries):
        thr = min_similarity_for_slot(type_hint)
        model_slice = chunks[i] if i < len(chunks) else ""
        hits = retrieve_parts(query_text, part_type_filter=type_hint, top_k=retrieve_k)
        best = hits[0] if hits else None
        sim = best.similarity if best else 0.0

        if best is None:
            rag_always_log(
                f"{ctx}: SLOT[{i}] chroma_query={query_text!r} type_filter={type_hint!r} → "
                f"NO HITS | model_slice_bp={len(model_slice)} | substituted=False"
            )
        else:
            verified_pre = sim >= thr
            registry_seq_pre = "".join(best.sequence.upper().split())
            mk_pre = getattr(best, "match_kind", "semantic") or "semantic"
            rag_always_log(
                f"{ctx}: SLOT[{i}] chroma_query={query_text!r} type_filter={type_hint!r} → "
                f"hit={best.part_name!r} part_id={best.part_id} match_kind={mk_pre} "
                f"sim={sim:.4f} thr={thr:.4f} verified={verified_pre} "
                f"hit_bp={len(registry_seq_pre)} model_slice_bp={len(model_slice)} "
                f"substitution_applied={verified_pre} "
                f"dna_differs_from_model_slice={verified_pre and registry_seq_pre != model_slice}"
            )

        verified = best is not None and sim >= thr

        if (
            best is not None
            and not verified
            and (type_hint or "").strip().lower() == "promoter"
            and sim >= min_similarity()
        ):
            rag_always_log(
                f"{ctx}: SLOT[{i}] PROMOTER hit {best.part_name!r} sim={sim:.4f} "
                f"< promoter_threshold={thr:.4f} — treating as unverified (keep model slice)"
            )

        if verified:
            registry_seq = "".join(best.sequence.upper().split())
            mk = getattr(best, "match_kind", "semantic") or "semantic"
            dna_replaced = registry_seq != model_slice
            merged.append(registry_seq)
            parts_out.append(
                {
                    "query": query_text,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": True,
                    "unverified": False,
                    "substituted": True,
                    "similarity": round(sim, 4),
                    "part_id": best.part_id,
                    "part_name": best.part_name,
                    "part_type": best.part_type,
                    "short_desc": best.short_desc,
                    "match_kind": mk,
                    "sequence_source": "registry",
                    "model_slice_bp": len(model_slice),
                    "dna_replaced_vs_model_slice": dna_replaced,
                    "sequence": registry_seq,
                }
            )
            continue

        # iGEM miss or below similarity threshold — try NCBI Gene (bacterial CDS).
        sym = gene_symbol_eligible_for_ncbi(query_text, type_hint)
        ncbi_hit = None
        if sym:
            try:
                ncbi_hit = fetch_gene_cds(
                    sym,
                    thought=thought or "",
                    log=lambda m: rag_always_log(f"{ctx}: {m}"),
                )
            except Exception as exc:  # noqa: BLE001
                rag_always_log(f"{ctx}: NCBI exception symbol={sym!r}: {exc!s}")

        if ncbi_hit:
            nseq = "".join(ncbi_hit.sequence.upper().split())
            dna_ncbi = nseq != model_slice
            rag_always_log(
                f"{ctx}: SLOT[{i}] NCBI substitution symbol={sym!r} gene_id={ncbi_hit.gene_id} "
                f"name={ncbi_hit.gene_name!r} org={ncbi_hit.organism!r} "
                f"{ncbi_hit.accession}:{ncbi_hit.seq_from}-{ncbi_hit.seq_to} "
                f"bp={len(nseq)} (iGEM verified=False)"
            )
            merged.append(nseq)
            parts_out.append(
                {
                    "query": query_text,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": True,
                    "unverified": False,
                    "substituted": True,
                    "similarity": 1.0,
                    "part_id": ncbi_hit.gene_id,
                    "part_name": f"NCBI:{ncbi_hit.gene_name}",
                    "part_type": "CDS",
                    "short_desc": (
                        f"NCBI Gene {ncbi_hit.gene_id} ({ncbi_hit.organism}) "
                        f"{ncbi_hit.accession}:{ncbi_hit.seq_from}-{ncbi_hit.seq_to}"
                    ),
                    "match_kind": "ncbi-gene",
                    "sequence_source": "ncbi",
                    "ncbi_gene_id": ncbi_hit.gene_id,
                    "ncbi_gene_name": ncbi_hit.gene_name,
                    "ncbi_organism": ncbi_hit.organism,
                    "ncbi_accession": ncbi_hit.accession,
                    "ncbi_range": f"{ncbi_hit.seq_from}-{ncbi_hit.seq_to}",
                    "ncbi_strand": ncbi_hit.strand,
                    "model_slice_bp": len(model_slice),
                    "dna_replaced_vs_model_slice": dna_ncbi,
                    "sequence": nseq,
                }
            )
            continue

        if best is None:
            merged.append(model_slice)
            parts_out.append(
                {
                    "query": query_text,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": False,
                    "unverified": True,
                    "substituted": False,
                    "similarity": None,
                    "best_candidate": None,
                    "match_kind": None,
                    "sequence_source": "model",
                    "model_slice_bp": len(model_slice),
                    "dna_replaced_vs_model_slice": False,
                    "sequence": model_slice,
                }
            )
        else:
            registry_seq = "".join(best.sequence.upper().split())
            mk = getattr(best, "match_kind", "semantic") or "semantic"
            merged.append(model_slice)
            parts_out.append(
                {
                    "query": query_text,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": False,
                    "unverified": True,
                    "substituted": False,
                    "similarity": round(sim, 4),
                    "part_id": best.part_id,
                    "part_name": best.part_name,
                    "part_type": best.part_type,
                    "short_desc": best.short_desc,
                    "match_kind": mk,
                    "sequence_source": "model",
                    "reject_reason": "below_similarity_threshold",
                    "registry_candidate_bp": len(registry_seq),
                    "model_slice_bp": len(model_slice),
                    "dna_replaced_vs_model_slice": False,
                    "sequence": model_slice,
                }
            )

    idx_order = sorted(
        range(len(parts_out)),
        key=lambda i: (_cassette_tier(parts_out[i].get("part_type_filter")), i),
    )
    if idx_order != list(range(len(parts_out))):
        perm = ",".join(str(i) for i in idx_order)
        rag_always_log(
            f"{ctx}: cassette_order — reassembled 5′→3′ "
            f"(Promoter→RBS→CDS→Terminator); slot permutation [{perm}]"
        )
        merged = [merged[i] for i in idx_order]
        parts_out = [parts_out[i] for i in idx_order]
        queries_ordered: List[Tuple[Optional[str], str]] = [queries[i] for i in idx_order]
    else:
        queries_ordered = list(queries)

    map_slots_out: List[Dict[str, str]] = []
    for type_hint, query_text in queries_ordered:
        map_slots_out.append(
            {
                "label": _part_display_label(type_hint, query_text),
                "sub": _part_map_subcategory(type_hint, query_text),
            }
        )

    final = "".join(merged)
    verified_count = sum(1 for p in parts_out if p.get("verified"))
    sequence_changed = final != seq_clean
    sub_n = sum(1 for p in parts_out if p.get("substituted"))
    rag_always_log(
        f"{ctx}: SEQUENCE after RAG: {len(final)} bp "
        f"(delta {len(final) - len(seq_clean):+d} bp vs before; "
        f"content_changed={sequence_changed}; substituted_slots={sub_n}/{len(parts_out)})"
    )

    slot_audit: List[Dict[str, Any]] = []
    for i, p in enumerate(parts_out):
        seq_bp = len(p.get("sequence") or "")
        src = p.get("sequence_source") or ("registry" if p.get("substituted") else "model")
        slot_audit.append(
            {
                "slot_index": i,
                "query": p.get("query"),
                "sequence_bp": seq_bp,
                "sequence_source": src,
                "verified": bool(p.get("verified")),
                "substituted": bool(p.get("substituted")),
            }
        )
        rag_always_log(
            f"{ctx}: assembly_audit slot[{i}] query={p.get('query')!r} "
            f"bp={seq_bp} source={src} substituted={bool(p.get('substituted'))}"
        )
    rag_always_log(
        f"{ctx}: assembly_audit summary parsing_strategy={parsing_strategy!r} "
        f"identified_slots={len(queries_ordered)} assembled_slots={len(parts_out)} "
        f"final_bp={len(final)} "
        f"parsed_vs_assembled_match={len(queries_ordered) == len(parts_out)}"
    )

    rag_debug_log(
        f"{ctx}: substitution summary — rag_min_sim={min_similarity():.3f} "
        f"promoter_min={min_similarity_for_slot('Promoter'):.3f} "
        f"verified_parts={verified_count}/{len(parts_out)} "
        f"model_bp={len(seq_clean)} final_bp={len(final)} "
        f"sequence_changed_vs_model={sequence_changed} "
        f"(registry DNA merged before compiler passes / API response)"
    )

    return final, {
        "enabled": True,
        "applied": True,
        "min_similarity": min_similarity(),
        "min_similarity_promoter": min_similarity_for_slot("Promoter"),
        "parts": parts_out,
        "model_sequence_bp": len(seq_clean),
        "final_sequence_bp": len(final),
        "verified_part_count": verified_count,
        "sequence_changed_from_model": sequence_changed,
        "assembly_audit": {
            "parsing_strategy": parsing_strategy,
            "identified_queries": [q for (_, q) in queries_ordered],
            "slot_count": len(queries_ordered),
            "slots_in_final_sequence": slot_audit,
            "final_sequence_bp": len(final),
            "parsed_vs_assembled_slot_match": len(queries_ordered) == len(parts_out),
        },
        "map_slots": map_slots_out,
    }
