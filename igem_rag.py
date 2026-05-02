"""iGEM parts RAG: JSONL → ChromaDB + sentence-transformers retrieval.

Dataset: ``igem_dataset.jsonl`` in the project root (``part_id``, ``part_name``,
``part_type``, ``short_desc``, ``sequence``).

Environment:

  • ``DGENE_IGEM_JSONL`` — override path to JSONL (default: beside this package).
  • ``DGENE_CHROMA_PATH`` — persistent Chroma directory (default: ``.chroma_igem``).
  • ``DGENE_RAG`` — set ``0`` / ``false`` to disable substitution in the compile pipeline.
  • ``DGENE_RAG_MIN_SIM`` — minimum cosine similarity to substitute (default ``0.6``).

Logging (stderr): set ``DGENE_RAG_DEBUG=1`` or ``DGENE_DEBUG=1`` to print retrieval queries,
hit lists with similarity scores, and whether registry sequences replaced model DNA.
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

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSONL = os.path.join(_MODULE_DIR, "igem_dataset.jsonl")
_DEFAULT_CHROMA = os.path.join(_MODULE_DIR, ".chroma_igem")
_EMBED_MODEL = "all-MiniLM-L6-v2"
_COLLECTION = "igem_parts"

_LOCK = threading.Lock()
_CLIENT = None
_COLLECTION_HANDLE = None
_MODEL = None


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


def _rag_env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def rag_debug_enabled() -> bool:
    return _rag_env_bool("DGENE_RAG_DEBUG", False) or _rag_env_bool("DGENE_DEBUG", False)


_RAG_DEBUG_MIRROR: Any = None


def set_rag_debug_mirror(cb: Optional[Any]) -> None:
    """Optional callback (e.g. async compile job line) — same text as :func:`rag_debug_log` when debug is on."""

    global _RAG_DEBUG_MIRROR
    _RAG_DEBUG_MIRROR = cb


def rag_debug_log(line: str) -> None:
    if not rag_debug_enabled():
        return
    ts = time.strftime("%H:%M:%S")
    sys.stderr.write(f"[dgene/rag {ts}] {line}\n")
    sys.stderr.flush()
    if _RAG_DEBUG_MIRROR is not None:
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
    sys.stderr.write(f"[dgene/rag {ts}] {line}\n")
    sys.stderr.flush()
    if _RAG_DEBUG_MIRROR is not None:
        try:
            _RAG_DEBUG_MIRROR(line)
        except Exception:
            pass


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


@dataclass
class RetrievedPart:
    part_name: str
    part_type: str
    short_desc: str
    sequence: str
    similarity: float
    part_id: str = ""


def retrieve_parts(
    query: str,
    part_type_filter: Optional[str] = None,
    top_k: int = 3,
) -> List[RetrievedPart]:
    """Plain-English (or keyword) query → top matching registry parts.

    Cosine distance ``d`` from Chroma on normalized embeddings yields
    similarity ``1 - d`` (identical to cosine similarity for unit vectors).
    """

    q = (query or "").strip()
    filt_suffix = (
        f" (part_type_filter={part_type_filter!r})" if part_type_filter else ""
    )
    rag_always_log(f"RAG called for: {q!r}{filt_suffix} top_k={top_k}")
    if not q:
        rag_always_log("RAG called with empty query — returning [] (no retrieval performed)")
        return []

    ensure_indexed()
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

    rag_debug_log(
        f"retrieve query={q!r} part_type_filter={part_type_filter!r} top_k={kwargs['n_results']}"
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
            f"retrieve hit #{i + 1} sim={rp.similarity:.4f} part_id={rp.part_id!r} "
            f"name={rp.part_name!r} type={rp.part_type!r} desc={desc!r} seq_bp={len(rp.sequence)}"
        )
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
    ("sfGFP", "CDS"),
    ("eGFP", "CDS"),
    ("mGFP", "CDS"),
    ("GFP", "CDS"),
    ("mRFP1", "CDS"),
    ("mRFP", "CDS"),
    ("RFP", "CDS"),
    ("mCherry", "CDS"),
    ("YFP", "CDS"),
    ("CFP", "CDS"),
    ("luxAB", "CDS"),
    ("luxCDABE", "CDS"),
    ("lacI", "CDS"),
    ("tetR", "CDS"),
    ("araC", "CDS"),
    ("cI", "CDS"),
    ("PbrR", "CDS"),
    ("lacO", None),
    ("tetO", None),
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


def _classify_window(window: str) -> Optional[str]:
    """Return the part type whose keyword appears at the smallest offset in ``window``.

    Picking the nearest keyword (rather than a fixed priority order) keeps each part name
    bound to *its* type word — so in ``"B0034 RBS, sfGFP coding sequence, B0015 terminator"``
    each name resolves to the right type even though all three keywords are in range.
    """

    padded = f" {window} "
    best: Optional[Tuple[int, str]] = None
    for needle, label in _TYPE_KEYWORDS:
        idx = padded.find(needle)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, label)
    return best[1] if best else None


def _scan_window_for_type(
    text: str,
    start: int,
    end: int,
    *,
    span_after: int = 40,
    span_before: int = 20,
) -> Optional[str]:
    """Look around a regex match for a part-type keyword.

    iGEM convention is ``<part_name> <type_word>`` (``B0034 RBS``, ``B0015 terminator``,
    ``J23100 promoter``). Prefer the chars *immediately after* the match, then fall back
    to a small window *before* it. A narrow window prevents one type word at the start
    of a long sentence from contaminating every part name in the rest of the sentence.
    """

    after = text[end : end + span_after].lower()
    hit_after = _classify_window(after)
    if hit_after is not None:
        return hit_after
    before = text[max(0, start - span_before) : start].lower()
    return _classify_window(before)


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


def extract_part_names(thought: str) -> List[Tuple[Optional[str], str]]:
    """Scan reasoning for explicit iGEM identifiers + canonical generic names.

    Returns ``[(type_hint, query_text), ...]`` in document order, deduped. The
    ``query_text`` is enriched with the type word so the embedding model gets a
    stronger signal (e.g. ``"B0034 RBS"`` rather than bare ``"B0034"``).
    """

    text = thought or ""
    if not text.strip():
        return []

    found: List[Tuple[Optional[str], str]] = []
    seen: set = set()

    def push(type_hint: Optional[str], name: str) -> None:
        key = (type_hint or "", name.lower())
        if key in seen:
            return
        seen.add(key)
        q = f"{name} {type_hint}".strip() if type_hint else name
        found.append((type_hint, q))

    for m in _BBA_RE.finditer(text):
        push(_scan_window_for_type(text, m.start(), m.end()), m.group(0))

    for m in _J_PROMOTER_RE.finditer(text):
        push("Promoter", m.group(0))

    for m in _B_PART_RE.finditer(text):
        name = m.group(0)
        type_hint = _scan_window_for_type(text, m.start(), m.end()) or _b_part_type_prior(name)
        push(type_hint, name)

    for canonical, default_type in _GENERIC_PARTS:
        for m in re.finditer(rf"\b{re.escape(canonical)}\b", text, flags=re.IGNORECASE):
            type_hint = _scan_window_for_type(text, m.start(), m.end()) or default_type
            push(type_hint, canonical)

    return found


def extract_part_queries(thought: str) -> List[Tuple[Optional[str], str]]:
    """Build (type_hint, query_text) pairs for RAG retrieval.

    Prefers explicit part identifiers (``BBa_*``, ``J#####``, ``B####``, ``sfGFP``,
    ``lacO``, …) so the registry sees crisp queries. Falls back to per-line extraction
    when no canonical names are present, with prompt-skeleton echoes filtered out.
    """

    named = extract_part_names(thought)
    if named:
        return named

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
    return out


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
    """Replace model DNA with concatenated iGEM sequences where similarity is high enough.

    For descriptions below the similarity threshold, keeps the corresponding
    slice of the **model** sequence (equal-length contiguous chunks).

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

    seq_clean = "".join((model_sequence or "").upper().split())
    queries = extract_part_queries(thought)
    rag_always_log(
        f"{ctx}: extracted {len(queries)} part query(ies) from thought "
        f"(strategy={'named-parts' if extract_part_names(thought) else 'free-text-lines'})"
    )
    for qi, (type_hint, query_text) in enumerate(queries):
        rag_always_log(
            f"{ctx}: part_query[{qi}] type_hint={type_hint!r} query={query_text!r}"
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
        }

    thr = min_similarity()
    n = len(queries)
    chunks = _split_sequence_chunks(seq_clean, n)
    merged: List[str] = []
    parts_out: List[Dict[str, Any]] = []
    retrieve_k = max(1, min(10, 5 if rag_debug_enabled() else 1))

    for i, (type_hint, query_text) in enumerate(queries):
        hits = retrieve_parts(query_text, part_type_filter=type_hint, top_k=retrieve_k)
        best = hits[0] if hits else None
        sim = best.similarity if best else 0.0

        if best is None:
            rag_always_log(
                f"RAG result: <none> — registry returned no hits for {query_text!r}"
            )
            fallback = chunks[i] if i < len(chunks) else ""
            merged.append(fallback)
            rag_always_log(
                f"{ctx}: slot[{i}] KEPT model DNA slice bp={len(fallback)} "
                f"(no registry hits at all — sequence may be hallucinated)"
            )
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
                    "sequence": fallback,
                }
            )
            continue

        verified = sim >= thr
        verdict = "VERIFIED" if verified else "unverified-but-substituted"
        rag_always_log(
            f"RAG result: {best.part_id} ({best.part_name}, type={best.part_type}) "
            f"similarity={sim:.4f} verdict={verdict}"
        )

        registry_seq = "".join(best.sequence.upper().split())
        merged.append(registry_seq)
        if verified:
            rag_always_log(
                f"{ctx}: slot[{i}] SUBSTITUTED bp={len(registry_seq)} "
                f"sim={sim:.4f} (≥ {thr:.4f}) part={best.part_name!r} id={best.part_id!r} VERIFIED"
            )
        else:
            rag_always_log(
                f"{ctx}: slot[{i}] SUBSTITUTED bp={len(registry_seq)} "
                f"sim={sim:.4f} (< {thr:.4f}) part={best.part_name!r} id={best.part_id!r} "
                "unverified — registry DNA preferred over hallucinated model slice"
            )
        parts_out.append(
            {
                "query": query_text,
                "retrieval_query": query_text,
                "part_type_filter": type_hint,
                "verified": verified,
                "unverified": not verified,
                "substituted": True,
                "similarity": round(sim, 4),
                "part_id": best.part_id,
                "part_name": best.part_name,
                "part_type": best.part_type,
                "short_desc": best.short_desc,
                "sequence": registry_seq,
            }
        )

    final = "".join(merged)
    verified_count = sum(1 for p in parts_out if p.get("verified"))
    sequence_changed = final != seq_clean
    rag_debug_log(
        f"{ctx}: substitution summary — threshold={thr:.3f} verified_parts={verified_count}/"
        f"{len(parts_out)} model_bp={len(seq_clean)} final_bp={len(final)} "
        f"sequence_changed_vs_model={sequence_changed} "
        f"(registry DNA merged before compiler passes / API response)"
    )

    return final, {
        "enabled": True,
        "applied": True,
        "min_similarity": thr,
        "parts": parts_out,
        "model_sequence_bp": len(seq_clean),
        "final_sequence_bp": len(final),
        "verified_part_count": verified_count,
        "sequence_changed_from_model": sequence_changed,
    }
