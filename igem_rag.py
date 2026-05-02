"""iGEM parts RAG: JSONL → ChromaDB + sentence-transformers retrieval.

Dataset: ``igem_dataset.jsonl`` in the project root (``part_id``, ``part_name``,
``part_type``, ``short_desc``, ``sequence``).

Environment:

  • ``DGENE_IGEM_JSONL`` — override path to JSONL (default: beside this package).
  • ``DGENE_CHROMA_PATH`` — persistent Chroma directory (default: ``.chroma_igem``).
  • ``DGENE_RAG`` — set ``0`` / ``false`` to disable substitution in the compile pipeline.
  • ``DGENE_RAG_MIN_SIM`` — minimum cosine similarity to **substitute** registry DNA for a
    slot (default ``0.6``). Below threshold the model's DNA slice for that slot is kept.

Logging (stderr): Every compile prints chroma query strings, ``ALIAS HIT:`` lines for the
alias table, per-slot hits (part_name, sim, bp), and sequence length before/after RAG.
Set ``DGENE_RAG_DEBUG=1`` or ``DGENE_DEBUG=1`` for extra retrieval hit lists.

Alias table (subset): ``luxR``→BBa_C0062, ``luxI``→BBa_C0061, ``plux``→BBa_R0062,
``mcherry``→BBa_E1010, ``j23100``→BBa_K4233030 (dataset snapshot has no BBa_J23100),
``b0034``→BBa_K812053 (no BBa_B0034), ``b0015``→BBa_B0015.
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
    ("LuxR", "CDS"),
    ("LuxI", "CDS"),
    ("Plux", "Promoter"),
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

    events: List[Tuple[int, Optional[str], str]] = []
    seen: set = set()

    def add(start: int, type_hint: Optional[str], name: str) -> None:
        key = (type_hint or "", name.lower())
        if key in seen:
            return
        seen.add(key)
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

    events.sort(key=lambda t: t[0])
    return [(th, q) for _, th, q in events]


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
    th_prev = (thought or "").strip()
    if th_prev:
        cap = 4000
        if len(th_prev) > cap:
            rag_always_log(
                f"{ctx}: MODEL REASONING ({len(th_prev)} chars, first {cap} shown):\n"
                f"{th_prev[:cap]}\n… [truncated]"
            )
        else:
            rag_always_log(f"{ctx}: MODEL REASONING ({len(th_prev)} chars):\n{th_prev}")
    else:
        rag_always_log(f"{ctx}: MODEL REASONING: (empty)")
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

    thr = min_similarity()
    n = len(queries)
    chunks = _split_sequence_chunks(seq_clean, n)
    merged: List[str] = []
    parts_out: List[Dict[str, Any]] = []
    retrieve_k = max(1, min(10, 5 if rag_debug_enabled() else 1))

    for i, (type_hint, query_text) in enumerate(queries):
        model_slice = chunks[i] if i < len(chunks) else ""
        hits = retrieve_parts(query_text, part_type_filter=type_hint, top_k=retrieve_k)
        best = hits[0] if hits else None
        sim = best.similarity if best else 0.0

        if best is None:
            rag_always_log(
                f"{ctx}: SLOT[{i}] chroma_query={query_text!r} type_filter={type_hint!r} → "
                f"NO HITS | model_slice_bp={len(model_slice)} | substituted=False"
            )
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
            continue

        verified = sim >= thr
        registry_seq = "".join(best.sequence.upper().split())
        mk = getattr(best, "match_kind", "semantic") or "semantic"
        dna_replaced = verified and registry_seq != model_slice

        rag_always_log(
            f"{ctx}: SLOT[{i}] chroma_query={query_text!r} type_filter={type_hint!r} → "
            f"hit={best.part_name!r} part_id={best.part_id} match_kind={mk} "
            f"sim={sim:.4f} thr={thr:.4f} verified={verified} "
            f"hit_bp={len(registry_seq)} model_slice_bp={len(model_slice)} "
            f"substitution_applied={verified} dna_differs_from_model_slice={dna_replaced}"
        )

        if verified:
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
        else:
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
        f"identified_slots={len(queries)} assembled_slots={len(parts_out)} "
        f"final_bp={len(final)} parsed_vs_assembled_match={len(queries) == len(parts_out)}"
    )

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
        "assembly_audit": {
            "parsing_strategy": parsing_strategy,
            "identified_queries": [q for (_, q) in queries],
            "slot_count": len(queries),
            "slots_in_final_sequence": slot_audit,
            "final_sequence_bp": len(final),
            "parsed_vs_assembled_slot_match": len(queries) == len(parts_out),
        },
    }
