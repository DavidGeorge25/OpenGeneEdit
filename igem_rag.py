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

    # De-dupe while preserving order
    seen = set()
    out: List[str] = []
    for c in collected:
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
    queries = extract_part_descriptions(thought)
    rag_always_log(
        f"{ctx}: extracted {len(queries)} part line(s) from thought for retrieval"
    )
    for qi, line in enumerate(queries):
        rag_debug_log(f"{ctx}: part_query[{qi}]={line!r}")

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

    for i, q in enumerate(queries):
        type_hint = None
        m = re.match(
            r"(?i)^\s*(Promoter|RBS|CDS|Terminator)\s*[:.\-–]\s*(.+)$",
            q.strip(),
        )
        query_text = q.strip()
        if m:
            raw_t = m.group(1).strip()
            type_hint = _TYPE_PREFIX_CANON.get(raw_t.lower(), raw_t)
            query_text = m.group(2).strip()

        hits = retrieve_parts(query_text, part_type_filter=type_hint, top_k=retrieve_k)
        best = hits[0] if hits else None
        sim = best.similarity if best else 0.0

        if best and sim >= thr:
            merged.append("".join(best.sequence.upper().split()))
            rag_debug_log(
                f"{ctx}: slot[{i}] SUBSTITUTED iGEM sequence bp={len(merged[-1])} "
                f"sim={sim:.4f} (≥ {thr:.4f}) part={best.part_name!r} id={best.part_id!r}"
            )
            parts_out.append(
                {
                    "query": q,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": True,
                    "similarity": round(sim, 4),
                    "part_id": best.part_id,
                    "part_name": best.part_name,
                    "part_type": best.part_type,
                    "short_desc": best.short_desc,
                    "sequence": merged[-1],
                }
            )
        else:
            fallback = chunks[i] if i < len(chunks) else ""
            merged.append(fallback)
            sim_s = f"{sim:.4f}" if best else "n/a"
            rag_debug_log(
                f"{ctx}: slot[{i}] KEPT model DNA slice bp={len(fallback)} "
                f"best_sim={sim_s} (need ≥ {thr:.4f})"
            )
            parts_out.append(
                {
                    "query": q,
                    "retrieval_query": query_text,
                    "part_type_filter": type_hint,
                    "verified": False,
                    "unverified": True,
                    "similarity": round(sim, 4) if best else None,
                    "best_candidate": (
                        {
                            "part_id": best.part_id,
                            "part_name": best.part_name,
                            "part_type": best.part_type,
                            "short_desc": best.short_desc,
                        }
                        if best
                        else None
                    ),
                    "sequence": fallback,
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
