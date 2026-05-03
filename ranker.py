"""Score candidates from pass results and compute the Pareto-optimal set.

Objectives (all on [0,1] for clean Pareto comparisons):

  - expression   ↑ better — derived from CAI and RBS strength
  - low_burden   ↑ better — derived from CAI penalty, length penalty, repeat count
  - gc_balance   ↑ better — 1 - 2|GC - 0.5|
  - cleanliness  ↑ better — penalizes Type IIS sites (Golden Gate friction)

The composite score is a weighted sum used to break ties within the same
pipeline / prompt-fit band. ``best_id`` sorts by pipeline tier (verified topology
first), then prompt token overlap with assembly metadata, then composite. The
Pareto front is still computed only from the four sequence objectives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from passes import PassResult


@dataclass
class CandidateScores:
    expression: float
    low_burden: float
    gc_balance: float
    cleanliness: float
    composite: float


# Weights for sequence composite (breaks ties after pipeline + prompt fit).
WEIGHTS = {
    "expression":  0.40,
    "low_burden":  0.25,
    "gc_balance":  0.15,
    "cleanliness": 0.20,
}


def _get_metric(passes: List[PassResult], pass_id: str, default: float = 0.0) -> float:
    for p in passes:
        if p.pass_id == pass_id and p.metric is not None:
            return float(p.metric)
    return default


def _diag_count(passes: List[PassResult], pass_id: str, severity: str = "warn") -> int:
    for p in passes:
        if p.pass_id == pass_id:
            return sum(1 for d in p.diagnostics if d.severity == severity)
    return 0


def score_candidate(passes: List[PassResult], seq_len: int) -> CandidateScores:
    cai = _get_metric(passes, "cai", 0.5)
    rbs = _get_metric(passes, "rbs", 0.5)
    gc = _get_metric(passes, "gc", 0.5)

    repeats = _diag_count(passes, "repeats", "warn")
    type_iis = _diag_count(passes, "type_iis", "warn")

    expression = max(0.0, min(1.0, 0.55 * cai + 0.45 * rbs))

    # Burden penalties: lower CAI raises burden; long constructs raise burden;
    # repeats raise loss probability and thus effective burden.
    burden_raw = (1.0 - cai) * 0.55 + min(1.0, seq_len / 4000.0) * 0.25 + min(1.0, repeats / 8.0) * 0.20
    low_burden = max(0.0, min(1.0, 1.0 - burden_raw))

    gc_balance = max(0.0, min(1.0, gc))

    cleanliness = max(0.0, min(1.0, 1.0 - min(1.0, type_iis / 4.0)))

    composite = (
        WEIGHTS["expression"]  * expression
        + WEIGHTS["low_burden"]  * low_burden
        + WEIGHTS["gc_balance"]  * gc_balance
        + WEIGHTS["cleanliness"] * cleanliness
    )

    return CandidateScores(
        expression=round(expression, 4),
        low_burden=round(low_burden, 4),
        gc_balance=round(gc_balance, 4),
        cleanliness=round(cleanliness, 4),
        composite=round(composite, 4),
    )


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """True iff `a` weakly dominates `b` (>= on all objectives, > on at least one)."""
    keys = ("expression", "low_burden", "gc_balance", "cleanliness")
    ge_all = all(a[k] >= b[k] for k in keys)
    gt_any = any(a[k] > b[k] for k in keys)
    return ge_all and gt_any


def pareto_front_ids(candidates: List[Dict]) -> List[str]:
    pareto: List[str] = []
    for i, c in enumerate(candidates):
        ci = c["scores"]
        dominated = False
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if _dominates(other["scores"], ci):
                dominated = True
                break
        if not dominated:
            pareto.append(c["id"])
    return pareto


def scores_to_dict(s: CandidateScores) -> Dict[str, float]:
    return asdict(s)


# Small stopword strip so “Design a plasmid…” doesn’t drown signal.
_ALIGNMENT_STOP = frozenset(
    {
        "design",
        "research",
        "prototype",
        "plasmid",
        "construct",
        "using",
        "with",
        "when",
        "both",
        "only",
        "from",
        "that",
        "this",
        "have",
        "high",
        "minimal",
        "standard",
        "biology",
        "synthetic",
        "coli",
        "e",
        "cell",
        "cells",
        "turn",
        "into",
        "your",
        "brief",
        "gate",
        "logic",
        "boolean",
    }
)


def pipeline_tier(rag: Optional[Dict[str, Any]]) -> int:
    """Higher = closer to formal / structured compilation (used for ``best_id`` ordering)."""

    if not isinstance(rag, dict):
        return 0
    pipe = str(rag.get("pipeline") or "").strip()
    if pipe == "circuit_synth":
        return 3
    if pipe == "slot_template":
        return 2
    if pipe == "rag_first":
        return 1
    return 0


def _alignment_tokens(prompt: str) -> List[str]:
    p = (prompt or "").lower()
    raw = re.findall(r"[a-z][a-z0-9]{3,}", p)
    return [t for t in raw if t not in _ALIGNMENT_STOP]


def alignment_haystack(thought: str, rag: Optional[Dict[str, Any]]) -> str:
    chunks: List[str] = [thought or ""]
    if isinstance(rag, dict):
        intent = rag.get("intent")
        if isinstance(intent, dict):
            for key in ("input_analytes", "gate", "reporter", "logic_summary", "notes"):
                v = intent.get(key)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    chunks.extend(str(x) for x in v)
                else:
                    chunks.append(str(v))
        for p in rag.get("parts") or []:
            if isinstance(p, dict):
                chunks.append(str(p.get("part_name") or ""))
                chunks.append(str(p.get("query") or ""))
        for slot in rag.get("map_slots") or []:
            if isinstance(slot, dict):
                chunks.append(str(slot.get("label") or ""))
                chunks.append(str(slot.get("part_name") or ""))
                chunks.append(str(slot.get("part_type") or ""))
        ov = rag.get("ordered_part_names")
        if isinstance(ov, list):
            chunks.extend(str(x) for x in ov)
    return " ".join(chunks)


def prompt_alignment_score(prompt: str, thought: str, rag: Optional[Dict[str, Any]]) -> float:
    """Share of salient prompt words that appear in assembly metadata (0–1)."""

    toks = _alignment_tokens(prompt)
    if not toks:
        return 0.5
    blob = alignment_haystack(thought, rag).lower()
    hits = sum(1 for t in toks if t in blob)
    return max(0.0, min(1.0, hits / len(toks)))


def attach_fidelity_scores(row_scores: Dict[str, float], *, prompt: str, row: Dict[str, Any]) -> None:
    """Mutate ``row_scores`` with ``prompt_alignment`` / ``pipeline_tier`` (preserves composite)."""

    rag = row.get("rag")
    if not isinstance(rag, dict):
        rag = None
    row_scores["pipeline_tier"] = float(pipeline_tier(rag))
    row_scores["prompt_alignment"] = round(
        prompt_alignment_score(prompt, str(row.get("thought") or ""), rag), 4
    )


def rank(candidates: List[Dict]) -> List[Dict]:
    """Sort for default ``best_id``: pipeline tier → prompt overlap → composite; then Pareto flags."""

    pareto = set(pareto_front_ids(candidates))

    def sort_key(c: Dict) -> tuple:
        s = c.get("scores") or {}
        tier = float(s.get("pipeline_tier", 0))
        align = float(s.get("prompt_alignment", 0))
        comp = float(s.get("composite", 0))
        return (-tier, -align, -comp)

    ordered = sorted(candidates, key=sort_key)
    for i, c in enumerate(ordered):
        c["rank"] = i + 1
        c["is_pareto"] = c["id"] in pareto
    return ordered
