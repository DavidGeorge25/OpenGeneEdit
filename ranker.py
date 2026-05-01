"""Score candidates from pass results and compute the Pareto-optimal set.

Objectives (all on [0,1] for clean Pareto comparisons):

  - expression   ↑ better — derived from CAI and RBS strength
  - low_burden   ↑ better — derived from CAI penalty, length penalty, repeat count
  - gc_balance   ↑ better — 1 - 2|GC - 0.5|
  - cleanliness  ↑ better — penalizes Type IIS sites (Golden Gate friction)

The composite score is a weighted sum used to break Pareto ties for sorting,
but the Pareto front is computed in objective space (no weights) so that
domination is honest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from passes import PassResult


@dataclass
class CandidateScores:
    expression: float
    low_burden: float
    gc_balance: float
    cleanliness: float
    composite: float


# Weights for composite score — used for sort order and "best" selection only.
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


def rank(candidates: List[Dict]) -> List[Dict]:
    """Return candidates sorted by composite desc, with `rank` and `is_pareto` added."""
    pareto = set(pareto_front_ids(candidates))
    ordered = sorted(candidates, key=lambda c: -c["scores"]["composite"])
    for i, c in enumerate(ordered):
        c["rank"] = i + 1
        c["is_pareto"] = c["id"] in pareto
    return ordered


def scores_to_dict(s: CandidateScores) -> Dict[str, float]:
    return asdict(s)
