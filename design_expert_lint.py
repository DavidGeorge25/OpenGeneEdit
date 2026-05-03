"""Deterministic 'expert sniff test' for linear BioBrick orders.

Uses the same cognate promoter ↔ TF relationships as :mod:`circuit_parts` (not full simulation).
If a regulated promoter appears in the construct, the corresponding regulator CDS must appear
somewhere in the same ordered list. This catches the most common RAG-first failure mode: chemically
plausible sentences glued to unrelated regulatory parts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from circuit_parts import PROMOTERS, TFS
from igem_rag import _PART_ALIASES

_TERMINATOR_BBAS = frozenset(
    {
        "BBa_B0015",
        "BBa_B0010",
        "BBa_B0012",
        "BBa_B0011",
        "BBa_B0013",
    }
)


def _normalize_bba(part_name: str) -> str:
    pn = (part_name or "").strip()
    if not pn.startswith("BBa_"):
        return pn
    tail = pn[4:].casefold()
    return _PART_ALIASES.get(tail) or pn


def _promoter_requirement_rules() -> List[Dict[str, object]]:
    rules: List[Dict[str, object]] = []
    for pm in PROMOTERS.values():
        need: List[str] = []
        for tf_name in list(pm.repressed_by or []) + list(pm.activated_by or []):
            tf = TFS.get(tf_name)
            if tf and tf.bba_id not in need:
                need.append(tf.bba_id)
        if need:
            who: List[str] = []
            if pm.repressed_by:
                who.append("repressed by " + ", ".join(pm.repressed_by))
            if pm.activated_by:
                who.append("activated by " + ", ".join(pm.activated_by))
            rules.append(
                {
                    "promoter_bba": pm.bba_id,
                    "promoter_name": pm.name,
                    "need_cds_bba": need,
                    "rationale": "; ".join(who),
                }
            )
    return rules


_RULES: Optional[List[Dict[str, object]]] = None


def _rules_cached() -> List[Dict[str, object]]:
    global _RULES
    if _RULES is None:
        _RULES = _promoter_requirement_rules()
    return _RULES


def lint_ordered_construct(ordered_bb_as: List[str]) -> dict:
    """Return a JSON-serializable audit dict for UI and snapshots."""

    raw = [str(x).strip() for x in (ordered_bb_as or []) if str(x).strip()]
    normalized: List[str] = [_normalize_bba(x) for x in raw]
    bset: Set[str] = set(normalized)

    issues: List[dict] = []
    rules = _rules_cached()
    promoters_seen: List[str] = []
    rules_fired = 0

    for rule in rules:
        pb = str(rule["promoter_bba"])
        if pb not in bset:
            continue
        promoters_seen.append(f'{rule["promoter_name"]} ({pb})')
        rules_fired += 1
        missing = [nb for nb in rule["need_cds_bba"] if nb not in bset]
        if missing:
            issues.append(
                {
                    "severity": "error",
                    "code": "missing_regulator_cds",
                    "message": (
                        f"Promoter {rule['promoter_name']} ({pb}) is present but the construct "
                        f"does not include the usual cognate regulator CDS "
                        f"({', '.join(missing)}). A specialist would expect those TFs on the same "
                        f"plasmid (or supplied in host) for this promoter to behave as intended."
                    ),
                    "promoter_bba": pb,
                    "missing_cds_bba": missing,
                }
            )

    term_hit = bool(bset & _TERMINATOR_BBAS)
    if not term_hit and len(raw) >= 4:
        issues.append(
            {
                "severity": "warn",
                "code": "no_common_terminator",
                "message": (
                    "No common strong terminator (e.g. BBa_B0015) was found by ID. "
                    "If terminators are embedded inside composite parts, this can be a false alarm."
                ),
            }
        )

    n_err = sum(1 for i in issues if i.get("severity") == "error")
    n_warn = sum(1 for i in issues if i.get("severity") == "warn")
    score = max(0.0, min(1.0, 1.0 - 0.22 * n_err - 0.06 * n_warn))

    if score >= 0.85 and n_err == 0:
        grade = "strong"
    elif n_err == 0:
        grade = "mixed"
    else:
        grade = "weak"

    summary_parts = [
        f"Cognate regulator rules checked for {len(rules)} promoter(s) in the curated table.",
    ]
    if rules_fired == 0:
        summary_parts.append(
            "No catalog-regulated promoter IDs from the strict rule set were recognized "
            "(or only constitutive / composite parts were used)."
        )
    elif n_err == 0:
        summary_parts.append(
            "All triggered promoter ↔ regulator pairings are satisfied by IDs in this order."
        )
    else:
        summary_parts.append(
            f"{n_err} regulatory consistency issue(s) would likely draw review comments."
        )

    return {
        "score": round(score, 3),
        "grade": grade,
        "summary": " ".join(summary_parts),
        "part_count": len(raw),
        "promoters_matched_rules": promoters_seen,
        "issues": issues,
        "rules_triggered": rules_fired,
    }
