"""Orchestrator: verified circuit synthesis first, then RAG-first variants to fill the candidate count."""
from __future__ import annotations

from typing import Callable, Iterator, List, Optional

from circuit_intent import extract_circuit_spec, spec_to_dict

from circuit_synth import TopologyError, synthesize
from circuit_verify import verify_plasmid
from circuit_parts import BACKBONE


def _map_slots_to_parts(slots: List[dict]) -> List[dict]:
    out: List[dict] = []
    for s in slots:
        out.append(
            {
                "part_name": s.get("part_name"),
                "part_type": s.get("part_type"),
                "verified": bool(s.get("verified", True)),
                "sequence_source": s.get("sequence_source", "registry"),
                "similarity": 1.0,
                "query": "circuit_synth·vetted_catalog",
                "start_bp": s.get("start_bp"),
                "end_bp": s.get("end_bp"),
            }
        )
    return out


def _thought_from(spec_dict: dict, plasmid, passed: bool, summary: str) -> str:
    logic = spec_dict.get("logic") or {}
    ins = spec_dict.get("inputs") or []
    out = spec_dict.get("output") or {}
    gate = logic.get("op", "?")
    names = ", ".join(str(x.get("name")) for x in ins)
    rep = out.get("name", "?")
    status = "passed static boolean verification" if passed else "verification failed (should not ship)"
    notes = "\n".join(getattr(plasmid, "notes", []) or [])
    return (
        f"**Topology compiler · {gate}** over [{names}] → reporter **{rep}**. "
        f"{status}: {summary} "
        f"DNA is a linearized plasmid: ori + CmR + MCS → verified cassettes → MCS closure; "
        f"compare to distribution {plasmid.backbone.name} ({plasmid.backbone.bba_id}). "
        f"{notes}"
    )


def build_circuit_candidate(
    user_prompt: str,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional["Candidate"]:
    """Return a verified circuit candidate or ``None`` to fall back to RAG-first."""

    from inference import Candidate

    spec, intent_dict = extract_circuit_spec(user_prompt, progress_cb=progress_cb)
    if spec is None:
        if progress_cb:
            reason = intent_dict.get("reason") if isinstance(intent_dict, dict) else ""
            try:
                progress_cb(
                    f"circuit_synth · skip · {(reason or 'not a supported boolean design')[:120]}"
                )
            except Exception:
                pass
        return None

    try:
        if progress_cb:
            progress_cb(
                f"circuit_synth · synthesizing {spec.logic.op} · "
                f"{[i.name for i in spec.inputs]} → {spec.output.name}…"
            )
        plasmid = synthesize(spec)
    except TopologyError as exc:
        if progress_cb:
            try:
                progress_cb(f"circuit_synth · topology unavailable · {exc}")
            except Exception:
                pass
        return None

    ok, rows, summary = verify_plasmid(spec, plasmid)
    if not ok:
        if progress_cb:
            try:
                progress_cb(f"circuit_synth · verification FAILED · {summary}")
            except Exception:
                pass
        return None

    seq = plasmid.assembled_sequence()
    if not seq:
        return None

    spec_dict = spec_to_dict(spec)
    slots = plasmid.map_slots()
    detail = {
        "enabled": True,
        "applied": True,
        "pipeline": "circuit_synth",
        "min_similarity": 1.0,
        "intent": intent_dict,
        "circuit_spec": spec_dict,
        "map_slots": slots,
        "parts": _map_slots_to_parts(slots),
        "assembly_trace": slots,
        "verification": {
            "passes": True,
            "summary": summary,
            "truth_table": [{"inputs": list(bits), "expected": ex, "actual": act} for bits, ex, act in rows],
        },
        "backbone": {
            "name": plasmid.backbone.name,
            "bba_id": plasmid.backbone.bba_id,
            "note": plasmid.backbone.note,
            "ori_bba_id": BACKBONE.ori_bba_id,
            "resistance_bba_id": BACKBONE.resistance_bba_id,
            "mcs_prefix": BACKBONE.mcs_prefix,
            "mcs_suffix": BACKBONE.mcs_suffix,
        },
    }

    thought = _thought_from(spec_dict, plasmid, True, summary)
    return Candidate(
        candidate_id="cand_circuit_0",
        thought=thought,
        sequence=seq,
        strategy="circuit_synth verified",
        strategy_name="Topology compiler (truth-table verified)",
        raw=f"{thought}\n\n[assembled linear plasmid {len(seq)} bp]\n",
        rag_first_detail=detail,
    )


def compile_hybrid_variants(
    user_prompt: str,
    n: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> List["Candidate"]:
    """Up to one verified circuit candidate, then RAG-first for the remainder."""

    from circuit_rag_first import rag_first_configured, run_rag_first_variants

    out: List = []
    cc = build_circuit_candidate(user_prompt, progress_cb=progress_cb)
    if cc is not None:
        cc.candidate_id = "cand_0"
        out.append(cc)
    need = max(0, n - len(out))
    if need and rag_first_configured():
        if progress_cb:
            try:
                progress_cb(f"circuit_synth · adding {need} RAG-first variant(s)…")
            except Exception:
                pass
        rest = run_rag_first_variants(user_prompt, need, progress_cb=progress_cb)
        base = len(out)
        for i, c in enumerate(rest):
            c.candidate_id = f"cand_{base + i}"
            out.append(c)
    elif need and not rag_first_configured():
        if progress_cb:
            try:
                progress_cb("circuit_synth · WARN · no API key for RAG-first padding — fewer candidates")
            except Exception:
                pass
    return out


def compile_hybrid_variants_iter(
    user_prompt: str,
    n: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Iterator["Candidate"]:
    """Yield circuit candidate first (if any), then RAG-first iterator."""

    from circuit_rag_first import rag_first_configured, run_rag_first_variants_iter

    yielded = 0
    cc = build_circuit_candidate(user_prompt, progress_cb=progress_cb)
    if cc is not None:
        cc.candidate_id = f"cand_{yielded}"
        yield cc
        yielded += 1

    need = n - yielded
    if need <= 0:
        return

    if not rag_first_configured():
        if yielded == 0:
            raise RuntimeError(
                "Circuit synthesis produced no candidate and RAG-first needs GEMINI_API_KEY"
            )
        return

    if progress_cb:
        try:
            progress_cb(f"circuit_synth · streaming {need} RAG-first variant(s)…")
        except Exception:
            pass

    for c in run_rag_first_variants_iter(user_prompt, need, progress_cb=progress_cb):
        c.candidate_id = f"cand_{yielded}"
        yield c
        yielded += 1
