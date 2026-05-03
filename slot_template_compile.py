"""Deterministic slot templates: gate + analytes → ordered Promoter/RBS/CDS/Terminator from iGEM RAG.

Used when intent JSON includes ``gate``, ``input_analytes``, and ``reporter`` so we assemble a small
linear cassette without equal-chunk post-hoc substitution. Complements ``circuit_rag_first`` menu+LLM
path — first variant can be slot-filled, then stochastic compiler picks fill diversity.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

_PROM_GATE_NOISE_TERMS_AND = (
    "or gate",
    "xor gate",
    "not gate",
)

_PROM_GATE_BAD_IF_OR = ("and gate", "dual input and", "two-input and")

_PROM_LEN_MAX_DEFAULT = 4000


def _part_type_for_map(part_type: str) -> str:
    """Map iGEM part_type to plasmid-map categories (mirror ``circuit_rag_first._part_type_to_map_sub``)."""

    t = (part_type or "").strip().lower()
    if not t:
        return "feature"
    if "promoter" in t:
        return "promoter"
    if "terminator" in t:
        return "terminator"
    if "rbs" in t or "ribosome" in t:
        return "rbs"
    if t == "cds" or "coding" in t or "protein domain" in t:
        return "cds"
    if "operator" in t:
        return "operator"
    if "origin" in t:
        return "backbone"
    return "feature"


def _slot_template_embed_backbone_enabled() -> bool:
    raw = (os.environ.get("DGENE_SLOT_TEMPLATE_EMBED_BACKBONE") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _trace_sub_for_backbone_map(seg: Dict[str, object]) -> str:
    pl = str(seg.get("part_type") or "").lower()
    if "origin" in pl:
        return "backbone"
    if pl == "backbone":
        return "backbone"
    return "feature"


def _slots_from_backbone_segments(
    segments: List[Dict[str, object]],
    *,
    pos0: int,
    source_tag: str,
) -> Tuple[List[dict], int]:
    out: List[dict] = []
    pos = pos0
    for seg in segments:
        L = int(seg.get("bp") or 0)
        if L <= 0:
            continue
        sub = _trace_sub_for_backbone_map(seg)
        out.append(
            {
                "part_name": str(seg.get("part_name") or "backbone"),
                "normalized_name": str(seg.get("part_name") or "backbone"),
                "part_type": str(seg.get("part_type") or "Backbone"),
                "label": str(seg.get("label") or seg.get("part_name") or "backbone"),
                "sub": sub,
                "ok": True,
                "bp": L,
                "source": source_tag,
                "similarity": 1.0,
                "retrieval_query": source_tag,
                "start_bp": pos,
                "end_bp": pos + L - 1,
                "verified": bool(seg.get("verified", True)),
                "sequence_source": str(seg.get("sequence_source") or "registry"),
            }
        )
        pos += L
    return out, pos


def embed_slot_template_in_ecoli_backbone(
    cassette_dna: str,
    cassette_trace: List[dict],
) -> Tuple[str, List[dict]]:
    """Surround the cassette with the same RFC10 coli backbone used by ``circuit_synth`` / ``circuit_ir``."""

    from circuit_parts import make_backbone_ref

    bb = make_backbone_ref()
    prefix_dna = "".join((bb.sequence or "").upper().split())
    suffix_dna = "".join((bb.suffix_sequence or "").upper().split())
    cassette_clean = "".join((cassette_dna or "").upper().split())
    full = prefix_dna + cassette_clean + suffix_dna

    prefix_slots, pos = _slots_from_backbone_segments(
        bb.prefix_segments, pos0=1, source_tag="slot_template_backbone"
    )
    off = len(prefix_dna)
    shifted: List[dict] = []
    for t in cassette_trace:
        u = dict(t)
        sb = u.get("start_bp")
        eb = u.get("end_bp")
        if sb is not None and eb is not None:
            u["start_bp"] = int(sb) + off
            u["end_bp"] = int(eb) + off
        shifted.append(u)

    if shifted:
        pos = int(shifted[-1]["end_bp"]) + 1  # type: ignore[arg-type]
    elif off > 0:
        pos = off + 1
    suffix_slots, _pos_after = _slots_from_backbone_segments(
        bb.suffix_segments, pos0=pos, source_tag="slot_template_backbone"
    )

    merged = prefix_slots + shifted + suffix_slots
    if merged:
        last = int(merged[-1]["end_bp"])  # type: ignore[arg-type]
        if last != len(full):
            raise ValueError(f"map_slots end_bp {last} != sequence len {len(full)}")
    return full, merged


def slot_template_enabled() -> bool:
    v = (os.environ.get("DGENE_SLOT_TEMPLATE") or "1").strip().lower()
    return v not in ("0", "false", "no")


def slot_template_min_promoter_similarity() -> float:
    raw = (os.environ.get("DGENE_SLOT_TEMPLATE_MIN_SIM") or "0.52").strip()
    try:
        return max(0.35, min(0.92, float(raw)))
    except ValueError:
        return 0.52


def slot_template_promoter_max_bp() -> int:
    raw = (os.environ.get("DGENE_SLOT_TEMPLATE_MAX_PROMOTER_BP") or str(_PROM_LEN_MAX_DEFAULT)).strip()
    try:
        return max(800, min(12000, int(raw)))
    except ValueError:
        return _PROM_LEN_MAX_DEFAULT


def normalize_gate(raw: Optional[str]) -> str:
    g = str(raw or "").strip().upper()
    if g in ("AND", "OR", "BUF", "NOT"):
        return g
    if g in ("BUFFER",):
        return "BUF"
    return "UNKNOWN"


def infer_gate_from_summary(intent: dict) -> str:
    ls = str(intent.get("logic_summary") or "").lower()
    combined = ls
    for r in intent.get("roles") or []:
        if isinstance(r, dict):
            combined += " " + str(r.get("summary") or "").lower()
    if re.search(r"\band\b|\btwo\b.*\bboth\b|\bboth\b.*\bins", combined):
        if " either " not in combined or "both" in combined:
            return "AND"
    if re.search(r"\bor\b|\beither\b", combined) and "both" not in combined:
        return "OR"
    return "UNKNOWN"


def extract_analytes(intent: dict) -> List[str]:
    raw = intent.get("input_analytes") or intent.get("inputs_analytes")
    out: List[str] = []
    if isinstance(raw, list):
        for x in raw:
            s = str(x).strip()
            if s and len(s) < 200:
                out.append(s)
    if out:
        return out[:6]
    for r in intent.get("roles") or []:
        if not isinstance(r, dict):
            continue
        role = str(r.get("role") or "").lower()
        if "sensor" not in role and "input" not in role and "signal" not in role:
            continue
        summary = str(r.get("summary") or "").strip()
        if summary and summary not in out:
            out.append(summary[:180])
        for q in r.get("retrieval_queries") or ():
            if isinstance(q, str) and len(q.strip()) > 6:
                qs = q.strip()[:140]
                if qs not in out and len(out) < 6:
                    out.append(qs)
                break
    return out[:4]


def normalize_reporter(raw: Optional[str], user_prompt: str) -> str:
    x = str(raw or "").strip()
    low = user_prompt.casefold()
    if not x:
        if "amilcp" in low or ("blue" in low and "pigment" in low) or "chromoprotein" in low:
            return "amilCP"
        if "gfp" in low or "green fluorescent" in low:
            return "GFP"
        if "mrfp" in low or "red fluorescent" in low:
            return "mRFP1"
    return x or "amilCP"


def _reporter_pick(
    reporter: str,
) -> Tuple[str, Optional[str]]:
    """Return (semantic_query, preferred_BBa_optional)."""

    k = reporter.strip().casefold().replace(" ", "")
    if "amil" in k:
        return "amilCP blue chromoprotein CDS reporter iGEM", "BBa_K592009"
    if k == "gfp" or "gfp" in k:
        return "GFP CDS coding BioBrick sfGFP iGEM", "BBa_E0040"
    if "mrfp" in k or "rfp" in k:
        return "mRFP1 CDS monomeric red fluorescent BioBrick", "BBa_E1010"
    return f"{reporter} CDS protein coding sequence iGEM BioBrick", None


def _rank_promoter_hits(gate: str, analytes: List[str], hits) -> List:
    """Prefer descriptions matching gate; penalize contradictory labels."""

    analyte_lc = [a.casefold() for a in analytes]
    mx = slot_template_promoter_max_bp()

    scored: List[Tuple[float, object]] = []
    for h in hits:
        seq = str(getattr(h, "sequence", "") or "")
        if len(seq) > mx:
            continue
        desc = str(getattr(h, "short_desc", "") or "").casefold()
        sim = float(getattr(h, "similarity", 0.0) or 0.0)
        adj = sim
        if gate == "AND":
            for bad in _PROM_GATE_NOISE_TERMS_AND:
                if bad in desc:
                    adj -= 0.12
            if "and gate" in desc or ("and" in desc and "dual" in desc):
                adj += 0.06
            for a in analyte_lc:
                if len(a) > 4 and a in desc:
                    adj += 0.04
        elif gate == "OR":
            for bad in _PROM_GATE_BAD_IF_OR:
                if bad in desc:
                    adj -= 0.08
            if "or gate" in desc or (" or " in desc and "xor" not in desc):
                adj += 0.06
            for a in analyte_lc:
                if len(a) > 4 and a in desc:
                    adj += 0.04
        elif gate == "BUF":
            for a in analyte_lc:
                if len(a) > 4 and a in desc:
                    adj += 0.05
        scored.append((adj, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored]


def _retrieve_best_promoter(
    gate: str,
    analytes: List[str],
    progress_cb: Optional[Callable[[str], None]],
):
    from igem_rag import ensure_indexed, retrieve_parts

    ensure_indexed(progress_cb=progress_cb)
    ajoin = ", ".join(analytes) if analytes else "regulatable bacterial"
    if gate == "AND":
        q = f"{ajoin} transcriptional genetic AND gate promoter bacterial iGEM"
    elif gate == "OR":
        q = f"{ajoin} transcriptional genetic OR gate promoter bacterial iGEM"
    else:
        q = f"{ajoin} inducible promoter sensor Escherichia coli BioBrick"

    hits = retrieve_parts(q, part_type_filter="Promoter", top_k=28)
    if not hits and gate == "AND":
        hits = retrieve_parts(
            "dual input transcriptional AND promoter bacterial iGEM",
            part_type_filter="Promoter",
            top_k=20,
        )
    if not hits and gate == "OR":
        hits = retrieve_parts(
            "genetic OR gate promoter dual signal iGEM",
            part_type_filter="Promoter",
            top_k=20,
        )

    ranked = _rank_promoter_hits(gate, analytes, hits)
    if not ranked:
        return None
    min_sim = slot_template_min_promoter_similarity()
    for h in ranked:
        if float(getattr(h, "similarity", 0) or 0) >= min_sim:
            return h
    best = ranked[0]
    if float(getattr(best, "similarity", 0) or 0) >= max(0.42, min_sim - 0.07):
        return best
    return None


def _retrieve_standard_rbs_terminator(
    progress_cb: Optional[Callable[[str], None]],
) -> Tuple[Optional["RetrievedPart"], Optional["RetrievedPart"]]:
    from igem_rag import ensure_indexed, retrieve_parts

    ensure_indexed(progress_cb=progress_cb)
    rb = retrieve_parts("B0034 RBS BioBrick strong", part_type_filter="RBS", top_k=8)
    tr = retrieve_parts("B0015 double terminator BioBrick", part_type_filter="Terminator", top_k=8)
    return (rb[0] if rb else None), (tr[0] if tr else None)


def _retrieve_cds(query: str, bba_fallback: Optional[str], progress_cb):
    from igem_rag import ensure_indexed, retrieve_parts

    ensure_indexed(progress_cb=progress_cb)
    if bba_fallback:
        exact = retrieve_parts(bba_fallback, part_type_filter="CDS", top_k=2)
        if exact and getattr(exact[0], "similarity", 0) >= 0.999:
            return exact[0]
    hits = retrieve_parts(query, part_type_filter="CDS", top_k=12)
    return hits[0] if hits else None


def _hit_to_menu_row(hit) -> dict:
    seq = str(getattr(hit, "sequence", "") or "")
    return {
        "part_name": str(getattr(hit, "part_name", "") or ""),
        "part_type": str(getattr(hit, "part_type", "") or ""),
        "short_desc": str(getattr(hit, "short_desc", "") or "")[:380],
        "sequence": seq,
        "similarity": float(getattr(hit, "similarity", 0.0) or 0.0),
        "match_kind": getattr(hit, "match_kind", "semantic"),
        "retrieval_query": "slot_template",
    }


def try_slot_template_assembly(
    user_prompt: str,
    intent: dict,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Tuple[str, List[dict], List[str], str, Dict[str, dict]]]:
    """Return (dna, assembly_trace_like, ordered_names, prose_thought, by_name_subset) or None."""

    def _prog(msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    if not slot_template_enabled():
        return None

    gate = normalize_gate(intent.get("gate"))
    if gate == "UNKNOWN":
        gate = infer_gate_from_summary(intent)

    lowp = user_prompt.casefold()
    if gate == "UNKNOWN" and "both" in lowp and re.search(r"\band\b|\bonly\s+when\b", lowp):
        gate = "AND"

    reporter = normalize_reporter(intent.get("reporter"), user_prompt)
    analytes = extract_analytes(intent)

    if gate in ("AND", "OR") and len(analytes) < 2:
        # Try user prompt heuristic: "X and Y"
        blob = user_prompt.casefold()
        if ("pyocyanin" in blob or " pyocyanin" in blob) and (
            "lactic" in blob or "lactate" in blob
        ):
            analytes = ["pyocyanin Pseudomonas", "lactic acid lactate metabolism"]
        elif "both" in blob and "and" in blob:
            chunk = blob
            m = re.search(
                r"both\s+([^.;\n]{5,120})\s+and\s+([^.;\n]{5,120})",
                chunk,
                re.I,
            )
            if m:
                analytes = [m.group(1).strip(), m.group(2).strip()]

    if gate in ("AND", "OR") and len(analytes) < 2:
        _prog(f"slot_template · skip ({gate}) — need ≥2 analytes")
        return None

    if gate == "UNKNOWN" or gate == "NOT":
        if gate == "NOT":
            _prog("slot_template · skip — NOT topology not templated yet")
        else:
            _prog("slot_template · skip — unknown gate")
        return None

    _prog(
        f"slot_template · {gate} · analytes={[a[:42] + '…' if len(a) > 42 else a for a in analytes]} "
        f"· reporter={reporter}"
    )

    prom = _retrieve_best_promoter(gate, analytes if gate != "BUF" else analytes[:1], progress_cb)
    if prom is None:
        _prog("slot_template · skip — no promoter hit")
        return None

    rbs, term = _retrieve_standard_rbs_terminator(progress_cb)
    cq, fbba = _reporter_pick(reporter)
    cds = _retrieve_cds(cq, fbba, progress_cb)

    slots = [prom, rbs, cds, term]
    if any(x is None for x in slots):
        _prog(f"slot_template · skip — missing slot (have {[bool(s) for s in slots]})")
        return None

    ordered_names: List[str] = []
    by_name: Dict[str, dict] = {}
    trace: List[dict] = []

    seq_run = []
    pos = 1
    for hit in slots:
        row = _hit_to_menu_row(hit)
        pname = row["part_name"]
        ordered_names.append(pname)
        by_name[pname] = row
        dna = "".join(row["sequence"].upper().split())
        L = len(dna)
        sub = _part_type_for_map(row.get("part_type") or "")
        trace.append(
            {
                "part_name": pname,
                "normalized_name": pname,
                "part_type": row.get("part_type"),
                "label": pname,
                "sub": sub,
                "ok": True,
                "bp": L,
                "source": "slot_template",
                "similarity": row.get("similarity"),
                "retrieval_query": row.get("retrieval_query") or "slot_template",
                "start_bp": pos,
                "end_bp": pos + L - 1,
                "verified": True,
                "sequence_source": "registry",
            }
        )
        seq_run.append(dna)
        pos += L

    dna_final = "".join(seq_run)

    pname = getattr(prom, "part_name", "?")
    pdesc = (getattr(prom, "short_desc", "") or "")[:140]
    thought = (
        f"Slot-template **{gate}** cassette (~{len(dna_final)} bp): promoter **{pname}** ({pdesc}) drives "
        f"standard **{getattr(rbs, 'part_name', 'RBS')}**, **{reporter}** (**{getattr(cds, 'part_name', '')}**), "
        f"terminator **{getattr(term, 'part_name', '')}**. Analytes in the brief: {_snippet_analytes(analytes)}. "
        f"Sequences are stitched in registry order — no proportional LLM-slot substitution."
    )

    return dna_final, trace, ordered_names, thought, by_name


def _snippet_analytes(analytes: List[str]) -> str:
    if not analytes:
        return "(none distilled)"
    return "; ".join(a[:72] + ("…" if len(a) > 72 else "") for a in analytes[:3])


def candidate_from_slot_template(
    user_prompt: str,
    intent: dict,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional["Candidate"]:
    """Build :class:`~inference.Candidate` when assembly succeeds."""

    from inference import Candidate

    pack = try_slot_template_assembly(
        user_prompt, intent, progress_cb=progress_cb
    )
    if pack is None:
        return None
    dna, trace, ordered, thought, _by = pack
    if _slot_template_embed_backbone_enabled():
        try:
            dna, trace = embed_slot_template_in_ecoli_backbone(dna, trace)
            thought += (
                " · **Vector scaffold:** cassette inserted into the DGene E. coli RFC10 backbone "
                "(ColE1-class ori, chloramphenicol resistance, BioBrick MCS) matching `circuit_synth`."
            )
        except Exception as exc:
            sys.stderr.write(
                f"[slot_template] backbone embed failed ({type(exc).__name__}: {exc}) — cassette only.\n"
            )
    min_sim = 0.6
    try:
        raw = (os.environ.get("DGENE_RAG_MIN_SIM") or "0.6").strip()
        min_sim = max(0.3, min(0.95, float(raw)))
    except ValueError:
        pass
    detail = {
        "enabled": True,
        "applied": True,
        "min_similarity": min_sim,
        "pipeline": "slot_template",
        "intent": intent,
        "ordered_part_names": ordered,
        "assembly_trace": trace,
        "map_slots": trace,
        "parts": [
            {
                "part_name": t.get("part_name"),
                "part_type": t.get("part_type"),
                "verified": True,
                "sequence_source": "registry",
                "similarity": t.get("similarity")
                if t.get("similarity") is not None
                else 1.0,
                "query": t.get("retrieval_query") or "slot_template",
            }
            for t in trace
        ],
        "slot_template_gate": normalize_gate(intent.get("gate")),
    }
    return Candidate(
        candidate_id="cand_slot_template",
        thought=thought,
        sequence=dna,
        strategy="slot_template deterministic",
        strategy_name="Slot template (promoter+RBS+CDS+terminator)",
        raw=f"{thought}\n\nORDERED: {' → '.join(ordered)}\n",
        rag_first_detail=detail,
    )
