"""Boolean verification of a synthesized :class:`circuit_ir.Plasmid`.

The verifier does **not** simulate RNAP kinetics or growth — it evaluates a conservative regulatory
graph: per input bit-vector, which CDS products are eventually expressed (fixpoint), whether each
reporter cassette's promoter is active, and whether that matches ``CircuitSpec.truth_table()``.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from circuit_ir import Cassette, CircuitSpec, Input, Plasmid
from circuit_parts import REPORTERS


def _bits_dict(inputs: List[Input], bits: Tuple[int, ...]) -> Dict[str, int]:
    return {inp.name: bits[i] for i, inp in enumerate(inputs)}


def _reporter_names(spec: CircuitSpec) -> Set[str]:
    key = (spec.output.name or "").strip()
    for k in REPORTERS:
        if k.lower() == key.lower():
            return {REPORTERS[k].name}
    return {key}


def _protein_product(c: Cassette) -> str:
    if c.drives:
        return c.drives
    return c.cds.name


def promoter_active(
    c: Cassette,
    bits: Dict[str, int],
    expressed: Set[str],
) -> bool:
    """Return whether transcription from ``c``'s promoter proceeds (simplified ON/OFF)."""

    if c.constitutive:
        return True

    ind = c.induced_by
    rep = c.repressed_by
    act = c.activated_by

    if ind is not None:
        bit = int(bits.get(ind, 0))
        if rep:
            if not all(r in expressed for r in rep):
                return False
            return bool(bit)
        if act:
            return bool(bit) and all(a in expressed for a in act)

    if act and ind is None:
        return all(a in expressed for a in act)

    if rep and ind is None and not act:
        return not any(r in expressed for r in rep)

    return False


def reporter_expressed(spec: CircuitSpec, plasmid: Plasmid, bits: Tuple[int, ...]) -> int:
    """1 if any reporter CDS cassette is transcribed under ``bits``, else 0."""

    bmap = _bits_dict(spec.inputs, bits)
    names = _reporter_names(spec)
    expressed: Set[str] = set()
    max_passes = max(32, len(plasmid.cassettes) * 8)
    for _ in range(max_passes):
        grew = False
        for c in plasmid.cassettes:
            if promoter_active(c, bmap, expressed):
                prod = _protein_product(c)
                if prod not in expressed:
                    expressed.add(prod)
                    grew = True
        if not grew:
            break

    on = 0
    for c in plasmid.cassettes:
        if c.cds.name not in names and (not names or not any(
            n.lower() == c.cds.name.lower() for n in names
        )):
            continue
        if promoter_active(c, bmap, expressed):
            on = 1
            break
    return on


def verify_plasmid(spec: CircuitSpec, plasmid: Plasmid) -> Tuple[bool, List[Tuple[Tuple[int, ...], int, int]], str]:
    """Return ``(passes, truth_table_with_actual, summary)``.

    Each truth-table row is ``(input_bits, expected_output, actual_output)``.
    """

    rows_out: List[Tuple[Tuple[int, ...], int, int]] = []
    all_ok = True
    for bits, expected in spec.truth_table():
        actual = reporter_expressed(spec, plasmid, bits)
        rows_out.append((bits, expected, actual))
        if actual != expected:
            all_ok = False
    if all_ok:
        summary = (
            f"Verified: {len(rows_out)} state(s) match truth table for "
            f"{spec.logic.op} over {[i.name for i in spec.inputs]}"
        )
    else:
        failures = [r for r in rows_out if r[1] != r[2]]
        summary = (
            "Verification failed: "
            + "; ".join(
                f"in{f[0]} expected {f[1]} got {f[2]}" for f in failures[:4]
            )
        )
        if len(failures) > 4:
            summary += f" (+{len(failures) - 4} more)"

    plasmid.truth_table = rows_out
    plasmid.truth_table_passes = all_ok
    plasmid.verification_summary = summary
    return all_ok, rows_out, summary
