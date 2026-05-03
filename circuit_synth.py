"""Deterministic topology synthesis from :class:`circuit_ir.CircuitSpec` to :class:`circuit_ir.Plasmid`.

Raises :class:`circuit_parts.PartLookupError` or :class:`TopologyError` if the request cannot be
wired with the curated catalog (caller should fall back to RAG-first).
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from circuit_ir import Cassette, CircuitSpec, LogicSpec, Output, PartRef, Plasmid
from circuit_parts import (
    BACKBONE,
    PROMOTERS,
    RBSES,
    REPORTERS,
    TERMINATORS,
    TFS,
    PartLookupError,
    SensorWiring,
    make_backbone_ref,
    make_promoter_partref,
    make_rbs_partref,
    make_reporter_partref,
    make_tf_partref,
    make_terminator_partref,
    sensor_for_input,
)


class TopologyError(ValueError):
    """The catalog cannot implement this boolean / input combination."""


def _rbs() -> PartRef:
    return make_rbs_partref(RBSES["B0034"])


def _term() -> PartRef:
    return make_terminator_partref(TERMINATORS["B0015"])


def _constitutive_tf_cassette(role: str, tf_name: str) -> Cassette:
    tf_meta = TFS.get(tf_name)
    if not tf_meta:
        raise TopologyError(f"unknown TF {tf_name!r}")
    prom = make_promoter_partref(PROMOTERS["J23100"])
    return Cassette(
        role=role,
        promoter=prom,
        rbs=_rbs(),
        cds=make_tf_partref(tf_meta),
        terminator=_term(),
        drives=tf_meta.name,
        constitutive=True,
    )


def _sensor_output_cassette(
    role: str,
    wiring: SensorWiring,
    cds: PartRef,
    *,
    drives: str,
) -> Cassette:
    prom = make_promoter_partref(wiring.promoter)
    c = Cassette(
        role=role,
        promoter=prom,
        rbs=_rbs(),
        cds=cds,
        terminator=_term(),
        drives=drives,
        induced_by=wiring.input_name,
        activated_by=list(wiring.promoter.activated_by or []),
        repressed_by=list(wiring.promoter.repressed_by or []),
    )
    return c


def _reporter_cassette(
    role: str,
    promoter_name: str,
    reporter_key: str,
    *,
    strip_inducer: bool = False,
) -> Cassette:
    if promoter_name not in PROMOTERS:
        raise TopologyError(f"unknown promoter {promoter_name!r}")
    rep_key = None
    for k in REPORTERS:
        if k.lower() == reporter_key.strip().lower():
            rep_key = k
            break
    if not rep_key:
        raise TopologyError(
            f"reporter {reporter_key!r} not in curated set {list(REPORTERS)}"
        )
    prom_meta = PROMOTERS[promoter_name]
    prom = make_promoter_partref(prom_meta)
    act = list(prom_meta.activated_by or [])
    rep = list(prom_meta.repressed_by or [])
    ind = None if strip_inducer else prom_meta.induced_by
    return Cassette(
        role=role,
        promoter=prom,
        rbs=_rbs(),
        cds=make_reporter_partref(REPORTERS[rep_key]),
        terminator=_term(),
        drives=REPORTERS[rep_key].name,
        activated_by=act,
        repressed_by=rep,
        induced_by=ind,
    )


def _resolve_reporter_key(output: Output) -> str:
    name = (output.name or "").strip()
    for k in REPORTERS:
        if k.lower() == name.lower():
            return k
    for k, meta in REPORTERS.items():
        if meta.phenotype.lower() == name.lower():
            return k
    return name


def _supply_tfs_for_sensors(w0: SensorWiring, w1: SensorWiring) -> Set[str]:
    s0 = {x for x in (w0.promoter.activated_by or [])} | {x for x in (w0.promoter.repressed_by or [])}
    s1 = {x for x in (w1.promoter.activated_by or [])} | {x for x in (w1.promoter.repressed_by or [])}
    return s0 | s1


def _two_distinct_wirings(spec: CircuitSpec) -> Tuple[SensorWiring, SensorWiring]:
    if len(spec.inputs) != 2:
        raise TopologyError("internal: expected 2 inputs")
    w0 = sensor_for_input(spec.inputs[0].name)
    w1 = sensor_for_input(spec.inputs[1].name)
    if not w0 or not w1:
        raise TopologyError(
            "both inputs need catalog sensors (e.g. ahl, iptg, atc, arabinose, lactate, pyocyanin)"
        )
    if w0.input_name == w1.input_name:
        raise TopologyError(
            "2-input gate needs two different inducers in the catalog (e.g. lactate + pyocyanin)"
        )
    return w0, w1


def synthesize(spec: CircuitSpec) -> Plasmid:
    """Build a plasmid (cassettes + backbone ref) for ``spec``. Raises if impossible."""

    rep_key = _resolve_reporter_key(spec.output)
    if rep_key not in REPORTERS:
        raise TopologyError(
            f"choose a curated reporter: {', '.join(sorted(REPORTERS))}"
        )

    logic = spec.logic
    n_in = len(spec.inputs)

    cassettes: List[Cassette] = []
    notes: List[str] = [
        "Topology is deterministic from the certified parts catalog; truth table is checked by "
        "circuit_verify before the candidate is returned.",
        f"Export is a single linear sequence: registry ori + CmR + RFC10 MCS, then cassettes, "
        f"then MCS suffix (see {BACKBONE.name} note).",
    ]

    try:

        if logic.op == "BUF":
            if n_in != 1:
                raise TopologyError("BUF needs exactly 1 input")
            w = sensor_for_input(spec.inputs[0].name)
            if not w:
                raise TopologyError(f"no sensor for input {spec.inputs[0].name!r}")
            need = {x for x in w.promoter.activated_by or []} | {x for x in w.promoter.repressed_by or []}
            for tf in sorted(need):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            cassettes.append(
                _reporter_cassette("output", w.promoter.name, rep_key)
            )


        elif logic.op == "NOT":
            if n_in != 1:
                raise TopologyError("NOT needs exactly 1 input")
            w = sensor_for_input(spec.inputs[0].name)
            if not w:
                raise TopologyError(f"no sensor for input {spec.inputs[0].name!r}")
            need = {x for x in w.promoter.activated_by or []} | {x for x in w.promoter.repressed_by or []}
            for tf in sorted(need):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            cI = TFS["cI"]
            cassettes.append(
                _sensor_output_cassette(
                    "inverter_tf",
                    w,
                    make_tf_partref(cI),
                    drives=cI.name,
                )
            )
            cassettes.append(_reporter_cassette("output", "pCI", rep_key))


        elif logic.op == "AND":
            if n_in != 2:
                raise TopologyError("AND is implemented for 2 inputs in this catalog")
            w0, w1 = _two_distinct_wirings(spec)
            for tf in sorted(_supply_tfs_for_sensors(w0, w1)):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            hrpr = TFS["HrpR"]
            hrps = TFS["HrpS"]
            cassettes.append(
                _sensor_output_cassette("arm_a", w0, make_tf_partref(hrpr), drives=hrpr.name)
            )
            cassettes.append(
                _sensor_output_cassette("arm_b", w1, make_tf_partref(hrps), drives=hrps.name)
            )
            cassettes.append(_reporter_cassette("output", "pHrpL", rep_key))
            notes.append(
                "AND uses Wang–Ellis-style PhrpL co-activation (HrpR + HrpS); both inducers required "
                "for reporter output."
            )


        elif logic.op == "OR":
            if n_in != 2:
                raise TopologyError("OR is implemented for 2 inputs in this catalog")
            w0, w1 = _two_distinct_wirings(spec)
            need0 = {x for x in w0.promoter.activated_by or []} | {x for x in w0.promoter.repressed_by or []}
            need1 = {x for x in w1.promoter.activated_by or []} | {x for x in w1.promoter.repressed_by or []}
            for tf in sorted(need0 | need1):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            rep_part = make_reporter_partref(REPORTERS[rep_key])
            cassettes.append(
                _sensor_output_cassette("or_arm_a", w0, rep_part, drives=rep_key)
            )
            rep_part_b = make_reporter_partref(REPORTERS[rep_key])
            cassettes.append(
                _sensor_output_cassette("or_arm_b", w1, rep_part_b, drives=rep_key)
            )
            notes.append(
                "OR is two parallel inducible transcription units with the same reporter CDS (either "
                "arm can express)."
            )


        elif logic.op == "NAND":
            if n_in != 2:
                raise TopologyError("NAND is implemented for 2 inputs in this catalog")
            w0, w1 = _two_distinct_wirings(spec)
            for tf in sorted(_supply_tfs_for_sensors(w0, w1)):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            hrpr = TFS["HrpR"]
            hrps = TFS["HrpS"]
            cI = TFS["cI"]
            cassettes.append(
                _sensor_output_cassette("arm_a", w0, make_tf_partref(hrpr), drives=hrpr.name)
            )
            cassettes.append(
                _sensor_output_cassette("arm_b", w1, make_tf_partref(hrps), drives=hrps.name)
            )
            cassettes.append(
                Cassette(
                    role="and_to_ci",
                    promoter=make_promoter_partref(PROMOTERS["pHrpL"]),
                    rbs=_rbs(),
                    cds=make_tf_partref(cI),
                    terminator=_term(),
                    drives=cI.name,
                    activated_by=["HrpR", "HrpS"],
                )
            )
            cassettes.append(_reporter_cassette("output", "pCI", rep_key))
            notes.append(
                "NAND = NOT(AND): Hrp AND drives cI; λ pR (pCI) drives reporter only when cI absent."
            )


        elif logic.op == "NOR":
            if n_in != 2:
                raise TopologyError("NOR is implemented for 2 inputs in this catalog")
            w0, w1 = _two_distinct_wirings(spec)
            need0 = {x for x in w0.promoter.activated_by or []} | {x for x in w0.promoter.repressed_by or []}
            need1 = {x for x in w1.promoter.activated_by or []} | {x for x in w1.promoter.repressed_by or []}
            for tf in sorted(need0 | need1):
                cassettes.append(_constitutive_tf_cassette(f"supply_{tf}", tf))
            tet = TFS["tetR"]
            cassettes.append(
                _sensor_output_cassette("nor_arm_a", w0, make_tf_partref(tet), drives=tet.name)
            )
            cassettes.append(
                _sensor_output_cassette("nor_arm_b", w1, make_tf_partref(tet), drives=tet.name)
            )
            cassettes.append(
                _reporter_cassette("output", "pTet", rep_key, strip_inducer=True)
            )
            notes.append(
                "NOR: either input expresses TetR; pTet reporter is off when TetR is present (both "
                "inputs off = reporter on)."
            )

        else:
            raise TopologyError(f"unsupported op {logic.op!r}")

    except PartLookupError as exc:
        raise TopologyError(str(exc)) from exc

    # LogicSpec.operands must align with CircuitSpec.inputs for verify
    backbone = make_backbone_ref()
    return Plasmid(
        name=f"circuit_{logic.op.lower()}",
        backbone=backbone,
        cassettes=cassettes,
        inputs=list(spec.inputs),
        output=spec.output,
        logic=LogicSpec(op=logic.op, operands=list(logic.operands)),
        notes=notes,
    )
