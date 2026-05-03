"""Typed intermediate representation for genetic-circuit designs.

The point of this IR is to separate **what the user wants** (a boolean function over named
inducers, expressed in a chosen reporter) from **how DGene wires it on a plasmid** (specific
cassettes drawn from a vetted parts catalog using a known topology). Everything downstream of
``CircuitSpec`` is deterministic and auditable: ``circuit_synth`` picks a topology, ``circuit_parts``
fills in DNA, and ``circuit_verify`` proves the assembled cassette graph implements the requested
truth table for all 2^n input combinations. If verification fails, the candidate is rejected — the
server falls back to RAG-first and labels the output as a draft.

Conventions:
- Input ``name`` is a canonical, snake-case key (e.g. ``ahl``, ``atc``, ``iptg``, ``arabinose``).
- ``LogicSpec.operands`` references those names directly, with ``"NOT(name)"`` shorthand allowed for
  IR-level inversion of a single operand. The synthesizer sees the gate as a single primitive — it
  does not unfold ``NOT(NOT(x))`` etc. (the intent extractor must normalize).
- Coordinates on ``Cassette`` / ``Plasmid`` are 1-based inclusive (matches the rest of the app).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


LOGIC_OPS = ("BUF", "NOT", "AND", "OR", "NAND", "NOR")
INPUT_KINDS = ("small_molecule", "light", "temperature", "rna", "protein", "stress", "metal")


@dataclass
class Input:
    name: str
    kind: str = "small_molecule"
    display: str = ""
    sensor_available: bool = True
    sensor_note: str = ""

    def __post_init__(self) -> None:
        if not self.display:
            self.display = self.name
        if self.kind not in INPUT_KINDS:
            self.kind = "small_molecule"


@dataclass
class Output:
    name: str
    phenotype: str = ""


@dataclass
class LogicSpec:
    op: str
    operands: List[str]

    def __post_init__(self) -> None:
        op = (self.op or "").upper().strip()
        if op not in LOGIC_OPS:
            raise ValueError(f"LogicSpec.op must be one of {LOGIC_OPS}; got {self.op!r}")
        self.op = op
        if not self.operands:
            raise ValueError("LogicSpec.operands cannot be empty")
        if op in ("BUF", "NOT") and len(self.operands) != 1:
            raise ValueError(f"{op} takes exactly 1 operand")
        if op in ("AND", "OR", "NAND", "NOR") and len(self.operands) < 2:
            raise ValueError(f"{op} takes >= 2 operands")


@dataclass
class CircuitSpec:
    """User-level design intent: inputs, output, requested boolean."""

    inputs: List[Input]
    output: Output
    logic: LogicSpec
    chassis: str = "Escherichia coli K-12"
    notes: str = ""

    def truth_table(self) -> List[Tuple[Tuple[int, ...], int]]:
        """Return ``[((bit0, bit1, ...), expected_output), ...]`` for all input combinations.

        Bit ordering matches ``self.inputs`` (index 0 first). The expected output is computed
        purely from ``self.logic`` — no chemistry assumed.
        """

        n = len(self.inputs)
        if n == 0:
            return [((), 1 if self.logic.op == "BUF" else 0)]
        rows: List[Tuple[Tuple[int, ...], int]] = []
        for k in range(2 ** n):
            bits = tuple((k >> (n - 1 - i)) & 1 for i in range(n))
            rows.append((bits, self._eval(bits)))
        return rows

    def _eval(self, bits: Tuple[int, ...]) -> int:
        ix: Dict[str, int] = {inp.name: i for i, inp in enumerate(self.inputs)}
        vals: List[int] = []
        for op_str in self.logic.operands:
            negated = False
            name = op_str
            if name.startswith("NOT(") and name.endswith(")"):
                negated = True
                name = name[4:-1]
            if name not in ix:
                raise ValueError(f"LogicSpec operand {op_str!r} not in inputs {list(ix)}")
            v = bits[ix[name]]
            if negated:
                v = 1 - v
            vals.append(v)
        op = self.logic.op
        if op == "BUF":
            return vals[0]
        if op == "NOT":
            return 1 - vals[0]
        if op == "AND":
            r = 1
            for v in vals:
                r &= v
            return r
        if op == "OR":
            r = 0
            for v in vals:
                r |= v
            return r
        if op == "NAND":
            r = 1
            for v in vals:
                r &= v
            return 1 - r
        if op == "NOR":
            r = 0
            for v in vals:
                r |= v
            return 1 - r
        raise ValueError(f"unhandled LogicSpec.op={op!r}")  # pragma: no cover


@dataclass
class PartRef:
    """A concrete DNA part with its provenance recorded."""

    name: str                              # canonical short name, e.g. "pLac", "B0034", "amilCP"
    bba_id: Optional[str]
    kind: str                              # "Promoter" | "RBS" | "CDS" | "Terminator"
    sequence: str
    length_bp: int = 0
    source: str = "iGEM"                   # "iGEM" | "needs_synthesis" | "external_ref"
    external_ref: Optional[str] = None     # NCBI accession when source != "iGEM"
    note: str = ""

    def __post_init__(self) -> None:
        seq = "".join((self.sequence or "").upper().split())
        self.sequence = seq
        self.length_bp = len(seq)


@dataclass
class BackboneRef:
    name: str
    bba_id: Optional[str]
    sequence: str
    length_bp: int = 0
    ori: str = ""
    selection_marker: str = ""
    note: str = ""
    # RFC10 MCS suffix after the synthesized cassette stack (SpeI–NotI–PstI side).
    suffix_sequence: str = ""
    # Optional 1-based map segments (ori / resistance / MCS) emitted before cassette features.
    prefix_segments: List[Dict[str, object]] = field(default_factory=list)
    suffix_segments: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        seq = "".join((self.sequence or "").upper().split())
        suf = "".join((self.suffix_sequence or "").upper().split())
        self.sequence = seq
        self.suffix_sequence = suf
        self.length_bp = len(seq) + len(suf)


@dataclass
class Cassette:
    """One transcription unit: ``Promoter → RBS → CDS → Terminator``.

    The regulator metadata (``activated_by``, ``repressed_by``, ``induced_by``, ``constitutive``)
    is what the verifier walks to compute the truth table — it is set by the synthesizer when it
    chooses parts, *not* parsed back out of DNA.
    """

    role: str
    promoter: PartRef
    rbs: PartRef
    cds: PartRef
    terminator: PartRef
    drives: Optional[str] = None
    activated_by: List[str] = field(default_factory=list)
    repressed_by: List[str] = field(default_factory=list)
    induced_by: Optional[str] = None
    constitutive: bool = False

    def length_bp(self) -> int:
        return (
            self.promoter.length_bp
            + self.rbs.length_bp
            + self.cds.length_bp
            + self.terminator.length_bp
        )

    def labels(self) -> List[str]:
        return [
            f"P:{self.promoter.name}",
            f"R:{self.rbs.name}",
            f"C:{self.cds.name}",
            f"T:{self.terminator.name}",
        ]


@dataclass
class Plasmid:
    name: str
    backbone: BackboneRef
    cassettes: List[Cassette]
    inputs: List[Input]
    output: Output
    logic: LogicSpec
    assembly_method: str = "Gibson assembly (BioBrick RFC[10] compatible)"
    notes: List[str] = field(default_factory=list)
    truth_table: List[Tuple[Tuple[int, ...], int, int]] = field(default_factory=list)
    truth_table_passes: bool = False
    verification_summary: str = ""

    def total_bp(self) -> int:
        bp = self.backbone.length_bp
        for c in self.cassettes:
            bp += c.length_bp()
        return bp

    def insert_sequence(self) -> str:
        """Transcription units only (no ori / resistance / MCS)."""

        chunks: List[str] = []
        for c in self.cassettes:
            chunks.append(c.promoter.sequence)
            chunks.append(c.rbs.sequence)
            chunks.append(c.cds.sequence)
            chunks.append(c.terminator.sequence)
        return "".join(chunks)

    def assembled_sequence(self) -> str:
        """Linearized plasmid scaffold: ori + CmR + MCS prefix → cassettes → MCS suffix (iGEM RFC10)."""

        chunks: List[str] = []
        chunks.append(self.backbone.sequence)
        for c in self.cassettes:
            chunks.append(c.promoter.sequence)
            chunks.append(c.rbs.sequence)
            chunks.append(c.cds.sequence)
            chunks.append(c.terminator.sequence)
        chunks.append(self.backbone.suffix_sequence)
        return "".join(chunks)

    def map_slots(self) -> List[Dict[str, object]]:
        """Per-feature dict matching the front-end ``map_slots`` schema.

        Each promoter/RBS/CDS/terminator becomes one slot with its 1-based ``start_bp``/``end_bp``.
        ``verified`` is true when the part came from the iGEM registry; ``needs_synthesis`` parts
        carry a clear note in the tooltip via the ``note`` field.
        """

        out: List[Dict[str, object]] = []
        pos = 1

        def push_segment(seg: Dict[str, object]) -> None:
            nonlocal pos
            L = int(seg.get("bp") or 0)
            if L <= 0:
                return
            slot = {
                "label": seg.get("label", ""),
                "sub": seg.get("sub", "feature"),
                "start_bp": pos,
                "end_bp": pos + L - 1,
                "verified": bool(seg.get("verified", True)),
                "sequence_source": seg.get("sequence_source", "registry"),
                "part_name": seg.get("part_name", ""),
                "part_type": seg.get("part_type", "Backbone"),
                "role": seg.get("role", "backbone"),
                "bp": L,
                "ok": True,
                "note": seg.get("note", ""),
            }
            out.append(slot)
            pos += L

        def push(part: PartRef, sub: str, role: str) -> None:
            nonlocal pos
            L = part.length_bp
            if L == 0:
                return
            verified = part.source == "iGEM"
            seq_source = "registry" if verified else (
                "ncbi" if part.source == "external_ref" and part.external_ref else "model"
            )
            label_parts = [part.name]
            if part.bba_id:
                label_parts.append(part.bba_id)
            slot: Dict[str, object] = {
                "label": " · ".join(label_parts),
                "sub": sub,
                "start_bp": pos,
                "end_bp": pos + L - 1,
                "verified": verified,
                "sequence_source": seq_source,
                "part_name": part.bba_id or part.name,
                "part_type": part.kind,
                "role": role,
                "bp": L,
                "ok": True,
                "note": part.note,
            }
            if part.external_ref:
                slot["external_ref"] = part.external_ref
            out.append(slot)
            pos += L

        for seg in self.backbone.prefix_segments:
            push_segment(seg)

        for c in self.cassettes:
            push(c.promoter, "promoter", c.role)
            push(c.rbs, "rbs", c.role)
            push(c.cds, "cds", c.role)
            push(c.terminator, "terminator", c.role)

        for seg in self.backbone.suffix_segments:
            push_segment(seg)

        backbone = self.backbone
        if (
            not backbone.prefix_segments
            and not backbone.suffix_segments
            and backbone.length_bp > 0
        ):
            out.append({
                "label": f"{backbone.name}" + (f" · {backbone.bba_id}" if backbone.bba_id else ""),
                "sub": "backbone",
                "start_bp": pos,
                "end_bp": pos + backbone.length_bp - 1,
                "verified": True,
                "sequence_source": "registry" if backbone.bba_id else "model",
                "part_name": backbone.bba_id or backbone.name,
                "part_type": "Backbone",
                "role": f"backbone::{backbone.ori}/{backbone.selection_marker}",
                "bp": backbone.length_bp,
                "ok": True,
                "note": backbone.note,
            })
        return out
