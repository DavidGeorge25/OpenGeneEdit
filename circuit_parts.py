"""Curated synthetic-biology parts catalog with regulatory binding metadata.

This module is the **lab-readiness layer** under :mod:`circuit_synth`. Unlike the open-ended iGEM
RAG retrieval, every entry here is hand-picked for:

- **Provenance**: each part has a real ``BBa_*`` ID present in ``igem_dataset.jsonl`` (or is clearly
  marked ``needs_synthesis`` with an NCBI accession the lab can order).
- **Mechanism**: each promoter knows which TF activates / represses it; each TF CDS knows which
  inducer, if any, switches it on; each RBS / terminator carries a usability rating. This is what
  :mod:`circuit_verify` walks to *prove* the assembled topology implements the requested boolean
  for all 2^n input combinations.
- **Cloning context**: exports include a **pSB1C3-style linear scaffold** (ori + chloramphenicol
  resistance + RFC10 MCS) from registry fragments plus the verified cassette stack — compare to
  the physical distribution vector (BBa_J04450) before lab use.

Sequence resolution is lazy: on first access we read ``igem_dataset.jsonl`` once and cache a
``part_name → row`` index. Parts whose ``BBa_*`` ID is missing from the local snapshot raise
:class:`PartLookupError` so the synthesizer can fall back to a different topology rather than emit
a silently-empty cassette.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from circuit_ir import BackboneRef, PartRef


_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSONL = os.path.join(_MODULE_DIR, "data", "igem_dataset.jsonl")

_JSONL_LOCK = threading.Lock()
_JSONL_INDEX: Optional[Dict[str, dict]] = None


class PartLookupError(LookupError):
    """Raised when a curated part's BBa ID is missing from the local JSONL snapshot."""


def _jsonl_path() -> str:
    return os.environ.get("DGENE_IGEM_JSONL", _DEFAULT_JSONL).strip() or _DEFAULT_JSONL


def _load_jsonl_index() -> Dict[str, dict]:
    """Map ``part_name → row`` for the entire iGEM dataset (cached after first call)."""

    global _JSONL_INDEX
    with _JSONL_LOCK:
        if _JSONL_INDEX is not None:
            return _JSONL_INDEX
        path = _jsonl_path()
        index: Dict[str, dict] = {}
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = str(row.get("part_name", "")).strip()
                    if name:
                        index[name] = row
        _JSONL_INDEX = index
        return _JSONL_INDEX


# ---------------------------------------------------------------------------
# Curated catalog metadata
# ---------------------------------------------------------------------------


@dataclass
class PromoterMeta:
    """Curated promoter entry. ``bba_id`` is the *primary* registry source for the DNA.

    ``activated_by`` and ``repressed_by`` are TF *names* (must match keys in :data:`TFS`).
    ``induced_by`` names an :class:`circuit_ir.Input` (must match the prompt's input name once
    canonicalized) and is set when the promoter is the cognate output of a small-molecule sensor —
    e.g. ``pLux`` is induced by AHL via the LuxR receiver.
    """

    name: str
    bba_id: str
    short_desc: str = ""
    activated_by: List[str] = field(default_factory=list)
    repressed_by: List[str] = field(default_factory=list)
    induced_by: Optional[str] = None
    polarity: str = "active-high"


@dataclass
class TFMeta:
    """Curated transcription-factor CDS entry."""

    name: str
    bba_id: str
    short_desc: str = ""
    activates: List[str] = field(default_factory=list)   # promoter names
    represses: List[str] = field(default_factory=list)   # promoter names
    inducer_name: Optional[str] = None                   # e.g. "ahl" → activates promoter
    inducer_polarity: str = "activates"                  # "activates" | "derepresses"


@dataclass
class ReporterMeta:
    name: str
    bba_id: str
    phenotype: str
    short_desc: str = ""


@dataclass
class RBSMeta:
    name: str
    bba_id: str
    relative_strength: float = 1.0       # 1.0 ~ B0034
    short_desc: str = ""


@dataclass
class TerminatorMeta:
    name: str
    bba_id: str
    is_double: bool = True               # B0015 = double terminator
    short_desc: str = ""


@dataclass
class BackboneMeta:
    name: str
    bba_id: Optional[str]
    ori: str
    selection_marker: str
    note: str
    # Registry sources in igem_dataset.jsonl; prefix is ori + resistance + RFC10 MCS (XbaI side).
    ori_bba_id: str = "BBa_K3868024"
    resistance_bba_id: str = "BBa_K150003"
    mcs_prefix: str = "GAATTCGCGGCCGCTTCTAGA"
    mcs_suffix: str = "TACTAGTAGCGGCCGCTGCAG"


# Promoters: only inducibles + the canonical Anderson constitutive (used for "always-on" TF cassettes).
PROMOTERS: Dict[str, PromoterMeta] = {
    "pLac": PromoterMeta(
        name="pLac",
        bba_id="BBa_R0010",
        short_desc="lacI-repressed lac promoter; derepressed by IPTG / lactose",
        repressed_by=["lacI"],
        induced_by="iptg",
        polarity="active-high",
    ),
    "pTet": PromoterMeta(
        name="pTet",
        bba_id="BBa_R0040",
        short_desc="tetR-repressed tetracycline operator promoter; derepressed by aTc",
        repressed_by=["tetR"],
        induced_by="atc",
        polarity="active-high",
    ),
    "pBAD": PromoterMeta(
        name="pBAD",
        bba_id="BBa_R0080",
        short_desc="araC-regulated arabinose promoter; activated by araC + arabinose",
        activated_by=["araC"],
        induced_by="arabinose",
        polarity="active-high",
    ),
    "pLux": PromoterMeta(
        name="pLux",
        bba_id="BBa_R0062",
        short_desc="luxR/HSL activated promoter (lux pR)",
        activated_by=["luxR"],
        induced_by="ahl",
        polarity="active-high",
    ),
    "pCI": PromoterMeta(
        name="pCI",
        bba_id="BBa_R0051",
        short_desc="lambda cI-repressed promoter (pR)",
        repressed_by=["cI"],
        polarity="active-high",
    ),
    "pHrpL": PromoterMeta(
        name="pHrpL",
        bba_id="BBa_K3994005",
        short_desc="sigma54 PhrpL output promoter; activated only when HrpR + HrpS are both present",
        activated_by=["HrpR", "HrpS"],
        polarity="active-high",
    ),
    "pLld": PromoterMeta(
        name="pLld",
        bba_id="BBa_K822000",
        short_desc="lldPRD promoter + RBS (E. coli); ON when LldR + L-lactate",
        activated_by=["lldR"],
        induced_by="lactate",
        polarity="active-high",
    ),
    "pPqs": PromoterMeta(
        name="pPqs",
        bba_id="BBa_K1157000",
        short_desc="pqsA promoter (PQS regulon); proxy for Pseudomonas/pyocyanin-pathway input with PqsR",
        activated_by=["PqsR"],
        induced_by="pyocyanin",
        polarity="active-high",
    ),
    "J23100": PromoterMeta(
        name="J23100",
        bba_id="BBa_K4233030",
        short_desc="Anderson J23100 strong constitutive promoter",
        polarity="active-high",
    ),
}


# Transcription factors and small-molecule sensors. ``inducer_name`` keys must align with the
# canonical ``Input.name`` keys produced by ``circuit_intent`` (lowercase, snake_case).
TFS: Dict[str, TFMeta] = {
    "lacI": TFMeta(
        name="lacI",
        bba_id="BBa_C0012",
        short_desc="lacI repressor (LVA-tagged); blocks pLac unless IPTG is present",
        represses=["pLac"],
        inducer_name="iptg",
        inducer_polarity="derepresses",
    ),
    "tetR": TFMeta(
        name="tetR",
        bba_id="BBa_C0040",
        short_desc="tetR repressor (LVA); blocks pTet unless aTc is present",
        represses=["pTet"],
        inducer_name="atc",
        inducer_polarity="derepresses",
    ),
    "cI": TFMeta(
        name="cI",
        bba_id="BBa_C0051",
        short_desc="lambda cI repressor (LVA); blocks pCI",
        represses=["pCI"],
    ),
    "luxR": TFMeta(
        name="luxR",
        bba_id="BBa_C0062",
        short_desc="luxR receiver; activates pLux when bound to AHL",
        activates=["pLux"],
        inducer_name="ahl",
        inducer_polarity="activates",
    ),
    "araC": TFMeta(
        name="araC",
        bba_id="BBa_C0080",
        short_desc="araC; activates pBAD in the presence of arabinose, represses without",
        activates=["pBAD"],
        inducer_name="arabinose",
        inducer_polarity="activates",
    ),
    "HrpR": TFMeta(
        name="HrpR",
        bba_id="BBa_K1014001",
        short_desc="HrpR co-activator (sigma54 PhrpL); requires HrpS to activate output",
        activates=["pHrpL"],
    ),
    "HrpS": TFMeta(
        name="HrpS",
        bba_id="BBa_K1014000",
        short_desc="HrpS co-activator (sigma54 PhrpL); requires HrpR to activate output",
        activates=["pHrpL"],
    ),
    "lldR": TFMeta(
        name="lldR",
        bba_id="BBa_K1847001",
        short_desc="LldR (E. coli); required for lactate induction of pLld",
        activates=["pLld"],
        inducer_name="lactate",
        inducer_polarity="activates",
    ),
    "PqsR": TFMeta(
        name="PqsR",
        bba_id="BBa_K1157001",
        short_desc="PqsR (Pseudomonas); used with pqs promoter as registry proxy for pyocyanin/PQS input",
        activates=["pPqs"],
        inducer_name="pyocyanin",
        inducer_polarity="activates",
    ),
}


REPORTERS: Dict[str, ReporterMeta] = {
    "amilCP": ReporterMeta(
        name="amilCP",
        bba_id="BBa_K592009",
        phenotype="blue chromoprotein (visible without UV)",
        short_desc="amilCP from Acropora millepora; deep-blue chromoprotein",
    ),
    "GFP": ReporterMeta(
        name="GFP",
        bba_id="BBa_E0040",
        phenotype="green fluorescence",
        short_desc="GFP (Aequorea victoria wild-type)",
    ),
    "mRFP1": ReporterMeta(
        name="mRFP1",
        bba_id="BBa_E1010",
        phenotype="red fluorescence",
        short_desc="monomeric red fluorescent protein 1",
    ),
}


RBSES: Dict[str, RBSMeta] = {
    "B0034": RBSMeta(
        name="B0034",
        bba_id="BBa_K812053",
        relative_strength=1.0,
        short_desc="Elowitz RBS B0034 (Goldenbrick scar variant; canonical strong RBS)",
    ),
}


TERMINATORS: Dict[str, TerminatorMeta] = {
    "B0015": TerminatorMeta(
        name="B0015",
        bba_id="BBa_B0015",
        is_double=True,
        short_desc="B0015 strong double terminator (B0010-B0012)",
    ),
}


BACKBONE = BackboneMeta(
    name="pSB1C3",
    bba_id="BBa_J04450",
    ori="pMB1 (high copy)",
    selection_marker="chloramphenicol (cmR)",
    note=(
        "Linearized scaffold assembled from registry ColE1-class origin (BBa_K3868024), "
        "chloramphenicol cassette (BBa_K150003), and standard iGEM RFC10 MCS "
        "(5′ EcoRI–NotI–XbaI, 3′ SpeI–NotI–PstI). "
        "Compare insert length and resistance marker to the physical pSB1C3 distribution "
        "vector (BBa_J04450) before ordering. Use chloramphenicol (25 µg/mL) for selection."
    ),
)


# ---------------------------------------------------------------------------
# Sequence resolution helpers
# ---------------------------------------------------------------------------


def _resolve_sequence(bba_id: str) -> str:
    index = _load_jsonl_index()
    row = index.get(bba_id)
    if not row:
        raise PartLookupError(f"BBa ID {bba_id!r} not in {_jsonl_path()}")
    seq = str(row.get("sequence", "")).strip()
    if not seq:
        raise PartLookupError(f"BBa ID {bba_id!r} has empty sequence in JSONL")
    return seq


def _resolve_short_desc(bba_id: str, fallback: str) -> str:
    index = _load_jsonl_index()
    row = index.get(bba_id)
    if not row:
        return fallback
    return str(row.get("short_desc", "") or fallback).strip() or fallback


def make_promoter_partref(meta: PromoterMeta) -> PartRef:
    seq = _resolve_sequence(meta.bba_id)
    return PartRef(
        name=meta.name,
        bba_id=meta.bba_id,
        kind="Promoter",
        sequence=seq,
        source="iGEM",
        note=_resolve_short_desc(meta.bba_id, meta.short_desc),
    )


def make_tf_partref(meta: TFMeta) -> PartRef:
    seq = _resolve_sequence(meta.bba_id)
    return PartRef(
        name=meta.name,
        bba_id=meta.bba_id,
        kind="CDS",
        sequence=seq,
        source="iGEM",
        note=_resolve_short_desc(meta.bba_id, meta.short_desc),
    )


def make_reporter_partref(meta: ReporterMeta) -> PartRef:
    seq = _resolve_sequence(meta.bba_id)
    return PartRef(
        name=meta.name,
        bba_id=meta.bba_id,
        kind="CDS",
        sequence=seq,
        source="iGEM",
        note=f"Reporter · {meta.phenotype} · " + _resolve_short_desc(meta.bba_id, meta.short_desc),
    )


def make_rbs_partref(meta: RBSMeta) -> PartRef:
    seq = _resolve_sequence(meta.bba_id)
    return PartRef(
        name=meta.name,
        bba_id=meta.bba_id,
        kind="RBS",
        sequence=seq,
        source="iGEM",
        note=_resolve_short_desc(meta.bba_id, meta.short_desc),
    )


def make_terminator_partref(meta: TerminatorMeta) -> PartRef:
    seq = _resolve_sequence(meta.bba_id)
    return PartRef(
        name=meta.name,
        bba_id=meta.bba_id,
        kind="Terminator",
        sequence=seq,
        source="iGEM",
        note=_resolve_short_desc(meta.bba_id, meta.short_desc),
    )


def make_backbone_ref(meta: BackboneMeta = BACKBONE) -> BackboneRef:
    """Backbone 5′ (ori + resistance + MCS) and 3′ MCS suffix from the local registry JSONL."""

    ori_seq = _resolve_sequence(meta.ori_bba_id)
    cm_seq = _resolve_sequence(meta.resistance_bba_id)
    pre_mcs = "".join((meta.mcs_prefix or "").upper().split())
    suf_mcs = "".join((meta.mcs_suffix or "").upper().split())
    prefix = "".join([ori_seq, cm_seq, pre_mcs])

    prefix_segments: List[Dict[str, object]] = [
        {
            "label": f"pMB1-derived ori · {meta.ori_bba_id}",
            "sub": "feature",
            "bp": len(ori_seq),
            "part_name": meta.ori_bba_id,
            "part_type": "Origin",
            "role": f"backbone::ori::{meta.ori}",
            "verified": True,
            "sequence_source": "registry",
            "note": "ColE1-class replication origin fragment (registry snapshot).",
        },
        {
            "label": f"chloramphenicol resistance · {meta.resistance_bba_id}",
            "sub": "feature",
            "bp": len(cm_seq),
            "part_name": meta.resistance_bba_id,
            "part_type": "Backbone",
            "role": f"backbone::selection::{meta.selection_marker}",
            "verified": True,
            "sequence_source": "registry",
            "note": meta.selection_marker,
        },
        {
            "label": "RFC10 MCS (5′)",
            "sub": "feature",
            "bp": len(pre_mcs),
            "part_name": "MCS_5prime",
            "part_type": "Backbone",
            "role": "backbone::MCS:EcoRI-NotI-XbaI",
            "verified": True,
            "sequence_source": "model",
            "note": "Standard BioBrick prefix / MCS 5′ (EcoRI, NotI, XbaI).",
        },
    ]
    suffix_segments: List[Dict[str, object]] = [
        {
            "label": "RFC10 MCS (3′)",
            "sub": "feature",
            "bp": len(suf_mcs),
            "part_name": "MCS_3prime",
            "part_type": "Backbone",
            "role": "backbone::MCS:SpeI-NotI-PstI",
            "verified": True,
            "sequence_source": "model",
            "note": "Standard BioBrick suffix / MCS 3′ (SpeI, NotI, PstI).",
        },
    ]

    return BackboneRef(
        name=meta.name,
        bba_id=meta.bba_id,
        sequence=prefix,
        ori=meta.ori,
        selection_marker=meta.selection_marker,
        note=meta.note,
        suffix_sequence=suf_mcs,
        prefix_segments=prefix_segments,
        suffix_segments=suffix_segments,
    )


# ---------------------------------------------------------------------------
# Inducer → sensor wiring
# ---------------------------------------------------------------------------


# Canonical inducer aliases the intent extractor / synthesizer uses to map natural-language
# names to one of the supported sensors. Keys are lowercase. Values are the canonical Input.name
# the catalog understands.
INDUCER_ALIASES: Dict[str, str] = {
    # IPTG / lactose
    "iptg": "iptg",
    "lactose": "iptg",
    # aTc / tetracycline
    "atc": "atc",
    "anhydrotetracycline": "atc",
    "tetracycline": "atc",
    "doxycycline": "atc",
    # arabinose
    "arabinose": "arabinose",
    "l-arabinose": "arabinose",
    "ara": "arabinose",
    # AHL / N-acyl-homoserine lactones / quorum sensing
    "ahl": "ahl",
    "3-oxo-c6-hsl": "ahl",
    "3oc6hsl": "ahl",
    "n-3-oxohexanoyl-homoserine lactone": "ahl",
    "homoserine lactone": "ahl",
    "lux": "ahl",
    # Lactate (high lactate → same Boolean bit as catalog input \"lactate\")
    "l-lactate": "lactate",
    "l_lactate": "lactate",
    "high lactate": "lactate",
    # Pyocyanin / Pseudomonas small-molecule proxy (PQS-pathway parts in catalog)
    "pyocyanin": "pyocyanin",
    "pqs": "pyocyanin",
    "pqs signal": "pyocyanin",
    "phenazine": "pyocyanin",
}


@dataclass
class SensorWiring:
    """How a single small-molecule input is sensed into transcription.

    ``tf`` is the receiver TF and ``promoter`` is its cognate promoter; the synthesizer wires:
    a constitutive expression cassette for ``tf`` + an output cassette whose promoter is
    ``promoter``. The resulting ``Cassette.induced_by`` is set to ``input_name`` so the verifier
    treats the promoter as ON iff the input bit is 1 (for activator inducers) or iff the input bit
    is 1 and the repressor's TF is present (for derepression).
    """

    input_name: str
    tf: TFMeta
    promoter: PromoterMeta
    polarity: str    # "activates" | "derepresses"


def canonical_inducer_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    return INDUCER_ALIASES.get(raw.strip().lower())


def sensor_for_input(input_name: str) -> Optional[SensorWiring]:
    """Find the canonical sensor wiring for a given canonical input name.

    Returns ``None`` if no sensor in the curated catalog matches — the synthesizer will then
    decline the topology and the pipeline falls back to RAG-first.
    """

    name = canonical_inducer_name(input_name) or input_name.strip().lower()
    for tf in TFS.values():
        if tf.inducer_name and tf.inducer_name.lower() == name:
            target_promoter_name: Optional[str] = None
            if tf.inducer_polarity == "activates":
                target_promoter_name = tf.activates[0] if tf.activates else None
            else:  # derepresses
                target_promoter_name = tf.represses[0] if tf.represses else None
            if not target_promoter_name:
                continue
            promoter_meta = PROMOTERS.get(target_promoter_name)
            if not promoter_meta:
                continue
            return SensorWiring(
                input_name=name,
                tf=tf,
                promoter=promoter_meta,
                polarity=tf.inducer_polarity,
            )
    return None


def supported_inducers() -> List[str]:
    """Return the sorted list of canonical inducer names the catalog can sense."""

    out = sorted({tf.inducer_name for tf in TFS.values() if tf.inducer_name})
    return [s for s in out if s]


def supported_reporters() -> List[str]:
    return sorted(REPORTERS.keys())
