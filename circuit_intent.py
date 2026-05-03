"""Gemma-based extraction of :class:`circuit_ir.CircuitSpec` from a natural-language brief.

Only boolean designs over catalog-supported small-molecule inputs are marked ``applicable``; all
other prompts should use RAG-first instead (the pipeline treats ``applicable: false`` as skip).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from circuit_ir import CircuitSpec, Input, LogicSpec, Output
from circuit_parts import canonical_inducer_name, supported_inducers, supported_reporters

_INTENT_SYSTEM = """You are a synthetic biology design parser. Read the user brief and output \
ONLY one JSON object (no markdown fences, no commentary).

Schema:
{
  "applicable": true or false,
  "reason": "short string if applicable is false",
  "inputs": [ { "name": "canonical_key", "display": "human label", "kind": "small_molecule" } ],
  "output": { "name": "reporter_key", "phenotype": "short" },
  "logic": { "op": "BUF|NOT|AND|OR|NAND|NOR", "operands": ["name", ...] },
  "chassis": "organism or Escherichia coli",
  "notes": "optional lab constraints"
}

Rules:
- Set applicable=true ONLY if the brief asks for a **discrete logic gate** or clearly boolean \
combination (BUFFER, NOT, AND, OR, NAND, NOR) of **chemical inducers** or quorum signals you can map.
- supported_input canonical keys (use ONLY these names in inputs[] and operands[]): \
""" + ", ".join(sorted(supported_inducers())) + """
- supported reporter names (output.name): """ + ", ".join(sorted(supported_reporters())) + """
- For NOT use op NOT with one operand. For AND/OR/NAND/NOR use two distinct operands when two inputs \
are implied.
- If the brief mentions analytes you cannot map to the supported_input list, set applicable=false \
and explain in reason. Supported mappings now include **pyocyanin** (PQS-pathway proxy) and \
**lactate** (LldR / pLld) in addition to IPTG, aTc, arabinose, and AHL.
- Operands must exactly match the `name` field of entries in `inputs` (same spelling).

End with the closing brace } only.
"""

_REPAIR = (
    "**Parse repair:** Reply with ONLY valid JSON matching the schema. If the brief is not a "
    "boolean gate over supported inputs, set applicable false."
)


def _strip_json_fences(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        t = "\n".join(lines).strip()
    return t


def _parse_intent_json(raw: str) -> dict:
    text = _strip_json_fences(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Model often emits markdown / prose before the JSON object; anchor on "applicable".
    m = re.search(r"\{\s*\"applicable\"\s*:", text)
    if m:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(text[m.start() :])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    i = text.find("{")
    if i < 0:
        raise ValueError(f"intent JSON parse failed; head: {text[:400]!r}") from None
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[i:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    j = text.rfind("}")
    if j > i:
        return json.loads(text[i : j + 1])
    raise ValueError(f"intent JSON parse failed; head: {text[:400]!r}") from None


def _coerce_spec(data: dict) -> Optional[CircuitSpec]:
    if not data.get("applicable"):
        return None

    in_rows = data.get("inputs") or []
    if not isinstance(in_rows, list) or not in_rows:
        return None

    inputs: List[Input] = []
    for row in in_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip().lower()
        name = canonical_inducer_name(name) or name
        if name not in supported_inducers():
            return None
        inputs.append(
            Input(
                name=name,
                kind=str(row.get("kind") or "small_molecule"),
                display=str(row.get("display") or name),
            )
        )

    out_row = data.get("output") or {}
    if not isinstance(out_row, dict):
        return None
    rep = str(out_row.get("name", "")).strip()
    if not any(rep.lower() == r.lower() for r in supported_reporters()):
        return None

    log = data.get("logic") or {}
    if not isinstance(log, dict):
        return None
    op = str(log.get("op", "AND")).upper().strip()
    opands = log.get("operands") or []
    if not isinstance(opands, list) or not opands:
        return None

    norm_ops: List[str] = []
    for o in opands:
        s = str(o).strip()
        mo = re.match(r"^NOT\s*\(\s*([a-z0-9_]+)\s*\)$", s, re.I)
        if mo:
            inner = canonical_inducer_name(mo.group(1)) or mo.group(1).lower()
            norm_ops.append(f"NOT({inner})")
        else:
            norm_ops.append(canonical_inducer_name(s) or s.lower())

    input_names = {i.name for i in inputs}
    for o in norm_ops:
        bare = o
        if bare.startswith("NOT(") and bare.endswith(")"):
            bare = bare[4:-1]
        if bare not in input_names:
            return None

    chassis = str(data.get("chassis") or "Escherichia coli K-12")
    notes = str(data.get("notes") or "")

    try:
        logic = LogicSpec(op=op, operands=norm_ops)
    except ValueError:
        return None

    out = Output(name=rep, phenotype=str(out_row.get("phenotype") or ""))
    return CircuitSpec(inputs=inputs, output=out, logic=logic, chassis=chassis, notes=notes)


def extract_circuit_spec(user_prompt: str, *, progress_cb=None) -> tuple[Optional[CircuitSpec], dict]:
    """Return ``(CircuitSpec or None if not applicable), raw_intent_dict``."""

    from inference import generate_text_gemma4_custom

    if progress_cb:
        try:
            progress_cb("circuit_synth · extracting boolean intent (Gemma)…")
        except Exception:
            pass

    user = f"**User brief**\n{user_prompt.strip()}\n\nOutput JSON now."
    raw = generate_text_gemma4_custom(
        user,
        system_instruction=_INTENT_SYSTEM,
        temperature=0.1,
        max_output_tokens=2048,
        stop_sequences=[],
        debug_ctx="circuit_intent",
    )
    data: Dict[str, Any]
    try:
        data = _parse_intent_json(raw)
    except ValueError:
        raw2 = generate_text_gemma4_custom(
            user + "\n\n" + _REPAIR,
            system_instruction=_INTENT_SYSTEM,
            temperature=0.05,
            max_output_tokens=1024,
            stop_sequences=[],
            debug_ctx="circuit_intent_retry",
        )
        data = _parse_intent_json(raw2)

    spec = _coerce_spec(data)
    data["_raw_model"] = raw[:8000]
    return spec, data


def spec_to_dict(spec: CircuitSpec) -> dict:
    return {
        "inputs": [
            {"name": i.name, "kind": i.kind, "display": i.display} for i in spec.inputs
        ],
        "output": {"name": spec.output.name, "phenotype": spec.output.phenotype},
        "logic": {"op": spec.logic.op, "operands": list(spec.logic.operands)},
        "chassis": spec.chassis,
        "notes": spec.notes,
    }
