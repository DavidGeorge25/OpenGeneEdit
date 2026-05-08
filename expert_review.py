"""Optional Gemma 'PhD reviewer' pass over an ordered part list (extra completion).

Runs via ``generate_text_gemma4_custom`` — hosted API or local GGUF per ``get_backend()``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

_SYSTEM = """You are a picky synthetic biology PI reviewing a **linear BioBrick order** for an \
**E. coli** plasmid. Be concrete and conservative.

Reply with ONLY a JSON object:
{
  "verdict": "likely_coherent" | "needs_revision" | "unclear",
  "summary": "2–4 sentences in plain English — does the inducer/sensor → regulator → output logic \
make sense for these parts? Call out any mismatch.",
  "concerns": ["short bullet strings, max 5, or empty list"]
}

Do not invent BBa IDs not shown in the input. If the list is too fragmented to judge, use \
verdict "unclear" and explain briefly.
"""


def _strip_fence(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        t = "\n".join(lines).strip()
    return t


def _menu_lines(ordered: List[str], descriptions: Dict[str, str]) -> str:
    lines = []
    for i, bba in enumerate(ordered, start=1):
        d = (descriptions.get(bba) or "").replace("\n", " ").strip()[:220]
        lines.append(f"{i}. {bba} — {d}")
    return "\n".join(lines)


def _load_desc_map(ordered: List[str]) -> Dict[str, str]:
    from circuit_parts import _load_jsonl_index

    idx = _load_jsonl_index()
    out: Dict[str, str] = {}
    for bba in ordered:
        row = idx.get(bba)
        if row:
            out[bba] = str(row.get("short_desc", "") or "")
    return out


def expert_gemma_review(user_prompt: str, ordered_bb_as: List[str]) -> Optional[dict]:
    """Return parsed reviewer JSON, or ``None`` if skipped (no LLM / empty list)."""

    from inference import generate_text_gemma4_custom, hosted_generation_ready

    if not hosted_generation_ready():
        return None
    if not ordered_bb_as:
        return None
    v = (os.environ.get("DGENE_EXPERT_REVIEW") or "").strip().lower()
    if v not in ("1", "true", "yes", "on"):
        return None

    desc = _load_desc_map([str(x).strip() for x in ordered_bb_as if str(x).strip()])
    menu = _menu_lines([str(x).strip() for x in ordered_bb_as if str(x).strip()], desc)

    user = (
        f"**User brief**\n{user_prompt.strip()}\n\n"
        f"**Ordered parts (5' → 3' along the insert as emitted by the compiler)**\n{menu}\n\n"
        "JSON only."
    )
    raw = generate_text_gemma4_custom(
        user,
        system_instruction=_SYSTEM,
        temperature=0.2,
        max_output_tokens=1024,
        stop_sequences=[],
        debug_ctx="expert_review",
    )
    text = _strip_fence(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i = text.find("{")
        j = text.rfind("}")
        if i >= 0 and j > i:
            return json.loads(text[i : j + 1])
    mo = re.search(
        r'"verdict"\s*:\s*"([^"]+)"[\s\S]*?"summary"\s*:\s*"([^"]*)"',
        text,
    )
    if mo:
        return {
            "verdict": mo.group(1),
            "summary": mo.group(2),
            "concerns": [],
            "_parse": "partial",
        }
    return {
        "verdict": "unclear",
        "summary": "Reviewer output was not valid JSON.",
        "concerns": [text[:400]],
        "_parse": "failed",
    }
