"""RAG-first compile pipeline: retrieve registry parts before the LLM; DNA is assembled only from JSONL.

Flow: extract biological intent (Gemma) → Chroma retrieval (top-k per query) → compiler prompt with
part menu → Gemma may issue native ``search_igem_registry`` tool calls for extra Chroma lookups →
ordered BBa list → concatenate ``sequence`` fields from the registry.

Requires hosted Gemma (same API key as ``inference.generate_text_gemma4``). The **default** web compile
mode is ``DGENE_COMPILE_MODE=circuit_synth`` (:mod:`circuit_pipeline`); set ``DGENE_COMPILE_MODE=rag_first``
for this path exclusively. Legacy channel-DNA compile stays available via ``DGENE_COMPILE_MODE=legacy``.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Callable, Dict, List, Optional, Tuple

# BBa IDs in the registry use varying digit lengths (e.g. BBa_K3883001).
_BBA_PART_RE = re.compile(r"\bBBa_[A-Z]\d{4,}[a-zA-Z]?\b")


def _strip_markdownish_header(raw: str) -> str:
    """Normalize a line so we can match ORDERED_PART_LIST across Gemma formatting quirks."""

    s = (raw or "").strip()
    s = s.strip("`").strip()
    s = re.sub(r"^#+\s*", "", s)
    s = re.sub(r"^\*\*|\*\*$", "", s).strip()
    return s


def _line_is_ordered_marker(raw: str) -> bool:
    s = _strip_markdownish_header(raw)
    return bool(re.match(r"(?i)^ORDERED_PART_LIST\s*:?\s*$", s))


_REASONING_TAGS = re.compile(
    r"<reasoning>\s*(.*?)\s*</reasoning>",
    re.DOTALL | re.IGNORECASE,
)
# After ORDERED_PART_LIST: optional bullet / number, BBa id, optional short "(note)", trailing period.
_ORDERED_LINE = re.compile(
    r"^\s*(?:[-*•]+\s*|\d+[.)]\s+)?(BBa_[A-Z]\d{4,}[a-zA-Z]?)(?:\s*\([^)]{0,160}\))?\s*[.\u3002]?\s*$"
)

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSONL = os.path.join(_MODULE_DIR, "igem_dataset.jsonl")

_JSONL_INDEX_LOCK = threading.Lock()
_JSONL_BY_NAME: Optional[Dict[str, dict]] = None


def _jsonl_path() -> str:
    return os.environ.get("DGENE_IGEM_JSONL", _DEFAULT_JSONL).strip() or _DEFAULT_JSONL


def _ensure_jsonl_index() -> Dict[str, dict]:
    global _JSONL_BY_NAME
    with _JSONL_INDEX_LOCK:
        if _JSONL_BY_NAME is not None:
            return _JSONL_BY_NAME
        path = _jsonl_path()
        by_name: Dict[str, dict] = {}
        if not os.path.isfile(path):
            _JSONL_BY_NAME = by_name
            return by_name
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                pn = str(row.get("part_name", "")).strip()
                if pn:
                    by_name[pn] = row
        _JSONL_BY_NAME = by_name
        return _JSONL_BY_NAME


def compile_mode() -> str:
    v = (os.environ.get("DGENE_COMPILE_MODE") or "circuit_synth").strip().lower()
    if v in ("circuit_synth", "rag_first", "legacy"):
        return v
    return "circuit_synth"


def rag_first_configured() -> bool:
    """RAG-first needs hosted Gemma (API key) for intent + compiler calls."""

    from inference import _pick_google_api_key

    return bool(_pick_google_api_key())


def rag_first_top_k_per_query() -> int:
    raw = (os.environ.get("DGENE_RAG_FIRST_TOP_K") or "15").strip()
    try:
        return max(1, min(50, int(raw)))
    except ValueError:
        return 15


def rag_first_max_ordered_parts() -> int:
    """Reject compiler outputs longer than this (prevents runaway megabase concatenations)."""

    raw = (os.environ.get("DGENE_RAG_FIRST_MAX_PARTS") or "48").strip()
    try:
        return max(6, min(200, int(raw)))
    except ValueError:
        return 48


def rag_first_compiler_max_output_tokens() -> int:
    raw = (os.environ.get("DGENE_RAG_FIRST_COMPILER_MAX_TOKENS") or "3072").strip()
    try:
        return max(512, min(8192, int(raw)))
    except ValueError:
        return 3072


def rag_first_reasoning_display_max_chars() -> int:
    raw = (os.environ.get("DGENE_RAG_FIRST_REASONING_CHARS") or "1400").strip()
    try:
        return max(200, min(20000, int(raw)))
    except ValueError:
        return 1400


def rag_first_reasoning_max_sentences() -> int:
    raw = (os.environ.get("DGENE_RAG_FIRST_REASONING_SENTENCES") or "7").strip()
    try:
        return max(2, min(24, int(raw)))
    except ValueError:
        return 7


_INTERNAL_DIALOGUE_LINE = re.compile(
    r"(?i)^\s*(wait[,.]?|actually[,.]?|hmm[,.]?|oh[,.]?\s|"
    r"let\'?s\s+(look|try|re-examine|check)|looking\s+at|"
    r"option\s+[a-z]\s*:|checking\s+(the\s+)?menu|"
    r"i\s+don\'?t\s+see|what\s+if\s+we|"
    r"alternatively[,:]|correct\s+AND\s+gate|"
    r"step\s*\d+\s*:|building\s+an\s+AND)",
)

_LATEXISH = re.compile(r"\$[^$]{0,80}\$|\\\w+")


def _scrub_reasoning_prose(body: str) -> str:
    """Drop bullet scaffolding and obvious stream-of-consciousness lines."""

    chunks: List[str] = []
    for line in (body or "").splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\s*[*•\-]+\s*", "", s)
        s = re.sub(r"^\s*\d+[.)]\s+", "", s)
        if _INTERNAL_DIALOGUE_LINE.match(s):
            continue
        if re.match(r"(?i)^\s*`?roles?`?\s*:", s):
            continue
        chunks.append(s)
    text = " ".join(chunks)
    text = _LATEXISH.sub(" ", text)
    text = re.sub(r"[`]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _drop_chatter_sentences(text: str) -> str:
    """Remove sentences that read like model self-dialogue (not design content)."""

    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    keep: List[str] = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        low = s.lower()
        if re.match(r"(?i)^(wait|actually|hmm|oh)\b[,!\s]", s):
            continue
        if re.match(r"(?i)^let'?s\b", s):
            continue
        if re.match(r"(?i)^(looking|checking)\s+at\b", s):
            continue
        if re.match(r"(?i)^option\s+[a-z]\b", s):
            continue
        if re.match(r"(?i)^what\s+if\b", s):
            continue
        if re.match(r"(?i)^i\s+don'?t\s+see\b", s):
            continue
        if re.match(r"(?i)^alternatively\b", s):
            continue
        if "internal dialogue" in low:
            continue
        keep.append(s)
    return " ".join(keep)


def _clip_to_sentences(text: str, max_sentences: int, max_chars: int) -> str:
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    out: List[str] = []
    for p in parts[:max_sentences]:
        candidate = (" ".join(out + [p])).strip()
        if len(candidate) > max_chars:
            if out:
                break
            return (p[: max_chars - 1].rsplit(" ", 1)[0] + "…") if len(p) > max_chars else p
        out.append(p)
    joined = " ".join(out).strip()
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1].rsplit(" ", 1)[0] + "…"
    return joined


def _normalize_bba_registry_name(part_name: str) -> str:
    """Map legacy nicknames (e.g. BBa_B0034) to registry IDs present in ``igem_dataset.jsonl``."""

    from igem_rag import _PART_ALIASES

    pn = (part_name or "").strip()
    if not pn.startswith("BBa_"):
        return pn
    tail = pn[4:].casefold()
    return _PART_ALIASES.get(tail) or pn


def _progress(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        try:
            cb(msg)
        except Exception:
            pass


_INTENT_SYSTEM = """You are a synthetic biology analyst. Your job is to read a user design brief \
and extract structured biological intent for iGEM part retrieval.

Reply with ONLY a single JSON object (no markdown fences, no commentary). Use this exact schema:
{
  "gate": "AND" | "OR" | "BUF" | "NOT" | "unknown",
  "input_analytes": ["analyte 1", "analyte 2"],
  "reporter": "amilCP or GFP or mRFP1 or other CDS name stated by the user",
  "roles": [
    {
      "role": "short label e.g. reporter / sensor_A / sensor_B / logic / chassis_marker",
      "summary": "one phrase",
      "retrieval_queries": ["query 1", "query 2", ...]
    }
  ],
  "chassis": "organism if stated, else unknown",
  "logic_summary": "how regulation/combination should work in one sentence"
}

**gate / input_analytes / reporter (always fill when the brief is a biosensor or logic circuit):**
- ``gate``: AND = all listed analytes needed together; OR = any one suffices; BUF = single-input on/off; NOT = dominant repression; unknown if ambiguous.
- ``input_analytes``: use 2+ searchable phrases when the user wants two cues (examples: pyocyanin Pseudomonas, lactic acid lactate, glucose, arsenic — match their wording).
- ``reporter``: amilCP for blue pigment unless they name another CDS.

Each retrieval_queries entry must be a concrete English search phrase suitable for semantic search \
in an iGEM parts database (include organism, analyte, or protein names when relevant). \
Include 2–6 queries per role. Also add roles or queries for common needs: strong RBS, terminator, \
and genetic logic keywords if implied.

End your reply immediately after the closing brace }.
"""


_COMPILER_SYSTEM = """You are a genetic circuit compiler for BioBrick-style DNA.

Input: user brief + JSON intent + a numbered menu of registry parts (BBa IDs, types, descriptions, lengths). \
You also have an optional tool **search_igem_registry** that queries the same local iGEM index when you need \
additional verified BBa candidates not shown in the menu (or want filtered alternates by part type).

Output format — copy this structure exactly (no markdown headings, no LaTeX, no nested bullet essays):

<reasoning>
Maximum 7 sentences, plain English only, single short paragraph. In order: (1) what each input \
or inducer is sensed by, (2) which transcription factors or repressors mediate that, (3) how \
that combines logically to drive the output CDS/reporter, (4) one sentence on why the chosen \
promoters match those regulators (cognate pairing). \
No BBa_ IDs. No bullets, numbering, markdown, or LaTeX. No rhetorical questions or phrases like \
"Wait," "Actually," "Let's look," or "Option A:".
</reasoning>

ORDERED_PART_LIST
BBa_XXXXX
BBa_YYYYY
(one registry ID per line; optional single parenthetical note after the ID, e.g. BBa_K592009 (amilCP))
…
</circuit_design>

Hard rules:
- One linear construct only (typically 6–24 lines under ORDERED_PART_LIST). No alternatives, no repeated blocks.
- Every line under ORDERED_PART_LIST must contain exactly one BBa_ id and nothing else except an optional (note).
- You may prefix each line with a list number like `1.` or `2)` if helpful.
- Do NOT print DNA bases. Do NOT invent BBa IDs: every ID must appear either in the numbered menu **or** in a **search_igem_registry** tool result for this turn.
- Prefer the menu when it suffices; call **search_igem_registry** sparingly (typically 1–3 focused queries).
- Stop immediately after the line </circuit_design>
"""

_COMPILER_PARSE_RETRY_SUFFIX = (
    "**Parse repair — follow exactly:** Output `<reasoning>…</reasoning>` then a line with only "
    "`ORDERED_PART_LIST` (a colon is OK), then one `BBa_…` ID per line (optional `1.` / `2)` prefixes), "
    "then `</circuit_design>`. No markdown code fences, no DNA bases, no BBa IDs inside reasoning."
)


def extract_intent_json(user_prompt: str, *, progress_cb: Optional[Callable[[str], None]] = None) -> dict:
    from inference import generate_text_gemma4_custom

    _progress(progress_cb, "rag_first · step 1 · extracting biological intent (Gemma)…")
    user = (
        "**User brief**\n"
        f"{user_prompt.strip()}\n\n"
        "Output the JSON object now."
    )
    raw = generate_text_gemma4_custom(
        user,
        system_instruction=_INTENT_SYSTEM,
        temperature=0.15,
        max_output_tokens=4096,
        stop_sequences=[],
        debug_ctx="rag_first_intent",
    )
    text = raw.strip()
    # Strip accidental fences
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        intent = json.loads(text)
    except json.JSONDecodeError:
        # Try first { ... } slice
        i = text.find("{")
        j = text.rfind("}")
        if i >= 0 and j > i:
            intent = json.loads(text[i : j + 1])
        else:
            raise ValueError(f"Intent model did not return valid JSON. Raw head: {text[:500]!r}") from None
    # Normalize optional keys newer templates rely on (older models omit them).
    if "gate" not in intent:
        intent["gate"] = "unknown"
    if "input_analytes" not in intent:
        intent["input_analytes"] = intent.get("inputs_analytes") or []
    if "reporter" not in intent:
        intent["reporter"] = ""
    return intent


def _flatten_retrieval_queries(intent: dict, user_prompt: str = "") -> List[str]:
    out: List[str] = []
    seen: set = set()

    def add(q: str) -> None:
        q = (q or "").strip()
        if len(q) < 3:
            return
        k = q.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(q)

    roles = intent.get("roles")
    if isinstance(roles, list):
        for r in roles:
            if not isinstance(r, dict):
                continue
            for q in r.get("retrieval_queries") or []:
                if isinstance(q, str):
                    add(q)
    u = (user_prompt or "").casefold()
    if "amilcp" in u or "chromoprotein" in u or "blue pigment" in u:
        add("amilCP blue chromoprotein CDS reporter iGEM")
    for extra in (
        "strong RBS BioBrick iGEM",
        "strong double terminator iGEM B0015",
        "genetic AND gate transcriptional logic iGEM",
    ):
        add(extra)
    return out


def build_part_menu(
    intent: dict,
    *,
    user_prompt: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
    top_k: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Return (menu_rows for the prompt, part_name -> row dict with sequence)."""

    from igem_rag import RetrievedPart, ensure_indexed, retrieve_parts

    k = top_k if top_k is not None else rag_first_top_k_per_query()
    ensure_indexed(progress_cb=progress_cb)
    queries = _flatten_retrieval_queries(intent, user_prompt=user_prompt)
    by_name: Dict[str, dict] = {}

    _progress(
        progress_cb,
        f"rag_first · step 2 · RAG retrieval · {len(queries)} query strings · top {k} each…",
    )
    for qi, q in enumerate(queries):
        _progress(progress_cb, f"rag_first · retrieve · [{qi + 1}/{len(queries)}] {q[:72]!r}…")
        hits: List[RetrievedPart] = retrieve_parts(q, top_k=k)
        for h in hits:
            if not h.part_name:
                continue
            if h.part_name not in by_name:
                by_name[h.part_name] = {
                    "part_name": h.part_name,
                    "part_type": h.part_type,
                    "short_desc": h.short_desc,
                    "sequence": h.sequence,
                    "similarity": h.similarity,
                    "match_kind": getattr(h, "match_kind", "semantic"),
                    "retrieval_query": q,
                }

    # Stable sort: promoters/RBS before CDS for readability (optional heuristic)
    type_rank = {
        "Promoter": 0,
        "RBS": 1,
        "Protein Domain": 2,
        "CDS": 3,
        "Terminator": 4,
    }
    rows = list(by_name.values())
    rows.sort(
        key=lambda r: (
            type_rank.get(str(r.get("part_type", "")), 99),
            str(r.get("part_name", "")),
        )
    )
    menu = []
    for i, r in enumerate(rows, start=1):
        seq = str(r.get("sequence", ""))
        menu.append(
            {
                "n": i,
                "part_name": r["part_name"],
                "part_type": r.get("part_type", ""),
                "short_desc": (r.get("short_desc") or "")[:280],
                "length_bp": len(seq),
            }
        )
    _progress(progress_cb, f"rag_first · menu · {len(menu)} unique parts after merge")
    return menu, by_name


def _format_menu_for_prompt(menu: List[dict]) -> str:
    lines = []
    for m in menu:
        lines.append(
            f"{m['n']}. {m['part_name']} [{m.get('part_type', '')}] "
            f"({m.get('length_bp', 0)} bp) — {m.get('short_desc', '')}"
        )
    return "\n".join(lines)


def _merge_igem_tool_rows_into(by_name: Dict[str, dict]) -> None:
    """Fold native-tool Chroma hits into the menu map so assembly retains similarity metadata."""

    from inference import drain_igem_tool_merge_rows

    for row in drain_igem_tool_merge_rows():
        pn = str(row.get("part_name", "")).strip()
        if pn and pn not in by_name:
            by_name[pn] = row


def run_compiler(
    user_prompt: str,
    menu: List[dict],
    intent: dict,
    *,
    temperature: float = 0.35,
    progress_cb: Optional[Callable[[str], None]] = None,
    extra_user_suffix: str = "",
    menu_by_name: Optional[Dict[str, dict]] = None,
) -> str:
    from inference import generate_text_gemma4_custom

    _progress(progress_cb, "rag_first · step 3–4 · circuit compiler (Gemma, menu-constrained)…")
    user = (
        f"**User brief**\n{user_prompt.strip()}\n\n"
        f"**Extracted intent (JSON)**\n{json.dumps(intent, ensure_ascii=False)[:6000]}\n\n"
        "**Verified iGEM parts menu (ONLY source for BBa IDs and DNA)**\n"
        f"{_format_menu_for_prompt(menu)}\n\n"
        "Respond using the exact template in your instructions: <reasoning>…</reasoning>, "
        "then ORDERED_PART_LIST, then one BBa per line, then </circuit_design>."
    )
    if extra_user_suffix.strip():
        user = f"{user}\n\n{extra_user_suffix.strip()}"
    raw = generate_text_gemma4_custom(
        user,
        system_instruction=_COMPILER_SYSTEM,
        temperature=temperature,
        max_output_tokens=rag_first_compiler_max_output_tokens(),
        stop_sequences=["</circuit_design>"],
        debug_ctx="rag_first_compiler",
        igem_tools=True,
    )
    if menu_by_name is not None:
        _merge_igem_tool_rows_into(menu_by_name)
    return raw


def _fallback_one_bba_per_line(section_lines: List[str]) -> List[str]:
    """When strict line regexes miss (extra punctuation, odd spacing) but the model did list one ID per line."""

    ordered: List[str] = []
    for raw in section_lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        low = line.lower()
        if "</circuit_design>" in low or low == "</circuit_design>":
            break
        matches = _BBA_PART_RE.findall(line)
        if len(matches) == 1:
            ordered.append(matches[0])
        elif len(matches) > 1:
            break
    return ordered


def parse_ordered_bba(compiler_output: str) -> List[str]:
    """Parse only the ORDERED_PART_LIST section (or strict one-BBa-per-line lines).

    **Critical:** We must not scan free-form prose — every ``BBa_`` mention in chain-of-thought
    would otherwise be assembled into a megabase plasmid.
    """

    text = compiler_output or ""
    if "</circuit_design>" in text:
        text = text.split("</circuit_design>", 1)[0]
    lines = text.splitlines()

    start_idx: Optional[int] = None
    for i, raw in enumerate(lines):
        if _line_is_ordered_marker(raw):
            start_idx = i + 1
            break

    if start_idx is not None:
        section = lines[start_idx:]
    else:
        section = lines

    ordered: List[str] = []
    for raw in section:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        mo = _ORDERED_LINE.match(line)
        if mo:
            ordered.append(mo.group(1))
            continue
        # Tolerate "BBa_XXX — description" only when under explicit marker (already in section)
        if start_idx is not None:
            mo2 = re.match(
                r"^\s*(?:[-*•]+\s*|\d+[.)]\s+)?(BBa_[A-Z]\d{4,}[a-zA-Z]?)\s*(?:[—\-]\s*.+)?\s*[.\u3002]?\s*$",
                line,
            )
            if mo2 and len(_BBA_PART_RE.findall(line)) == 1:
                ordered.append(mo2.group(1))
    if not ordered and start_idx is not None:
        ordered = _fallback_one_bba_per_line(section)
    return ordered


def extract_reasoning_for_display(compiler_output: str) -> str:
    """Short, readable text for the UI — not the full raw compiler stream."""

    from inference import sanitize_thought_for_display

    t = (compiler_output or "").strip()
    cap = rag_first_reasoning_display_max_chars()
    max_sent = rag_first_reasoning_max_sentences()
    m = _REASONING_TAGS.search(t)
    if m:
        body = m.group(1).strip()
    else:
        pre_lines: List[str] = []
        for line in t.splitlines():
            if _line_is_ordered_marker(line):
                break
            pre_lines.append(line)
        body = "\n".join(pre_lines).strip()
        body = _BBA_PART_RE.sub("", body)
        body = re.sub(r"\s+", " ", body).strip()
        if len(body) > 6000:
            body = body[:6000] + "…"

    body = _scrub_reasoning_prose(body)
    body = sanitize_thought_for_display(body)
    body = _drop_chatter_sentences(body)
    body = _clip_to_sentences(body, max_sent, cap)
    return body


def _part_type_to_map_sub(part_type: str) -> str:
    """Map iGEM ``part_type`` strings to plasmid-map categories (see ``SUB_COLOR`` in ``app.js``)."""

    t = (part_type or "").strip().lower()
    if not t:
        return "feature"
    if "promoter" in t:
        return "promoter"
    if "terminator" in t:
        return "terminator"
    if "rbs" in t or "ribosome" in t:
        return "rbs"
    if t in ("cds",) or "coding" in t or "protein domain" in t or t == "orf":
        return "cds"
    if "operator" in t:
        return "operator"
    return "feature"


def _rag_parts_for_ui_from_trace(
    trace: List[dict],
    menu_by_name: Optional[Dict[str, dict]] = None,
) -> List[dict]:
    """Build ``rag.parts`` rows for ``web/js/app.js`` ``renderRagPanel``."""

    menu = menu_by_name or {}
    out: List[dict] = []
    for t in trace:
        pn = str(t.get("normalized_name") or t.get("part_name") or "").strip()
        alt = str(t.get("part_name") or "").strip()
        row = menu.get(pn) or menu.get(alt) or {}
        ok = bool(t.get("ok", True))
        src = str(t.get("source") or "")
        if src == "missing" or not ok:
            out.append(
                {
                    "part_name": pn or alt,
                    "part_type": t.get("part_type"),
                    "verified": False,
                    "sequence_source": "model",
                    "similarity": 0.0,
                    "query": "",
                }
            )
            continue
        sim_raw = row.get("similarity")
        try:
            sim = float(sim_raw) if sim_raw is not None else 1.0
        except (TypeError, ValueError):
            sim = 1.0
        q = str(row.get("retrieval_query") or "")
        registry = src in ("menu", "jsonl", "slot_template")
        out.append(
            {
                "part_name": pn or alt,
                "part_type": t.get("part_type"),
                "verified": registry,
                "sequence_source": "registry" if registry else "model",
                "similarity": sim,
                "query": q,
            }
        )
    return out


def _rag_detail_min_similarity() -> float:
    raw = (os.environ.get("DGENE_RAG_MIN_SIM") or "0.6").strip()
    try:
        return max(0.3, min(0.95, float(raw)))
    except ValueError:
        return 0.6


def assemble_sequence(
    ordered_parts: List[str],
    *,
    menu_by_name: Dict[str, dict],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[dict]]:
    """Concatenate registry sequences in order. Each entry in trace describes one segment."""

    index = _ensure_jsonl_index()
    seq_parts: List[str] = []
    trace: List[dict] = []
    pos = 1
    for raw_name in ordered_parts:
        pname = _normalize_bba_registry_name(raw_name)
        row = menu_by_name.get(pname) or menu_by_name.get(raw_name) or {}
        dna = str(row.get("sequence", "")).strip()
        source = "menu"
        if not dna:
            jr = index.get(pname) or index.get(raw_name)
            if jr:
                dna = str(jr.get("sequence", "")).strip()
                source = "jsonl"
        ptype = str(row.get("part_type", "") or "").strip()
        if not ptype and source == "jsonl":
            jr_meta = index.get(pname) or index.get(raw_name) or {}
            ptype = str(jr_meta.get("part_type", "") or "").strip()
        sub = _part_type_to_map_sub(ptype)
        label = pname if pname.startswith("BBa_") else raw_name
        if not dna:
            _progress(
                progress_cb,
                f"rag_first · WARN · no sequence for {raw_name!r}"
                + (f" (normalized {pname!r})" if pname != raw_name else "")
                + " — skipped",
            )
            trace.append(
                {
                    "part_name": raw_name,
                    "normalized_name": pname,
                    "part_type": ptype or None,
                    "label": label,
                    "sub": sub,
                    "ok": False,
                    "bp": 0,
                    "source": "missing",
                }
            )
            continue
        clean = "".join(dna.upper().split())
        seq_parts.append(clean)
        L = len(clean)
        trace.append(
            {
                "part_name": raw_name,
                "normalized_name": pname,
                "part_type": ptype or None,
                "label": label,
                "sub": sub,
                "ok": True,
                "bp": L,
                "source": source,
                "start_bp": pos,
                "end_bp": pos + L - 1,
            }
        )
        pos += L
    return "".join(seq_parts), trace


def run_rag_first_single(
    user_prompt: str,
    *,
    temperature: float = 0.35,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """One full RAG-first compile. Returns a :class:`inference.Candidate`."""

    from inference import Candidate

    intent = extract_intent_json(user_prompt, progress_cb=progress_cb)
    try:
        from slot_template_compile import candidate_from_slot_template, slot_template_enabled

        if slot_template_enabled():
            st = candidate_from_slot_template(
                user_prompt, intent, progress_cb=progress_cb
            )
            if st is not None:
                st.candidate_id = "cand_0"
                return st
    except Exception as exc:
        _progress(
            progress_cb,
            f"slot_template · WARN · {type(exc).__name__}: {exc} — falling back to menu compiler…",
        )

    menu, by_name = build_part_menu(
        intent, user_prompt=user_prompt, progress_cb=progress_cb
    )
    if not menu:
        raise RuntimeError(
            "RAG-first retrieval returned an empty part menu — check igem_dataset.jsonl and Chroma index."
        )
    compiler_out = run_compiler(
        user_prompt,
        menu,
        intent,
        temperature=temperature,
        progress_cb=progress_cb,
        menu_by_name=by_name,
    )
    ordered = parse_ordered_bba(compiler_out)
    if not ordered:
        _progress(
            progress_cb,
            "rag_first · compiler reply missing BBa list; one strict retry…",
        )
        compiler_out = run_compiler(
            user_prompt,
            menu,
            intent,
            temperature=min(0.12, temperature),
            progress_cb=progress_cb,
            extra_user_suffix=_COMPILER_PARSE_RETRY_SUFFIX,
            menu_by_name=by_name,
        )
        ordered = parse_ordered_bba(compiler_out)
    if not ordered:
        raise RuntimeError(
            "RAG-first compiler output contained no BBa part IDs — raw head: "
            f"{compiler_out[:1200]!r}"
        )
    cap = rag_first_max_ordered_parts()
    if len(ordered) > cap:
        raise RuntimeError(
            f"RAG-first compiler listed {len(ordered)} parts (limit {cap}). "
            "The model likely enumerated alternatives or repeated blocks — retry or raise "
            "`DGENE_RAG_FIRST_MAX_PARTS`. Raw head: "
            f"{compiler_out[:600]!r}"
        )
    dna, trace = assemble_sequence(ordered, menu_by_name=by_name, progress_cb=progress_cb)
    if not dna:
        raise RuntimeError(
            "RAG-first assembly produced empty DNA — check that ordered parts exist in the registry."
        )
    _progress(
        progress_cb,
        f"rag_first · step 5 · assembled {len(dna)} bp from {len(trace)} segments",
    )
    thought = extract_reasoning_for_display(compiler_out)
    detail = {
        "enabled": True,
        "applied": True,
        "min_similarity": _rag_detail_min_similarity(),
        "pipeline": "rag_first",
        "intent": intent,
        "retrieval_unique_parts": len(menu),
        "ordered_part_names": ordered,
        "assembly_trace": trace,
        "compiler_raw_chars": len(compiler_out),
        "compiler_raw": compiler_out,
        "map_slots": trace,
        "parts": _rag_parts_for_ui_from_trace(trace, by_name),
    }
    return Candidate(
        candidate_id="cand_0",
        thought=thought,
        sequence=dna,
        strategy=f"rag_first T{temperature:g}",
        strategy_name="RAG-first (registry DNA only)",
        raw=f"{thought}\n\n[assembled {len(dna)} bp]\n\nORDERED: {' → '.join(ordered)}",
        rag_first_detail=detail,
    )


def rag_first_candidate_temps(n: int) -> List[float]:
    base = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.1]
    return [base[i % len(base)] for i in range(n)]


def run_rag_first_variants_iter(
    user_prompt: str,
    n: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Yield ``Candidate`` objects one at a time (shared intent + menu)."""

    from inference import Candidate

    intent = extract_intent_json(user_prompt, progress_cb=progress_cb)
    yielded = 0

    slot_cand = None
    try:
        from slot_template_compile import candidate_from_slot_template, slot_template_enabled

        if slot_template_enabled() and n > 0:
            slot_cand = candidate_from_slot_template(
                user_prompt, intent, progress_cb=progress_cb
            )
    except Exception as exc:
        _progress(progress_cb, f"slot_template · WARN · {type(exc).__name__}: {exc}")

    if slot_cand is not None:
        yield slot_cand
        yielded += 1

    llm_budget = max(0, n - yielded)
    if llm_budget <= 0:
        return

    menu, by_name = build_part_menu(
        intent, user_prompt=user_prompt, progress_cb=progress_cb
    )
    if not menu:
        raise RuntimeError("RAG-first retrieval returned an empty part menu.")

    temps = rag_first_candidate_temps(llm_budget)
    last_err: Optional[str] = None
    for i, temp in enumerate(temps):
        _progress(
            progress_cb,
            f"rag_first · variant {i + 1}/{llm_budget} · compiler T={temp:g} (success so far: {yielded})…",
        )
        try:
            compiler_out = run_compiler(
                user_prompt,
                menu,
                intent,
                temperature=temp,
                progress_cb=progress_cb,
                menu_by_name=by_name,
            )
            ordered = parse_ordered_bba(compiler_out)
            if not ordered:
                _progress(
                    progress_cb,
                    f"rag_first · variant {i + 1}/{llm_budget} · missing BBa list; one strict retry…",
                )
                compiler_out = run_compiler(
                    user_prompt,
                    menu,
                    intent,
                    temperature=min(0.12, temp),
                    progress_cb=progress_cb,
                    extra_user_suffix=_COMPILER_PARSE_RETRY_SUFFIX,
                    menu_by_name=by_name,
                )
                ordered = parse_ordered_bba(compiler_out)
            if not ordered:
                raise ValueError(
                    "no BBa IDs in compiler output after retry — "
                    f"raw head: {compiler_out[:900]!r}"
                )
            cap = rag_first_max_ordered_parts()
            if len(ordered) > cap:
                raise ValueError(
                    f"compiler listed {len(ordered)} parts (limit {cap}); shorten ORDERED_PART_LIST"
                )
            dna, trace = assemble_sequence(
                ordered, menu_by_name=by_name, progress_cb=progress_cb
            )
            if not dna:
                raise ValueError("assembled empty DNA (missing sequences in menu/jsonl)")
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            _progress(progress_cb, f"rag_first · WARN · skipped variant {i + 1}/{llm_budget} — {last_err}")
            continue

        thought = extract_reasoning_for_display(compiler_out)
        detail = {
            "enabled": True,
            "applied": True,
            "min_similarity": _rag_detail_min_similarity(),
            "pipeline": "rag_first",
            "intent": intent,
            "retrieval_unique_parts": len(menu),
            "ordered_part_names": ordered,
            "assembly_trace": trace,
            "compiler_temperature": temp,
            "compiler_raw": compiler_out,
            "map_slots": trace,
            "parts": _rag_parts_for_ui_from_trace(trace, by_name),
        }
        yield Candidate(
            candidate_id=f"cand_{yielded}",
            thought=thought,
            sequence=dna,
            strategy=f"rag_first T{temp:g}",
            strategy_name="RAG-first (registry DNA only)",
            raw=f"{thought}\n\n[assembled {len(dna)} bp]\n\nORDERED: {' → '.join(ordered)}",
            rag_first_detail=detail,
        )
        yielded += 1

    if yielded == 0:
        raise RuntimeError(
            "RAG-first produced no successful variants after "
            f"{n} attempt(s). Last error: {last_err or 'unknown'}"
        )


def run_rag_first_variants(
    user_prompt: str,
    n: int,
    *,
    progress_cb: Optional[Callable[[str], None]] = None,
):
    """Run N compiler samples (shared intent + shared menu). Returns ``Candidate`` list."""

    return list(
        run_rag_first_variants_iter(user_prompt, n, progress_cb=progress_cb)
    )
