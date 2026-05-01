"""Inference layer for DGene.

Backends are chosen at runtime:

  - ``MockBackend`` always works (stdlib only) and returns N synthetic
    candidates with deterministic per-prompt seeds. Used for the demo flow.

  - ``GGUFBackend`` is auto-selected when ``DGENE_GGUF_PATH`` points at a
    valid .gguf file (drop the merged + quantized Gemma 4 fine-tune there
    once it's ready). Requires ``llama-cpp-python``; gracefully falls back
    to mock if the dependency is missing.

The legacy ``run_mock_inference`` and ``parse_thought_and_sequence`` exports
are preserved so the old Streamlit ``app.py`` keeps working.
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Mock corpus — short E. coli expression cassette (sfGFP-style reporter).
# ---------------------------------------------------------------------------

MOCK_SEQUENCE = (
    "TTGACATGATAAGTAAGGAGGTTTAAATGATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCC"
    "CAATTCTTGTTGAATTAGATGGTCATCCGCTTGAGCTACCATTATCAACAAAATACTCCAATTGGC"
    "GATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCC"
    "AACGAAAAGCGTGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATG"
    "GATGAACTATACAAATAA"
)

# Each design strategy = (label, thought template, mutation rate, codon bias)
DESIGN_STRATEGIES = [
    {
        "label": "high-expression",
        "name": "High expression (J23100 + B0034)",
        "thought": (
            "Selected the J23100 strong constitutive promoter paired with the canonical B0034 "
            "RBS to maximize translation initiation. Codons in the CDS were biased toward the "
            "most-frequent E. coli synonyms to push CAI above 0.8. Trade-off: higher metabolic "
            "burden from sustained transcription."
        ),
        "mut_rate": 0.02,
        "codon_bias": "preferred",
    },
    {
        "label": "balanced",
        "name": "Balanced (J23106 + B0032)",
        "thought": (
            "Dialed expression down with the J23106 medium promoter and B0032 RBS. This keeps "
            "the cassette in a regime where the host can sustain growth without depleting "
            "tRNA pools. CAI tuned to ~0.7 to leave headroom for native protein synthesis."
        ),
        "mut_rate": 0.05,
        "codon_bias": "balanced",
    },
    {
        "label": "low-burden",
        "name": "Low burden (J23114 + weak SD)",
        "thought": (
            "Optimized for minimal host impact: J23114 weak promoter, deliberately weakened "
            "Shine-Dalgarno spacing, GC content nudged toward 50% for stability. Expression "
            "is the lowest of the candidates but the construct should be near-neutral on growth."
        ),
        "mut_rate": 0.08,
        "codon_bias": "balanced",
    },
    {
        "label": "clean-assembly",
        "name": "Clean assembly (Type IIS scrubbed)",
        "thought": (
            "Same expression strategy as the balanced variant, but every BsaI/BsmBI/BbsI site "
            "in the CDS was removed via silent codon swaps so the part drops directly into "
            "Golden Gate / MoClo assemblies without re-domestication."
        ),
        "mut_rate": 0.04,
        "codon_bias": "clean",
    },
]


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    candidate_id: str
    thought: str
    sequence: str
    strategy: str = ""
    strategy_name: str = ""
    raw: str = ""


# ---------------------------------------------------------------------------
# Helpers shared by all backends
# ---------------------------------------------------------------------------


def parse_thought_and_sequence(model_output: str) -> Tuple[str, str]:
    """Extract thought + DNA from the canonical training tag format.

    Format::

        <|channel>thought
        ...reasoning...
        <channel|>
        DNA...
    """
    pattern = re.compile(
        r"<\|channel\>thought\s*(.*?)\s*<channel\|>\s*([ACGTNacgtn\s]+)\s*$",
        re.DOTALL,
    )
    match = pattern.search(model_output)
    if match:
        thought = match.group(1).strip()
        sequence = re.sub(r"\s+", "", match.group(2)).upper()
        return thought, sequence

    if "<|channel>thought" in model_output and "<channel|>" in model_output:
        thought_part, seq_part = model_output.split("<channel|>", 1)
        thought = thought_part.replace("<|channel>thought", "", 1).strip()
        sequence = re.sub(r"[^ACGTNacgtn]", "", seq_part).upper()
        if thought and sequence:
            return thought, sequence

    raise ValueError("Could not parse thought and DNA sequence from model output.")


def _seed_for(prompt: str, idx: int) -> int:
    h = 0
    for ch in prompt:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h ^ (idx * 2654435761)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


# Synonym map for codon "bias" knobs in the mock generator. Real Gemma will
# obviously do this end-to-end; the mock just needs enough variation that
# the passes/ranker produce visibly different scores per candidate.
_PREFERRED_SWAP = {
    "AAA": "AAA", "AAG": "AAA",     # Lys → preferred AAA
    "GAA": "GAA", "GAG": "GAA",     # Glu → GAA
    "CGT": "CGT", "CGC": "CGT", "CGA": "CGT", "CGG": "CGT", "AGA": "CGT", "AGG": "CGT",  # Arg → CGT
    "GCT": "GCG", "GCC": "GCG", "GCA": "GCG", "GCG": "GCG",   # Ala → GCG
    "CTT": "CTG", "CTC": "CTG", "CTA": "CTG", "TTA": "CTG", "TTG": "CTG", "CTG": "CTG",  # Leu → CTG
}

_BALANCED_SWAP = {
    "GCG": "GCC", "GCA": "GCT",
    "CTG": "CTT", "AGC": "TCT",
}


def _apply_codon_bias(seq: str, mode: str, rng: random.Random) -> str:
    """Apply lightweight codon swaps in the inferred CDS region only.

    'preferred'  → push toward most-frequent E. coli synonyms (raises CAI)
    'balanced'   → deliberately introduce some less-preferred synonyms (lowers CAI)
    'clean'      → remove BsaI/BsmBI/BbsI 6-mers from CDS via silent swaps
    """
    if mode not in ("preferred", "balanced", "clean"):
        return seq

    L = len(seq)
    cds_lo = max(1, round(0.29 * L) + 1)
    cds_hi = max(cds_lo + 3, round(0.86 * L))

    head = seq[:cds_lo - 1]
    cds = list(seq[cds_lo - 1:cds_hi])
    tail = seq[cds_hi:]

    swap_table = {}
    if mode == "preferred":
        swap_table = _PREFERRED_SWAP
    elif mode == "balanced":
        swap_table = _BALANCED_SWAP

    if swap_table:
        for i in range(0, len(cds) - 2, 3):
            codon = "".join(cds[i:i + 3])
            new = swap_table.get(codon, codon)
            if new != codon and rng.random() < 0.65:
                cds[i:i + 3] = list(new)

    if mode == "clean":
        forbidden = ("GGTCTC", "CGTCTC", "GAAGAC")  # BsaI, BsmBI, BbsI
        cds_str = "".join(cds)
        for motif in forbidden:
            while motif in cds_str:
                idx = cds_str.find(motif)
                # silent swap: bump the codon containing position `idx` to a synonym
                codon_idx = (idx // 3) * 3
                codon = cds_str[codon_idx:codon_idx + 3]
                alt = _silent_alt(codon)
                cds_str = cds_str[:codon_idx] + alt + cds_str[codon_idx + 3:]
                if alt == codon:
                    # last-ditch: punch a single base that breaks the motif
                    cds_str = cds_str[:idx] + _alt_base(cds_str[idx], rng) + cds_str[idx + 1:]
        cds = list(cds_str)

    return head + "".join(cds) + tail


_SILENT = {
    "GCT": "GCC", "GCC": "GCG", "GCA": "GCT", "GCG": "GCA",
    "CGT": "CGC", "CGC": "CGT",
    "GGT": "GGC", "GGC": "GGT",
    "CTG": "CTC", "CTC": "CTG",
    "GAA": "GAG", "GAG": "GAA",
    "TCT": "AGC", "AGC": "TCT",
}


def _silent_alt(codon: str) -> str:
    return _SILENT.get(codon, codon)


def _alt_base(b: str, rng: random.Random) -> str:
    pool = [x for x in "ACGT" if x != b]
    return rng.choice(pool)


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    bases = "ACGT"
    out = []
    for b in seq:
        if rng.random() < rate:
            out.append(rng.choice([x for x in bases if x != b]))
        else:
            out.append(b)
    return "".join(out)


class MockBackend:
    name = "mock"

    def generate(self, prompt: str, n: int = 4, sleep_s: float = 1.6) -> List[Candidate]:
        time.sleep(sleep_s)
        out: List[Candidate] = []
        for i in range(n):
            strat = DESIGN_STRATEGIES[i % len(DESIGN_STRATEGIES)]
            rng = random.Random(_seed_for(prompt, i))
            base = _apply_codon_bias(MOCK_SEQUENCE, strat["codon_bias"], rng)
            seq = _mutate(base, strat["mut_rate"], rng)
            out.append(Candidate(
                candidate_id=f"cand_{i}",
                thought=strat["thought"],
                sequence=seq,
                strategy=strat["label"],
                strategy_name=strat["name"],
            ))
        return out


# ---------------------------------------------------------------------------
# GGUF (fine-tuned Gemma 4) backend stub
# ---------------------------------------------------------------------------


class GGUFBackend:
    """Wraps a quantized Gemma 4 fine-tune via llama-cpp-python.

    Activated automatically when ``DGENE_GGUF_PATH`` env var points at a
    valid .gguf file. N candidates are generated by re-sampling at different
    temperatures + seeds — same prompt, different decodings.
    """

    name = "gemma-4-finetuned"

    def __init__(self, model_path: str):
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "GGUFBackend requires llama-cpp-python. Install with:\n"
                "  python3 -m pip install --upgrade llama-cpp-python\n"
                "Then restart the server."
            ) from exc

        self.model_path = model_path
        self._llm = Llama(
            model_path=model_path,
            n_ctx=int(os.environ.get("DGENE_GGUF_CTX", "4096")),
            n_gpu_layers=int(os.environ.get("DGENE_GGUF_GPU_LAYERS", "-1")),
            verbose=False,
        )

    def _format_prompt(self, user_prompt: str) -> str:
        # Mirrors the training format in gemma_train.jsonl: instruction + thought channel.
        return (
            "<|user|>\n"
            f"{user_prompt}\n"
            "<|assistant|>\n"
            "<|channel>thought\n"
        )

    def generate(self, prompt: str, n: int = 4) -> List[Candidate]:
        formatted = self._format_prompt(prompt)
        out: List[Candidate] = []
        # Temperature ladder for diversity across candidates.
        temps = [0.4, 0.7, 0.9, 1.1]
        for i in range(n):
            res = self._llm(
                formatted,
                max_tokens=int(os.environ.get("DGENE_GGUF_MAX_TOKENS", "1024")),
                temperature=temps[i % len(temps)],
                top_p=0.95,
                top_k=40,
                seed=_seed_for(prompt, i),
                stop=["</s>", "<|user|>"],
            )
            text = res["choices"][0]["text"]
            full = formatted + text
            try:
                thought, sequence = parse_thought_and_sequence(full)
            except ValueError:
                # Skip malformed sample but keep going for the others.
                continue
            out.append(Candidate(
                candidate_id=f"cand_{i}",
                thought=thought,
                sequence=sequence,
                strategy=f"sample_T{temps[i % len(temps)]}",
                strategy_name=f"Gemma sample (T={temps[i % len(temps)]})",
                raw=full,
            ))
        if not out:
            raise RuntimeError("All Gemma samples failed to parse — check prompt format.")
        return out


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def get_backend() -> "object":
    gguf_path = os.environ.get("DGENE_GGUF_PATH", "").strip()
    if gguf_path:
        if not os.path.isfile(gguf_path):
            print(
                f"[inference] DGENE_GGUF_PATH set but file not found: {gguf_path}\n"
                f"           falling back to MockBackend.",
                file=sys.stderr,
            )
        else:
            try:
                print(f"[inference] Loading GGUF backend from {gguf_path}", file=sys.stderr)
                return GGUFBackend(gguf_path)
            except Exception as exc:
                print(f"[inference] GGUFBackend failed: {exc}\n           falling back to MockBackend.",
                      file=sys.stderr)
    return MockBackend()


# ---------------------------------------------------------------------------
# Legacy single-shot API (kept for app.py / Streamlit demo)
# ---------------------------------------------------------------------------


def run_mock_inference(prompt: str) -> str:
    _ = prompt
    time.sleep(1.0)
    return (
        "<|channel>thought\n"
        "The circuit uses a constitutive promoter and tuned RBS to establish a stable "
        "transcription-translation baseline before the reporter CDS. The terminator "
        "must be placed downstream to prevent read-through and preserve modular behavior.\n"
        "<channel|>\n"
        f"{MOCK_SEQUENCE}"
    )
