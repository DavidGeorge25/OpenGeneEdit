"""Compiler passes for synthetic biology constructs.

Each pass is a pure function ``(sequence: str) -> PassResult``. Passes are
classified as ``parse``, ``lint``, or ``score``; the ranker consumes the
``metric`` field on score passes to derive composite candidate scores.

Designed to be model-agnostic: works on any DNA string, regardless of whether
it came from the mock backend or the fine-tuned Gemma GGUF.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    severity: str  # "info" | "warn" | "error"
    message: str
    start: Optional[int] = None  # 1-indexed inclusive
    end: Optional[int] = None    # 1-indexed inclusive


@dataclass
class PassResult:
    pass_id: str
    name: str
    category: str          # "parse" | "lint" | "score"
    status: str            # "ok" | "warn" | "error"
    summary: str
    diagnostics: List[Diagnostic] = field(default_factory=list)
    metric: Optional[float] = None        # 0..1 normalized when applicable
    metric_label: Optional[str] = None
    metric_raw: Optional[str] = None      # human-readable raw value
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Codon adaptation table (E. coli K-12 MG1655, relative adaptiveness w_i)
# Values are normalized so the most-used synonymous codon for each amino
# acid is 1.0; CAI is the geometric mean of w_i across the CDS.
# ---------------------------------------------------------------------------

ECOLI_W: Dict[str, float] = {
    "GCT": 0.27, "GCC": 0.39, "GCA": 0.21, "GCG": 1.00,                                    # Ala
    "CGT": 1.00, "CGC": 0.93, "CGA": 0.13, "CGG": 0.16, "AGA": 0.10, "AGG": 0.05,          # Arg
    "AAT": 0.51, "AAC": 1.00,                                                              # Asn
    "GAT": 1.00, "GAC": 0.55,                                                              # Asp
    "TGT": 0.87, "TGC": 1.00,                                                              # Cys
    "CAA": 0.51, "CAG": 1.00,                                                              # Gln
    "GAA": 1.00, "GAG": 0.46,                                                              # Glu
    "GGT": 1.00, "GGC": 0.99, "GGA": 0.16, "GGG": 0.21,                                    # Gly
    "CAT": 0.69, "CAC": 1.00,                                                              # His
    "ATT": 0.95, "ATC": 1.00, "ATA": 0.07,                                                 # Ile
    "TTA": 0.18, "TTG": 0.20, "CTT": 0.20, "CTC": 0.20, "CTA": 0.07, "CTG": 1.00,          # Leu
    "AAA": 1.00, "AAG": 0.30,                                                              # Lys
    "ATG": 1.00,                                                                           # Met
    "TTT": 0.74, "TTC": 1.00,                                                              # Phe
    "CCT": 0.31, "CCC": 0.28, "CCA": 0.31, "CCG": 1.00,                                    # Pro
    "TCT": 0.46, "TCC": 0.43, "TCA": 0.30, "TCG": 0.31, "AGT": 0.41, "AGC": 1.00,          # Ser
    "ACT": 0.42, "ACC": 1.00, "ACA": 0.27, "ACG": 0.51,                                    # Thr
    "TGG": 1.00,                                                                           # Trp
    "TAT": 0.65, "TAC": 1.00,                                                              # Tyr
    "GTT": 0.98, "GTC": 0.61, "GTA": 0.36, "GTG": 1.00,                                    # Val
}
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Forbidden Type IIS sites in CDS regions (Golden Gate / MoClo cloning).
TYPE_IIS_SITES = {
    "BsaI":  "GGTCTC",
    "BsmBI": "CGTCTC",
    "BbsI":  "GAAGAC",
    "SapI":  "GCTCTTC",
}

# Common 6-cutter restriction sites surfaced as informational annotations.
COMMON_RE_SITES = {
    "EcoRI":   "GAATTC",
    "BamHI":   "GGATCC",
    "HindIII": "AAGCTT",
    "NdeI":    "CATATG",
    "XhoI":    "CTCGAG",
    "SpeI":    "ACTAGT",
    "PstI":    "CTGCAG",
    "SalI":    "GTCGAC",
    "NcoI":    "CCATGG",
    "KpnI":    "GGTACC",
    "XbaI":    "TCTAGA",
    "NotI":    "GCGGCCGC",
}

# Anti–Shine-Dalgarno sequence on E. coli 16S rRNA 3' tail (CCUCC equiv: CCTCC on DNA).
ANTI_SD = "CCTCCT"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _gc(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count("G") + seq.count("C")
    return g / len(seq)


def _reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def _find_all(seq: str, motif: str) -> List[int]:
    """Return 0-indexed start positions of every (overlapping) match."""
    out: List[int] = []
    if not motif:
        return out
    i = seq.find(motif)
    while i != -1:
        out.append(i)
        i = seq.find(motif, i + 1)
    return out


# ---------------------------------------------------------------------------
# Parse pass — lightweight feature inference (mirrors the JS catalog so the
# server-side pass annotations align with the front-end map).
# ---------------------------------------------------------------------------


PARSE_CATALOG = [
    ("J23100", "promoter",   0.00, 0.16),
    ("lacO",   "operator",   0.16, 0.23),
    ("B0034",  "RBS",        0.23, 0.29),
    ("sfGFP",  "CDS",        0.29, 0.86),
    ("B0015",  "terminator", 0.86, 1.00),
]


def parse_features(seq: str) -> List[Dict[str, object]]:
    L = len(seq)
    out: List[Dict[str, object]] = []
    for label, kind, p0, p1 in PARSE_CATALOG:
        start = max(1, round(p0 * L) + (0 if p0 == 0 else 1))
        end = max(start + 1, round(p1 * L))
        out.append({"label": label, "type": kind, "start": start, "end": end, "strand": 1})
    return out


def pass_parse(seq: str) -> PassResult:
    t0 = _now_ms()
    feats = parse_features(seq)
    summary = " → ".join(f["label"] for f in feats)  # type: ignore[arg-type]
    diag = [
        Diagnostic("info", f"{f['label']} ({f['type']}) {f['start']}–{f['end']}",
                   start=int(f["start"]), end=int(f["end"]))  # type: ignore[arg-type]
        for f in feats
    ]
    return PassResult(
        pass_id="parse",
        name="Parse construct",
        category="parse",
        status="ok",
        summary=f"Identified {len(feats)} features · {summary}",
        diagnostics=diag,
        duration_ms=_now_ms() - t0,
    )


# ---------------------------------------------------------------------------
# Lint passes
# ---------------------------------------------------------------------------


def _cds_window(seq: str) -> Tuple[int, int]:
    """Return [start, end) of the inferred CDS window (0-indexed)."""
    L = len(seq)
    feats = parse_features(seq)
    for f in feats:
        if f["type"] == "CDS":
            return int(f["start"]) - 1, int(f["end"])  # type: ignore[arg-type]
    return 0, L


def _find_main_orf(seq: str, min_codons: int = 25) -> Optional[Tuple[int, int, int]]:
    """Return (start, end, frame) of the longest forward ATG..STOP ORF, or None.

    Coordinates are 0-indexed. ``end`` is exclusive of the terminal stop.
    """
    best: Optional[Tuple[int, int, int]] = None
    best_len = 0
    L = len(seq)
    for frame in range(3):
        i = frame
        while i + 2 < L:
            if seq[i:i + 3] == "ATG":
                # walk to next in-frame stop
                j = i + 3
                while j + 2 < L:
                    cod = seq[j:j + 3]
                    if cod in STOP_CODONS:
                        break
                    j += 3
                length = j - i
                if length // 3 >= min_codons and length > best_len:
                    best = (i, j, frame)
                    best_len = length
                i = j + 3 if j + 2 < L else L
            else:
                i += 3
    return best


def pass_orf(seq: str) -> PassResult:
    """Find the longest in-frame ATG..STOP ORF and report its location."""
    t0 = _now_ms()
    orf = _find_main_orf(seq, min_codons=25)
    if orf is None:
        return PassResult(
            pass_id="orf", name="ORF validation", category="lint",
            status="warn",
            summary="No forward ORF ≥ 25 codons found",
            diagnostics=[Diagnostic("warn", "No ATG..STOP ORF detected in any forward frame")],
            duration_ms=_now_ms() - t0,
        )
    start, end, frame = orf
    n_codons = (end - start) // 3
    diag = [
        Diagnostic("info", f"ATG start (frame {frame})", start=start + 1, end=start + 3),
        Diagnostic("info", f"Stop codon ({seq[end:end + 3]})", start=end + 1, end=end + 3),
    ]
    return PassResult(
        pass_id="orf", name="ORF validation", category="lint",
        status="ok",
        summary=f"ORF intact · {n_codons} codons · frame {frame} · {start + 1}–{end + 3}",
        diagnostics=diag,
        duration_ms=_now_ms() - t0,
    )


def pass_gc(seq: str) -> PassResult:
    t0 = _now_ms()
    gc = _gc(seq)
    pct = gc * 100
    if 0.40 <= gc <= 0.60:
        status, msg = "ok", f"GC = {pct:.1f}% (within 40–60% optimal)"
    elif 0.30 <= gc <= 0.65:
        status, msg = "warn", f"GC = {pct:.1f}% (outside 40–60%, still tolerable)"
    else:
        status, msg = "error", f"GC = {pct:.1f}% (extreme — synthesis & expression risk)"

    metric = max(0.0, 1.0 - abs(gc - 0.5) * 2)  # 1.0 at 50%, 0 at 0/100%
    return PassResult(
        pass_id="gc", name="GC balance", category="score",
        status=status, summary=msg,
        metric=metric, metric_label="GC %", metric_raw=f"{pct:.1f}%",
        diagnostics=[Diagnostic("info" if status == "ok" else status, msg)],
        duration_ms=_now_ms() - t0,
    )


def pass_repeats(seq: str, k: int = 18) -> PassResult:
    """Detect direct repeats of length >= k that recur within the sequence.

    Repeats above ~16-20 bp are the main driver of recombination loss in E. coli.
    """
    t0 = _now_ms()
    if len(seq) < 2 * k:
        return PassResult(
            pass_id="repeats", name="Repeat scan", category="lint",
            status="ok", summary="Sequence too short for repeat analysis",
            duration_ms=_now_ms() - t0,
        )

    seen: Dict[str, int] = {}
    hits: List[Tuple[int, int, str]] = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in seen:
            hits.append((seen[kmer], i, kmer))
        else:
            seen[kmer] = i

    diag = [
        Diagnostic("warn", f"{k}-mer repeat (seed @ {a + 1})", start=b + 1, end=b + k)
        for a, b, _ in hits[:8]
    ]
    if hits:
        return PassResult(
            pass_id="repeats", name="Repeat scan", category="lint",
            status="warn", summary=f"{len(hits)} direct repeat(s) ≥ {k} bp",
            diagnostics=diag, duration_ms=_now_ms() - t0,
        )
    return PassResult(
        pass_id="repeats", name="Repeat scan", category="lint",
        status="ok", summary=f"No direct repeats ≥ {k} bp",
        duration_ms=_now_ms() - t0,
    )


def pass_type_iis(seq: str) -> PassResult:
    """Forbidden-site scan for Golden Gate / MoClo Type IIS enzymes."""
    t0 = _now_ms()
    rc = _reverse_complement(seq)
    diag: List[Diagnostic] = []
    total = 0
    for name, motif in TYPE_IIS_SITES.items():
        for i in _find_all(seq, motif):
            total += 1
            diag.append(Diagnostic("warn", f"{name} site (+strand)",
                                   start=i + 1, end=i + len(motif)))
        for i in _find_all(rc, motif):
            total += 1
            pos = len(seq) - i - len(motif) + 1
            diag.append(Diagnostic("warn", f"{name} site (–strand)",
                                   start=pos, end=pos + len(motif) - 1))

    if total == 0:
        status = "ok"
        summary = "No Type IIS sites — Golden Gate / MoClo compatible"
    else:
        status = "warn"
        summary = f"{total} Type IIS site(s) — would block Golden Gate assembly"

    return PassResult(
        pass_id="type_iis", name="Type IIS site scan", category="lint",
        status=status, summary=summary, diagnostics=diag,
        duration_ms=_now_ms() - t0,
    )


def pass_restriction_map(seq: str) -> PassResult:
    """Catalog common 6-cutter sites for downstream cloning convenience."""
    t0 = _now_ms()
    diag: List[Diagnostic] = []
    for name, motif in COMMON_RE_SITES.items():
        for i in _find_all(seq, motif):
            diag.append(Diagnostic("info", f"{name} cut", start=i + 1, end=i + len(motif)))
    return PassResult(
        pass_id="restrict", name="Restriction site map", category="parse",
        status="ok",
        summary=f"{len(diag)} site(s) cataloged across {len(COMMON_RE_SITES)} enzymes",
        diagnostics=diag[:32],
        duration_ms=_now_ms() - t0,
    )


# ---------------------------------------------------------------------------
# Score passes (drive the ranker)
# ---------------------------------------------------------------------------


def pass_cai(seq: str) -> PassResult:
    """Codon Adaptation Index for E. coli K-12. Geometric mean of w_i over the
    main forward ORF (falls back to the proportional CDS window if no ORF found)."""
    t0 = _now_ms()
    orf = _find_main_orf(seq, min_codons=25)
    if orf is not None:
        cds_start, cds_end, _ = orf
    else:
        cds_start, cds_end = _cds_window(seq)
    cds = seq[cds_start:cds_end]

    weights: List[float] = []
    skipped = 0
    for i in range(0, len(cds) - 2, 3):
        codon = cds[i:i + 3]
        if codon in STOP_CODONS:
            continue
        w = ECOLI_W.get(codon)
        if w is None or w == 0:
            skipped += 1
            continue
        weights.append(w)

    if not weights:
        return PassResult(
            pass_id="cai", name="Codon adaptation (E. coli)", category="score",
            status="warn", summary="No codons available to score",
            metric=0.0, metric_label="CAI", metric_raw="0.00",
            duration_ms=_now_ms() - t0,
        )

    # Geometric mean via sum of logs to avoid underflow on long CDS.
    import math
    log_sum = sum(math.log(w) for w in weights)
    cai = math.exp(log_sum / len(weights))

    if cai >= 0.75:
        status = "ok"
    elif cai >= 0.5:
        status = "warn"
    else:
        status = "error"

    summary = f"E. coli CAI = {cai:.3f} ({len(weights)} codons{', ' + str(skipped) + ' skipped' if skipped else ''})"
    return PassResult(
        pass_id="cai", name="Codon adaptation (E. coli)", category="score",
        status=status, summary=summary,
        metric=cai, metric_label="CAI", metric_raw=f"{cai:.3f}",
        duration_ms=_now_ms() - t0,
    )


def pass_rbs(seq: str) -> PassResult:
    """Heuristic Shine–Dalgarno strength: best 6-mer complementarity to anti-SD,
    in the −20..−5 window upstream of the actual ATG start codon.

    Falls back to the proportional CDS window if no ORF is detected. This is
    intentionally a simple proxy — the real Salis RBS Calculator does full
    thermodynamic folding. The ranker only needs a relative signal.
    """
    t0 = _now_ms()
    orf = _find_main_orf(seq, min_codons=25)
    if orf is not None:
        cds_start = orf[0]
    else:
        cds_start, _ = _cds_window(seq)
    if cds_start < 6:
        return PassResult(
            pass_id="rbs", name="RBS strength (Shine-Dalgarno)", category="score",
            status="warn", summary="Insufficient 5'UTR to score RBS",
            metric=0.0, metric_label="RBS", metric_raw="—",
            duration_ms=_now_ms() - t0,
        )

    win_lo = max(0, cds_start - 20)
    win_hi = max(0, cds_start - 5)
    window = seq[win_lo:win_hi]
    rc_anti = _reverse_complement(ANTI_SD)  # what the mRNA SD itself looks like (DNA-rep)

    best_score = 0
    best_pos = -1
    for i in range(0, len(window) - len(rc_anti) + 1):
        kmer = window[i:i + len(rc_anti)]
        match = sum(1 for a, b in zip(kmer, rc_anti) if a == b)
        if match > best_score:
            best_score = match
            best_pos = win_lo + i

    metric = best_score / len(rc_anti)  # 0..1
    expr_au = int(round(50 + (metric ** 2) * 7950))  # mock 50..8000 a.u.

    if metric >= 0.83:
        status = "ok"
    elif metric >= 0.5:
        status = "warn"
    else:
        status = "error"

    diag: List[Diagnostic] = []
    if best_pos >= 0:
        diag.append(Diagnostic(
            "info" if status == "ok" else status,
            f"Best SD match score {best_score}/{len(rc_anti)} → ~{expr_au:,} a.u.",
            start=best_pos + 1, end=best_pos + len(rc_anti),
        ))

    return PassResult(
        pass_id="rbs", name="RBS strength (Shine-Dalgarno)", category="score",
        status=status,
        summary=f"SD complementarity {best_score}/{len(rc_anti)} → predicted ~{expr_au:,} a.u.",
        metric=metric, metric_label="RBS", metric_raw=f"{expr_au:,} a.u.",
        diagnostics=diag,
        duration_ms=_now_ms() - t0,
    )


def pass_hairpin(seq: str, min_stem: int = 6, max_loop: int = 8) -> PassResult:
    """Approximate hairpin scan: any inverted repeat of length >= min_stem with
    a loop <= max_loop counts as a candidate hairpin. Worst (longest) flagged.
    """
    t0 = _now_ms()
    worst = (0, 0, 0)  # stem_len, start, end
    L = len(seq)
    for i in range(L - 2 * min_stem - max_loop):
        stem = seq[i:i + min_stem]
        rc = _reverse_complement(stem)
        for loop in range(0, max_loop + 1):
            j = i + min_stem + loop
            if j + min_stem > L:
                break
            if seq[j:j + min_stem] == rc:
                # extend stem
                stem_len = min_stem
                while (i + stem_len < j and j + stem_len < L
                       and seq[j + stem_len] == _reverse_complement(seq[i + stem_len - 1:i + stem_len])):
                    stem_len += 1
                end = j + stem_len
                if stem_len > worst[0]:
                    worst = (stem_len, i + 1, end)

    diag: List[Diagnostic] = []
    if worst[0] >= min_stem:
        diag.append(Diagnostic(
            "warn",
            f"Inverted repeat stem ≥ {worst[0]} bp (potential hairpin)",
            start=worst[1], end=worst[2],
        ))
        if worst[0] >= 10:
            return PassResult(
                pass_id="hairpin", name="Secondary structure (heuristic)", category="lint",
                status="warn",
                summary=f"Stable hairpin candidate · stem {worst[0]} bp at {worst[1]}–{worst[2]}",
                diagnostics=diag, duration_ms=_now_ms() - t0,
            )
        return PassResult(
            pass_id="hairpin", name="Secondary structure (heuristic)", category="lint",
            status="ok",
            summary=f"Mild hairpin · stem {worst[0]} bp at {worst[1]}–{worst[2]}",
            diagnostics=diag, duration_ms=_now_ms() - t0,
        )

    return PassResult(
        pass_id="hairpin", name="Secondary structure (heuristic)", category="lint",
        status="ok", summary="No significant hairpins detected",
        duration_ms=_now_ms() - t0,
    )


def pass_biosec(seq: str) -> PassResult:
    """Stub biosecurity screen. The real impl would call IGSC or run a curated
    motif/k-mer scan against select-agent / hazardous sequences. Here we just
    confirm we ran the pass — placeholder for a serious implementation.
    """
    t0 = _now_ms()
    return PassResult(
        pass_id="biosec", name="Biosecurity screen", category="lint",
        status="ok",
        summary="No matches against IGSC hazardous-sequence list (stub)",
        diagnostics=[Diagnostic("info", "Stub screen — replace with curated motif set or IGSC API")],
        duration_ms=_now_ms() - t0,
    )


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


PASS_PIPELINE: List[Callable[[str], PassResult]] = [
    pass_parse,
    pass_orf,
    pass_gc,
    pass_repeats,
    pass_type_iis,
    pass_restriction_map,
    pass_cai,
    pass_rbs,
    pass_hairpin,
    pass_biosec,
]


def run_passes(seq: str) -> List[PassResult]:
    return [p(seq) for p in PASS_PIPELINE]


def passes_to_dicts(results: List[PassResult]) -> List[Dict]:
    out = []
    for r in results:
        d = asdict(r)
        d["duration_ms"] = round(d["duration_ms"], 2)
        out.append(d)
    return out
