"""NCBI Entrez Gene → genomic CDS slice for bacterial genes.

Used when a slot is typed as a gene / CDS but iGEM RAG does not return a
verified hit (missing BBa, wrong organism, etc.). Fetches the annotated gene
locus from NCBI Gene, then pulls the corresponding genomic interval from
RefSeq nucleotide (bacterial genes are typically intronless).

Environment:

  • ``NCBI_API_KEY`` or ``DGENE_NCBI_API_KEY`` — optional; raises Entrez rate
    limit from 3 req/s to 10 req/s. Free registration:
    https://www.ncbi.nlm.nih.gov/account/settings/

  • ``DGENE_NCBI_EMAIL`` — optional but recommended (NCBI polite-use policy).

  • ``DGENE_NCBI=0`` — disable NCBI fallback entirely.

  • ``DGENE_NCBI_ORGANISMS`` — comma-separated scientific names to try in order
    (default: Pseudomonas aeruginosa, Escherichia coli).

  • ``DGENE_NCBI_PROMOTER_SLOTS`` — if ``1``, allow Entrez on **Promoter**-typed slots
    (default ``0``). Gene summaries return CDS / locus intervals, not short cis-regulatory
    promoter DNA, so the default avoids misleading “fixes” for promoter queries.

Cache: ``.chroma_igem/ncbi_gene_cache.json`` (same gitignored tree as Chroma).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CHROMA = os.path.join(_MODULE_DIR, ".chroma_igem")


def _chroma_base() -> str:
    return os.environ.get("DGENE_CHROMA_PATH", _DEFAULT_CHROMA).strip() or _DEFAULT_CHROMA


def _ncbi_enabled() -> bool:
    v = os.environ.get("DGENE_NCBI", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _api_key() -> str:
    return (os.environ.get("NCBI_API_KEY") or os.environ.get("DGENE_NCBI_API_KEY") or "").strip()


def _email() -> str:
    return (os.environ.get("DGENE_NCBI_EMAIL") or "").strip()


def _default_organisms() -> List[str]:
    raw = os.environ.get("DGENE_NCBI_ORGANISMS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["Pseudomonas aeruginosa", "Escherichia coli"]


def _organism_order_for_thought(thought: str) -> List[str]:
    base = list(_default_organisms())
    t = (thought or "").lower()
    if "pseudomonas" in t or "pyocyanin" in t or "phzr" in t or "phzi" in t:
        pa = "Pseudomonas aeruginosa"
        if pa in base:
            base.remove(pa)
        return [pa] + base
    if "e. coli" in t or "ecoli" in t or "escherichia coli" in t:
        ec = "Escherichia coli"
        if ec in base:
            base.remove(ec)
        return [ec] + base
    return base


_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = {}
_CACHE_LOADED = False
_CACHE_VERSION = 1
_RATE_LOCK = threading.Lock()
_LAST_REQ_MONO = 0.0


def _cache_path() -> str:
    return os.path.join(os.path.abspath(_chroma_base()), "ncbi_gene_cache.json")


def _rate_limit_delay() -> float:
    return 0.11 if _api_key() else 0.34


def _throttled_get(url: str) -> bytes:
    global _LAST_REQ_MONO
    delay = _rate_limit_delay()
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _LAST_REQ_MONO + delay - now
        if wait > 0:
            time.sleep(wait)
        _LAST_REQ_MONO = time.monotonic()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "DGene/1.0 (synthetic biology compiler; +https://github.com)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _load_cache() -> None:
    global _CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        path = _cache_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if (
                    isinstance(data, dict)
                    and data.get("version") == _CACHE_VERSION
                    and isinstance(data.get("entries"), dict)
                ):
                    _CACHE = dict(data["entries"])
            except (OSError, json.JSONDecodeError, ValueError):
                _CACHE = {}
        _CACHE_LOADED = True


def _save_cache() -> None:
    path = _cache_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _CACHE_LOCK:
            payload = {"version": _CACHE_VERSION, "entries": dict(_CACHE)}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError:
        pass


def _entrez_base_params() -> Dict[str, str]:
    p: Dict[str, str] = {"tool": "dgene"}
    em = _email()
    if em:
        p["email"] = em
    key = _api_key()
    if key:
        p["api_key"] = key
    return p


def _url(base: str, extra: Dict[str, str]) -> str:
    q = {**_entrez_base_params(), **extra}
    return f"{base}?{urllib.parse.urlencode(q)}"


def _esearch_gene(term: str) -> List[str]:
    url = _url(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        {"db": "gene", "term": term, "retmode": "json", "retmax": "5"},
    )
    raw = _throttled_get(url)
    data = json.loads(raw.decode("utf-8"))
    res = data.get("esearchresult") or {}
    return list(res.get("idlist") or [])


def _esummary_gene(gene_ids: List[str]) -> Dict[str, Any]:
    if not gene_ids:
        return {}
    url = _url(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
        {"db": "gene", "id": ",".join(gene_ids), "retmode": "json"},
    )
    raw = _throttled_get(url)
    return json.loads(raw.decode("utf-8"))


def _parse_genomic_interval(summary: Dict[str, Any], uid: str) -> Optional[Tuple[str, int, int, int]]:
    """Return (chr_accver, seq_from, seq_to, strand) where strand 1=plus, 2=minus."""

    rec = (summary.get("result") or {}).get(uid) or {}
    ginfo = rec.get("genomicinfo") or []
    if not ginfo:
        return None
    gi0 = ginfo[0]
    acc = str(gi0.get("chraccver") or "").strip()
    if not acc:
        return None
    try:
        a = int(gi0.get("chrstart"))
        b = int(gi0.get("chrstop"))
    except (TypeError, ValueError):
        return None
    lo, hi = (a, b) if a <= b else (b, a)
    strand = 1 if a <= b else 2
    return acc, lo, hi, strand


def _efetch_fasta_slice(acc: str, seq_from: int, seq_to: int, strand: int) -> str:
    url = _url(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        {
            "db": "nuccore",
            "id": acc,
            "seq_start": str(seq_from),
            "seq_stop": str(seq_to),
            "strand": str(strand),
            "rettype": "fasta",
            "retmode": "text",
        },
    )
    raw = _throttled_get(url).decode("utf-8", errors="replace")
    lines = raw.strip().splitlines()
    if not lines:
        return ""
    seq_lines = [ln.strip() for ln in lines[1:] if ln.strip()]
    return "".join(seq_lines).upper()


@dataclass
class NcbiGeneFetch:
    sequence: str
    gene_id: str
    gene_name: str
    organism: str
    accession: str
    seq_from: int
    seq_to: int
    strand: int


def fetch_gene_cds(
    gene_symbol: str,
    *,
    thought: str = "",
    log: Optional[Any] = None,
) -> Optional[NcbiGeneFetch]:
    """Resolve ``gene_symbol`` via NCBI Gene and return the genomic CDS interval FASTA.

    Returns ``None`` if disabled, cache-negative, or Entrez returns no usable locus.
    """

    def lg(msg: str) -> None:
        if log:
            try:
                log(msg)
            except Exception:
                pass

    if not _ncbi_enabled():
        return None
    sym = (gene_symbol or "").strip()
    if len(sym) < 2 or len(sym) > 24:
        return None

    _load_cache()
    organisms = _organism_order_for_thought(thought)
    cache_keys = [f"{sym.casefold()}|{org.casefold()}" for org in organisms]

    with _CACHE_LOCK:
        for ck in cache_keys:
            ent = _CACHE.get(ck)
            if ent is None:
                continue
            if ent == "MISS":
                lg(f"NCBI cache HIT (miss) {ck!r}")
                return None
            if isinstance(ent, dict) and ent.get("sequence"):
                lg(f"NCBI cache HIT (seq) {ck!r} bp={len(ent['sequence'])}")
                return NcbiGeneFetch(
                    sequence=str(ent["sequence"]),
                    gene_id=str(ent.get("gene_id", "")),
                    gene_name=str(ent.get("gene_name", sym)),
                    organism=str(ent.get("organism", "")),
                    accession=str(ent.get("accession", "")),
                    seq_from=int(ent.get("seq_from", 0)),
                    seq_to=int(ent.get("seq_to", 0)),
                    strand=int(ent.get("strand", 1)),
                )

    sym_esc = sym.replace('"', "")
    for org in organisms:
        ck = f"{sym.casefold()}|{org.casefold()}"
        term = f'{sym_esc}[Gene Name] AND "{org}"[Organism]'
        try:
            ids = _esearch_gene(term)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            lg(f"NCBI esearch FAILED symbol={sym!r} org={org!r}: {exc!s}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            return None
        if not ids:
            lg(f"NCBI esearch 0 hits symbol={sym!r} org={org!r}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            continue

        gid = ids[0]
        try:
            summ = _esummary_gene([gid])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            lg(f"NCBI esummary FAILED gene_id={gid}: {exc!s}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            continue

        interval = _parse_genomic_interval(summ, gid)
        if not interval:
            lg(f"NCBI esummary no genomicinfo gene_id={gid}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            continue

        acc, lo, hi, strand = interval
        rec = (summ.get("result") or {}).get(gid) or {}
        gname = str(rec.get("name") or sym)
        org_hit = ((rec.get("organism") or {}) or {}).get("scientificname") or org

        try:
            dna = _efetch_fasta_slice(acc, lo, hi, strand)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            lg(f"NCBI efetch FAILED acc={acc}: {exc!s}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            continue

        dna = "".join(dna.upper().split())
        if len(dna) < 30:
            lg(f"NCBI efetch too short ({len(dna)} bp) gene_id={gid}")
            with _CACHE_LOCK:
                _CACHE[ck] = "MISS"
            _save_cache()
            continue

        payload = {
            "sequence": dna,
            "gene_id": gid,
            "gene_name": gname,
            "organism": org_hit,
            "accession": acc,
            "seq_from": lo,
            "seq_to": hi,
            "strand": strand,
        }
        with _CACHE_LOCK:
            _CACHE[ck] = payload
        _save_cache()
        lg(
            f"NCBI OK symbol={sym!r} gene_id={gid} name={gname!r} org={org_hit!r} "
            f"{acc}:{lo}-{hi} strand={strand} bp={len(dna)}"
        )
        return NcbiGeneFetch(
            sequence=dna,
            gene_id=gid,
            gene_name=gname,
            organism=org_hit,
            accession=acc,
            seq_from=lo,
            seq_to=hi,
            strand=strand,
        )

    return None


_BBA_HEAD = re.compile(r"^BBa_", re.I)
_B_NUM = re.compile(r"^B\d{4}$", re.I)
_J_NUM = re.compile(r"^J\d{5}$", re.I)


def _ncbi_promoter_slots_enabled() -> bool:
    """Gene fetch returns CDS / gene locus — not cis-regulatory promoter DNA.

    Default **off** so a "PldhA promoter" slot does not silently pull an unrelated
    genomic interval. Enable with ``DGENE_NCBI_PROMOTER_SLOTS=1`` when you rely on
    mis-typed regulator names in promoter slots."""
    v = os.environ.get("DGENE_NCBI_PROMOTER_SLOTS", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


def gene_symbol_eligible_for_ncbi(query_text: str, type_hint: Optional[str]) -> Optional[str]:
    """Extract a single gene symbol from an enriched RAG query, or ``None`` if NCBI should not run."""

    if type_hint in ("RBS", "Terminator"):
        return None
    if (type_hint or "").strip() == "Promoter" and not _ncbi_promoter_slots_enabled():
        return None
    q = (query_text or "").strip()
    if not q:
        return None
    th = (type_hint or "").strip()
    if th and len(q) > len(th) + 1 and q.endswith(f" {th}"):
        core = q[: -(len(th) + 1)].strip()
    else:
        core = q
    if not core:
        return None
    if " " in core:
        core = core.split()[0].strip()
    if len(core) < 2 or len(core) > 24:
        return None
    if _BBA_HEAD.match(core) or _B_NUM.match(core) or _J_NUM.match(core):
        return None
    if not re.match(r"^[A-Za-z][A-Za-z0-9\-]{1,23}$", core):
        return None
    return core
