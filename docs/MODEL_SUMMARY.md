# Model summary — OpenGeneEdit (DGENE)

This document follows the **Kaggle “Winning Model Documentation”** outline (sections A1–A9).  
OpenGeneEdit is a **natural-language → DNA compiler** with **retrieval-augmented** registry grounding, **deterministic synthesis** for boolean topology when applicable, **heuristic passes**, and **multi-objective ranking** — not a single tabular classifier. Where the template refers to “features” and “variable importance,” we interpret that as **design signals, pass metrics, and ranker objectives** (see A4, A7).

**Export:** for host submission in **Word/PDF**, paste this file or render Markdown to PDF; fill bracketed placeholders in **A1** and **A2**.

---

## A1. Team metadata (fill in)

| Field | Value |
|--------|--------|
| **Competition / program name** | *[e.g. host hackathon or competition name]* |
| **Team name** | *[Team name]* |
| **Private leaderboard score** | *[Metric and value, if applicable]* |
| **Private leaderboard place** | *[Rank]* |

**Team members** (repeat block per person):

| Name | Location | Email |
|------|----------|--------|
| *[Name]* | *[City, Country]* | *[email]* |
| … | … | … |

---

## A2. Background (per member; keep short if 3+)

*[For each team member, answer briefly:]*

- **Academic/professional background**
- **Prior experience that helped**
- **Why you entered**
- **Approximate time spent**
- **If a team: how you teamed up and who did what (roles)**

---

## A3. Summary (4–6 sentences)

The system combines **Google Gemma 4** (hosted **Generative Language API** and/or **local GGUF** via **llama-cpp-python** or **LM Studio HTTP**) with **ChromaDB** and **sentence-transformers** (`all-MiniLM-L6-v2`) over a filtered **iGEM parts JSONL** corpus. **Default compile mode (`circuit_synth`)** extracts a boolean **circuit intent**, runs **deterministic assembly and truth-table verification** for one topology-grounded candidate, and fills remaining diversity slots with **RAG-first** menu-constrained generation; **`rag_first`** skips topology proof; **`legacy`** uses channel-tagged generation plus **post-hoc RAG substitution**. **Quality** is enforced through **`passes.py`** (ORF, GC, repeats, Type IIS, CAI, RBS heuristics, etc.) and **`ranker.py`** (four Pareto objectives plus pipeline-tier and prompt-alignment ordering). **Optional** supervised JSONL for external **LoRA/SFT** is produced by **`scripts/generate_gemma_train.py`**; merged **GGUF** weights for local inference are distributed separately (see repository **README**). End-to-end **hosted** compiles are typically **tens of seconds to a few minutes** per multi-variant job depending on prompt, tool rounds, and API latency; **local 31B GGUF** on CPU can run **many minutes** per step.

---

## A4. “Features,” selection, and engineering

This pipeline is **not** an XGBoost-style model on a fixed columnar feature matrix. The closest analogues are:

| Signal / “feature” | Role |
|--------------------|------|
| **Natural-language prompt** | Drives intent JSON, topology extraction, and legacy channel output. |
| **Intent JSON fields** (`gate`, `input_analytes`, `reporter`, `retrieval_queries`, …) | Control **RAG-first** menus, slot-template cassettes, and compiler constraints. |
| **Embedding similarity** (Chroma cosine) | Gates which registry parts enter **legacy substitution** vs **RAG-first** menus (`DGENE_RAG_MIN_SIM`, stricter **`DGENE_RAG_MIN_SIM_PROMOTER`**). |
| **Pass metrics** (`cai`, `gc`, `rbs`, repeat / Type IIS diagnostics, …) | Feed **`ranker.score_candidate`**. |
| **`pipeline_tier`** | Orders candidates: verified topology path vs slot-template vs RAG-first vs legacy. |

**Variable importance (conceptual):** the **ranker composite** (see `ranker.WEIGHTS`) weights **expression** (CAI + RBS) highest, then **low_burden**, **gc_balance**, **cleanliness** (Type IIS pressure). **Pipeline tier** and **prompt token overlap** break ties before composite.

**Partial dependence (interpretation):** raising **GC** toward 50% improves **`gc_balance`**; stricter **Type IIS** cleanliness improves **`cleanliness`**; stronger **RBS/CAI** signals improve **`expression`** at possible **burden** cost.

**Transformations:** iGEM XML → normalized JSONL (**`scripts/extract_igem_dataset.py`**: type normalization, min length 40 bp, ACGT-only); optional **PhD-style rationale** targets in **`gemma_train.jsonl`**; DNA uppercase / whitespace stripping in parsers.

**Interactions:** **Promoter** slots use the **stricter** of global vs promoter similarity floors to reduce **pL\*** false positives; **NCBI Gene** may supply CDS when iGEM does not verify a symbol (**`ncbi_gene.py`**, env-gated).

**External data:** **iGEM registry** snapshot (and optional **NCBI Entrez**) where permitted by competition rules; **no** additional proprietary corpora are required for the default RAG index shipped as **`data/igem_dataset.jsonl`**.

---

## A5. Training method(s) and ensembling

- **Primary generative model:** **Gemma 4** instruction-tuned stack (**hosted API** or **quantized GGUF**).
- **Retrieval:** **sentence-transformers** + **Chroma** (persistent collection **`igem_parts`**).
- **Symbolic / deterministic layers:** **Boolean IR**, **`circuit_synth`**, **`circuit_verify`** (truth table vs regulatory graph simulation).
- **Ensembling:** **Multi-variant** compiles (`n` candidates) are **not** weighted model ensembles; they are **diverse samples** merged by **`ranker.rank`** (Pareto + ordering). **No** learned blend of heterogeneous base predictors.

---

## A6. Interesting findings

- **Topology-first slot** (when **`circuit_synth`** applies) yields a **verifiable** candidate distinct from “prompt-only” DNA — useful for reviewer trust even when other slots explore RAG-first diversity.
- **Mid-compile `search_igem_registry` tool loop** (hosted Gemini only) improves part grounding when the static menu is incomplete; **GGUF** paths skip tool rounds by design.
- **Legacy + post-hoc RAG** remains the closest match to the **channel-tagged fine-tune format** for local GGUF defaults.

---

## A7. Simple features / simple method (performance–simplicity tradeoff)

**Goal:** a deliberately **small** configuration approaching most practical utility.

| Item | Suggestion |
|------|------------|
| **Subset of “features” (≤10)** | (1) user **prompt** text, (2) **Chroma** top-1 **cosine** per substitution slot, (3–6) **`cai`**, **`gc`**, **`rbs`**, **repeat warn count**, (7) **Type IIS warn count**, (8) **sequence length**, (9) **`pipeline_tier`**, (10) **prompt_alignment** token overlap. |
| **Single method** | **`DGENE_COMPILE_MODE=legacy`** + **hosted Gemma** or **one GGUF** + **RAG on** (`DGENE_RAG=1`) + **default ranker** (no expert review). |
| **Expected behavior** | Loses **truth-table guarantees** and **RAG-first strict registry-only DNA** benefits; often **still usable** for ideation. **We do not quote a single numeric “90–95% leaderboard score”** here because this repository targets **design assistance**, not one Kaggle metric — ablate on your own held-out design rubric. |

---

## A8. Model execution time (order-of-magnitude; hardware-dependent)

| Stage | Typical order of magnitude |
|--------|-----------------------------|
| **Full LoRA/SFT + GGUF export** | **Hours to days** (external Unsloth/PEFT + conversion; not pinned in-repo). |
| **`generate_gemma_train.py` JSONL** | **Minutes to hours** (API rate limits, `--sample-size`). |
| **First Chroma index build** | **Minutes** on first import (embedding + disk persistence). |
| **Hosted multi-variant compile** | **~30s–5m+** per request (variants, streaming, tool rounds). |
| **Local 31B GGUF (CPU)** | **Many minutes** per variant; **GPU** reduces wall time subject to VRAM. |
| **Simplified legacy path** | Similar inference cost; **less** LLM JSON overhead than **`circuit_synth`**. |

**Prediction latency:** **`predict.py`** is **sequential** over prompts — total time ≈ sum of per-compile times.

---

## A9. References

- **Gemma / Gemini API:** Google Generative Language API documentation.  
- **llama.cpp / GGUF:** [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp), [abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python).  
- **Chroma:** [trychroma/chroma](https://github.com/chroma-core/chroma).  
- **Sentence-Transformers:** [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers).  
- **iGEM Registry** parts data (source per competition/host rules).  
- **Repository technical spec:** `docs/HACKATHON_TECHNICAL.md`, `docs/ARCHITECTURE.md`.

---

## B-series pointer (submission code bundle)

Reproduction paths, **`requirements.txt`**, **`directory_structure.txt`**, **`SETTINGS.json`**, **`entry_points.md`**, and **`README.md`** hardware/install instructions are maintained at the **repository root** for packaging as a **single zip** archive (exclude Kaggle-downloaded competition data if your host forbids redistribution).
