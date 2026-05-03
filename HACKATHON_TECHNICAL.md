# DGene — Technical overview (Google hackathon)

**DGene** is a synthetic-biology–oriented DNA “compiler.” A user describes a genetic circuit in natural language; the system returns structured reasoning, a nucleotide sequence, and—depending on **`DGENE_COMPILE_MODE`**—either **topology-verified plasmids** built from a curated catalog, **RAG-first** constructs assembled only from **menu-retrieved registry DNA**, or **legacy** channel-tagged model DNA with **post-hoc** iGEM/NCBI slot substitution. Downstream: heuristic **compiler-style passes**, **multi-objective ranking** (pipeline tier + prompt fit + Pareto objectives), optional **design QA** (deterministic regulator lint + optional Gemma “PI review”), plasmid visualization, **FASTA / GenBank** export, and **bookmarkable compile snapshots**. Inference is **Google Gemma 4 only**: either via the **Gemini API** (hosted) or a local **GGUF** checkpoint loaded with `llama-cpp-python`.

**Two model modes in practice**

- **Stock Gemma 4 (hosted)** — The default path for most users is **standard instruction-tuned Gemma 4** on the Gemini API (e.g. `gemma-4-31b-it` via `DGENE_GEMINI_MODEL`). No local GPU required; the same channel-tagged prompt and parser are used. Quality follows the base model plus prompting.
- **Fine-tuned Gemma 4 (local GGUF)** — The hackathon build also supports a **domain-specific fine-tune** trained on compiler-shaped examples derived from the iGEM parts corpus (see §3). That checkpoint is distributed as a **quantized `.gguf`** for self-hosted inference through `llama-cpp-python`. The app does not bundle the file; you download it (e.g. from Hugging Face) and point `DGENE_GGUF_PATH` at it.

This document walks through data acquisition, cleaning, model alignment / local weights, Hugging Face hosting, live demo, self-hosting, and how **compile modes** (**topology + hybrid padding**, **RAG-first**, **legacy**), **Chroma RAG**, **design QA**, snapshots, and ranking interact in **`server.py`** vs **`app.py`**.

---

## Live demo

**Demo URL (paste your deployment link on the next line; leave blank until ready):**


_________________________________________________

**How to use it**

1. Open the URL in a browser (Chrome or Safari is fine).
2. Enter a **natural-language genetic circuit brief** (e.g. inducible biosensor, promoter/RBS/CDS/terminator choices, organism constraints).
3. Submit and wait for the compile job to finish (the full web UI may show a **progress** panel during long hosted-inference runs).
4. Inspect **reasoning**, the **circular plasmid map**, **per-candidate scores and Pareto highlights**, and **RAG audit** (which slots got registry DNA). Download **FASTA** or **GenBank** if exposed in the UI.

If the demo is **Streamlit-only**, the same flow applies: one prompt → one sequence + map + exports. If the demo is the **full compiler server**, you get multi-candidate generation, ranking, and the richer artifact pane.

---

## 1. High-level architecture

**Default web path (`DGENE_COMPILE_MODE=circuit_synth`, requires hosted Gemma for intent + RAG-first padding):**

```
User prompt
    → Boolean intent (Gemma JSON) — circuit_intent.extract_circuit_spec
        → If applicable: CircuitSpec (circuit_ir) → circuit_synth → circuit_verify (truth table)
              → Candidate #1: full linear plasmid (ori + CmR + MCS + cassettes); rag.pipeline = circuit_synth
        → Else: skip verified topology for this slot
    → Remaining candidates: RAG-first (shared intent JSON + shared Chroma menu)
        → Optional first variant: slot-template cassette (§5.9) when gate/analytes/reporter parse
        → Other variants: Gemma “compiler” proposes ORDERED_PART_LIST (temperature ladder) → JSONL concat only
    → Per candidate: post-hoc igem_rag.apply_rag_substitution ONLY on legacy channel-DNA rows
    → design_expert_lint + optional expert_review (ordered BBa audit)
    → passes.py → ranker (pipeline_tier → prompt_alignment → composite; Pareto on four sequence objectives)
    → UI / snapshots
```

**Legacy path (`DGENE_COMPILE_MODE=legacy`, or API-key fallback):**

```
User prompt → Gemma generates <thought channel> + DNA + </circuit>
    → parse_thought_and_sequence
    → apply_rag_substitution (equal-chunk slots + Chroma + optional NCBI)
    → passes → ranker → UI
```

If `circuit_synth` or `rag_first` is selected but **no** `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set, **`server.py` falls back to legacy** inference+RAG and logs a warning — topology and menu compilers both need the hosted API today.

**Core modules**

| Module | Role |
|--------|------|
| `inference.py` | Backend selection, Gemma prompting, parsing `<|channel>thought` / `<channel|>` / DNA / `</circuit>`; loads `.env` on import |
| `igem_rag.py` | JSONL → ChromaDB index; retrieval; **slot-based** merge (legacy); registry token vocabulary; NCBI handoff; RAG-first menu retrieval |
| `ncbi_gene.py` | NCBI Entrez (Gene → genomic slice) for bacterial loci when iGEM does not verify a slot |
| `passes.py` | ORF, GC, repeats, Type IIS, restriction map, CAI, RBS heuristic, hairpins, etc. |
| `ranker.py` | Sequence objectives + Pareto front; **`best_id` sort** uses `pipeline_tier` (**`circuit_synth` (3) > `slot_template` (2) > `rag_first` (1) > legacy (0)**) then **`prompt_alignment`** then **composite** |
| `server.py` | `ThreadingHTTPServer`; `/api/compile`, `/api/compile/status`, `/api/fix`, `/api/snapshot`, `/api/health`; static `web/`; optional `.design_snapshots/` persistence |
| `app.py` | Streamlit single-shot UI + Bokeh plasmid map |
| `extract_igem_dataset.py` | Build `igem_dataset.jsonl` from `xml_parts.xml.gz` |
| `generate_gemma_train.py` | Build supervised `gemma_train.jsonl` for fine-tuning using hosted Gemma 4 |
| `circuit_ir.py` | Typed intermediate representation: `CircuitSpec`, `LogicSpec`, structured inputs/output, **executable truth tables** for verification |
| `circuit_parts.py` | Curated promoter–TF wiring, backbone reference DNA, inducer/reporter catalogs consumed by `circuit_synth` |
| `circuit_intent.py` | Gemma → strict JSON (`applicable`, inputs, logic, reporter); builds `CircuitSpec` when the brief maps to supported gates |
| `circuit_synth.py` | Deterministic topology + sequences from `igem_dataset.jsonl` / catalog BBa IDs |
| `circuit_verify.py` | Regulatory-graph fixpoint vs requested truth table; failure ⇒ no verified candidate |
| `circuit_pipeline.py` | **`compile_hybrid_variants` / `_iter`**: up to one verified `circuit_synth` candidate, then `run_rag_first_*` for diversity |
| `circuit_rag_first.py` | Intent JSON → flatten retrieval queries → **`build_part_menu`** → **`run_compiler`** (`<reasoning>` + `ORDERED_PART_LIST` + `</circuit_design>`) → **`parse_ordered_bba`** (never scans free-form CoT for BBa) → **`assemble_sequence`** |
| `slot_template_compile.py` | Deterministic Promoter/RBS/CDS/Terminator cassette from per-type retrieval when intent exposes `gate` / `input_analytes` / `reporter`; optional backbone embed (§5.9) |
| `design_expert_lint.py` | Rule-based check: regulated promoters in the ordered list must include cognate regulator CDS (`circuit_parts` PROMOTERS/TFS); emits `rag.expert_lint` |
| `expert_review.py` | Optional hosted Gemma JSON “PI review” of brief + ordered BBa list (`DGENE_EXPERT_REVIEW`) → `rag.expert_review` |

---

## 2. iGEM registry → `igem_dataset.jsonl`

**Source.** The project expects a gzip’d XML export of the iGEM parts registry (`xml_parts.xml.gz` at repo root). This is **not** fetched by code in-repo; you obtain the dump and place it next to the extractor.

**Extraction — `extract_igem_dataset.py`**

1. **Streaming gzip parse** with a tolerant reader (`iter_lines_with_tolerant_gzip`) so truncated/junk after valid gzip does not abort the whole file.
2. **Locate the `parts` table** via `<table_data name="parts">` and parse rows with regex-driven field extraction (`FIELD_RE`), HTML-unescaping values.
3. **Normalize part type** (`normalize_part_type`) into a small closed set used downstream:
   - `Promoter` (includes `regulatory`, `generator`, and category hints)
   - `RBS`, `CDS`, `Terminator`
   - Rows that do not map to these are **dropped**.
4. **Sequence cleaning** (`clean_sequence`): strip non-letters, uppercase.
5. **Quality filters** (rows must pass all):
   - Sequence length ≥ **40** bp
   - No **`N`** bases
   - **[ACGT]** only (no ambiguous IUPAC)

**Output record** (one JSON object per line in `igem_dataset.jsonl`):

- `part_id`, `part_name`, `part_type`, `short_desc`, `sequence`

This JSONL is the canonical corpus for RAG and for training-data sampling.

---

## 3. Training data for Gemma 4 — `generate_gemma_train.py`

The repo does **not** embed the full command-line for LoRA/SFT merge/GGUF conversion; it **does** automate a critical step: **curated supervised examples** in the same textual format the runtime parser expects.

**Flow**

1. **Load** `igem_dataset.jsonl` (or path from `--input`).
2. **Filter** to `TARGET_TYPES`: Promoter, RBS, CDS, Terminator; validate sequences (same spirit as extraction: length, no `N`, ACGT-only).
3. **Sample** a balanced set across the four types (`--sample-size` must be divisible by 4 for equal per-type counts; `--seed` for reproducibility).
4. For each sampled part, call **hosted Gemma 4** via `inference.generate_text_gemma4` with a **PhD synthetic biologist** system prompt asking for ~2 sentences of biochemical / design reasoning (with retries on rate limits).
5. **Write** `gemma_train.jsonl` rows as chat-style `conversations`:

   - **Human:** `Design a {part_type} for the following purpose: {short_desc}`
   - **Assistant (gpt):**  
     `<|channel>thought\n{model_reasoning}\n<channel|>\n{exact_registry_sequence}`

That assistant string matches the **channel tagging** that `inference.parse_thought_and_sequence` consumes at compile time.

### What the fine-tune is trained on

- **Supervision source:** Rows sampled from **`igem_dataset.jsonl`** (cleaned iGEM registry parts: Promoter, RBS, CDS, Terminator) with valid sequences.
- **Input (human turn):** A design brief per part, e.g. “Design a `{part_type}` for the following purpose: `{short_desc}`”.
- **Target (assistant turn):** Model-generated **PhD-style reasoning** (two short sentences from hosted Gemma 4 during dataset build) **plus the exact registry DNA** for that part, wrapped in **`<|channel>thought … <channel|> …`** so it matches the compiler’s runtime parser.
- **Intent:** Nudge the model toward **coherent synthetic-biology prose naming real parts** and **emitting long DNA in the enforced format**, so downstream **RAG** can align slots to registry entries using the thought channel.

### From fine-tuned weights to GGUF

This repository **stops at `gemma_train.jsonl`** for training data. The authors then run a **separate fine-tuning + export** pipeline (e.g. Gemma 4 SFT/LoRA with their chosen toolkit, **merge** adapters if applicable, then **quantize** to GGUF using **`llama.cpp`** or compatible converters). The artifact you run locally is a **single `.gguf` file** consumed by `GGUFBackend` (`llama-cpp-python`).

---

## 3b. Hugging Face — project checkpoint

**Hugging Face model / GGUF repository (paste your `huggingface.co/...` link on the next line):**


_________________________________________________

The repo should describe which **base Gemma 4** was fine-tuned, **quantization** (e.g. Q4_K_M), and **file name(s)** for the `.gguf` you expect users to download. This codebase does not pull from Hugging Face automatically; users download the file and set `DGENE_GGUF_PATH`.

---

## 3c. Self-hosting the fine-tuned model from this repo

Use this when you want **offline** or **local GPU/CPU** inference with the **attached fine-tuned** checkpoint instead of the **stock** Gemma 4 API.

1. **Clone** this repository and install Python deps (`requirements.txt`, plus `llama-cpp-python` built for your platform; see README).
2. **Download** the project’s **`.gguf`** from Hugging Face (link in §3b) or your own mirror.
3. **Create** a `.env` in the repo root (see `.env.example`) **or** export variables in your shell.

**Environment variables for local GGUF**

| Variable | Value |
|----------|--------|
| `DGENE_GGUF_PATH` | Absolute path to the fine-tuned `.gguf` file (required for local run). |
| `DGENE_INFERENCE` | Set to `gguf` (or `local` / `finetuned`) **if** you still have `GEMINI_API_KEY` set but want to **force** GGUF; otherwise `auto` picks hosted when a key exists. |
| *(optional)* `DGENE_GGUF_CTX` | Context length (default in code: `4096`). |
| *(optional)* `DGENE_GGUF_GPU_LAYERS` | `-1` typically uses all GPU layers when CUDA/Metal is available. |
| *(optional)* `DGENE_GGUF_MAX_TOKENS` | Generation cap per candidate (default in code: `1024`). |

**Important:** With `DGENE_INFERENCE=auto`, **if `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set**, the server will prefer **hosted stock Gemma 4**. To run **only** your GGUF, either unset the API keys or set `DGENE_INFERENCE=gguf`.

**Run the full compiler UI**

```bash
python3 server.py
```

Open the printed URL (default `http://127.0.0.1:8765/`). RAG still uses `igem_dataset.jsonl` and Chroma on first compile unless you disable it (`DGENE_RAG=0`).

**Run the Streamlit playground**

```bash
streamlit run app.py
```

Both entry points use the same `inference.get_backend()` logic once `.env` is loaded.

---

## 4. Inference layer — `inference.py`

### 4.1 Backend selection

- **`auto`** (default): API key present → `GeminiBackend`; else valid `DGENE_GGUF_PATH` → `GGUFBackend`; else error (no mock).
- **`gemini` / `hosted`**: API only.
- **`gguf` / `local` / `finetuned`**: GGUF only.

### 4.2 Hosted Gemma (`GeminiBackend`)

- REST **`generateContent`** / optional **SSE `streamGenerateContent`** to the Generative Language API (`urllib` only).
- **System instruction** constrains the model to a single rigid format: opening `<|channel>thought`, one short prose paragraph (no bullets), `<channel|>`, one continuous DNA line (A/C/G/T, no spaces), then `</circuit>`.
- **Stop sequence** `</circuit>` limits runaway generation.
- **Default `maxOutputTokens`** for hosted calls is **8192** (override with `DGENE_GEMINI_MAX_OUTPUT` for very long single-piece sequences; excessively large caps slow compiles by encouraging pre-`</circuit>` rambling).
- **SSE early close:** when `streamGenerateContent` is used (`DGENE_GEMINI_STREAM`), the client stops reading the stream once the accumulated text parses and includes `</circuit>` (on by default; disable with `DGENE_GEMINI_STREAM_EARLY_CLOSE=0` if needed).
- **Multi-candidate** `generate(prompt, n)` runs up to **n** completions (default 4 in the web compiler) at a **temperature ladder**, optionally **in parallel** (`DGENE_GEMINI_PARALLEL`, worker cap `DGENE_GEMINI_MAX_WORKERS`).
- **Parse retries**: up to 3 attempts per candidate with stricter format suffixes if `parse_thought_and_sequence` fails.

### 4.3 Local GGUF (`GGUFBackend`)

- Loads `.gguf` with `Llama(..., n_ctx, n_gpu_layers, ...)`.
- Prompt format: `<|user|>\n{user_prompt}\n<|assistant|>\n<|channel>thought\n` then samples until stop tokens including `</circuit>`.
- Candidates differ by **temperature ladder** and **deterministic seed** derived from prompt + index.

### 4.4 Parsing and display cleanup

- **`parse_thought_and_sequence`**: tolerates markdown fences, prefers the **last** well-formed channel block, concatenates ACGT runs after `<channel|>`, respects `</circuit>` as end of DNA.
- **`sanitize_thought_for_display`**: strips meta-fields like “Paragraph:” so UI and RAG see cleaner prose while preserving identifiers (e.g. BBa_, J23100).

---

## 5. iGEM RAG + NCBI fallback — `igem_rag.py` and `ncbi_gene.py`

There are **two** retrieval-heavy mechanisms:

1. **Legacy post-hoc substitution (`apply_rag_substitution`)** — Runs **only** when the candidate came from **channel DNA** (`DGENE_COMPILE_MODE=legacy` or API-key fallback). After Gemma emits a full sequence, the compiler splits it into **N proportional chunks** (one per discovered ordered slot) and tries to **replace each slice** with registry or NCBI DNA when verification thresholds pass.

2. **RAG-first menu retrieval (`circuit_rag_first.build_part_menu`)** — Runs **before** the Gemma “compiler” call: queries Chroma from flattened **`intent.roles[].retrieval_queries`**, merges unique **`BBa_`** rows into a numbered menu, and mandates that **`ORDERED_PART_LIST`** DNA come **only** from those rows (+ optional **`assemble_sequence` fallback** direct lookup in `igem_dataset.jsonl`). No proportional substitution step exists on this path.

### 5.1 Why slot count matters (equal-chunk pitfall)

Assembly maps the *i*-th parsed part to the *i*-th **equal fraction** of the model sequence. If the thought only names **two** parts (e.g. `B0034` and `B0015`) while the model actually encoded a **full circuit** (promoters, sensors, reporters), then **each “slot” is half the plasmid** — and substituting two short registry parts **replaces almost the entire construct**, destroying everything between them. **Mitigation in code:** (1) **broad part discovery** from the thought (see §5.4) so *N* matches the real part count; (2) server logs a **warning** when *N* ≤ 2 and the model sequence is very long; (3) **NCBI** can still supply real DNA for gene symbols missing from iGEM (§5.6).

### 5.2 Index

- **Corpus:** `igem_dataset.jsonl` (path override `DGENE_IGEM_JSONL`).
- **Vector store:** Chroma persistent client (`DGENE_CHROMA_PATH`, default `.chroma_igem`).
- **Embedding model:** `sentence-transformers` **`all-MiniLM-L6-v2`**, L2-normalized; collection metadata sets **`hnsw:space: cosine`**.
- **Indexed document text:** concatenation of `part_name`, `part_type`, `short_desc` (not the raw DNA — DNA lives in metadata).
- **First run:** `ensure_indexed()` loads JSONL in batches and writes embeddings if the collection is empty.
- **Extending the corpus:** New lines in `igem_dataset.jsonl` are **not** visible to Chroma until the collection is rebuilt (delete the Chroma directory or point `DGENE_CHROMA_PATH` at a fresh path). The separate **registry token vocabulary** (§5.4) also keys off JSONL **mtime** for its cache.
- **Legacy substitution thresholds:** **`DGENE_RAG_MIN_SIM`** (default **0.6**) gates non-promoter slots; **`DGENE_RAG_MIN_SIM_PROMOTER`** (default **0.80**) raises the floor for **`Promoter`** hints via **`min_similarity_for_slot`** — see §5.5.

### 5.3 Retrieval (per slot query)

For each query string:

1. **Exact path:** If the query contains **`BBa_…`**, Chroma `where={"part_name": …}` exact lookup (similarity 1.0).
2. **Alias path:** Word-boundary aliases (e.g. `luxR` → `BBa_C0062`, `b0034` → `BBa_K812053`) also resolve to exact `part_name` lookup (type filter relaxed for alias hits when needed).
3. Else **semantic query:** embed the query string; optional **`part_type` filter** (Promoter / RBS / CDS / Terminator) to reduce cross-type confusion.

Similarity is **`1 - distance`** in `[0, 1]`.

### 5.4 Part discovery — ordered `(type_hint, query_text)` list

Thought text is scanned **in document order**. Multiple passes contribute **deduplicated gene names** (case-insensitive **first occurrence wins** for type hint):

1. **Formal IDs:** `BBa_*`, `J#####`, `B####` (with **B0010–B0019 → Terminator**, **B0030–B0039 → RBS** priors when the prose does not spell out a type).
2. **Curated synonyms:** Expanded `_GENERIC_PARTS` (chromoproteins e.g. `amilCP`, quorum proteins, common promoters) — maintained as human-readable shortcuts, not as a substitute for the full registry vocabulary.
3. **Registry-derived vocabulary (`_ensure_registry_token_index`):** One pass over **`short_desc`** in the entire JSONL extracts gene-shaped tokens (`amilCP`, `mCherry`, `pBAD`, `LldR`, …), tallies **`part_type`**, filters **English / method stopwords** and **small-molecule inducers** (`IPTG`, `aTc`, `AHL`, …), and requires ≥ **2** occurrences. Result (~**3k+** symbols, typical build **~100 ms**) is cached in **`.chroma_igem/registry_tokens.json`** (versioned + JSONL **mtime**, same gitignored tree as Chroma).
4. **CamelCase fallback:** Tokens like `PhzR` **not** in iGEM — matched with a conservative regex **only if** a part-type-ish keyword appears within ±60 characters (`promoter`, `expression`, `sensor`, …) and the token passes a short **blocklist** (`DNA`, restriction enzymes, boolean gate words, …).

**Type attachment (`_scan_window_for_type`):** Prefers the **text after** the symbol (standard iGEM prose: “B0034 **RBS**”). The **after-window** is truncated at the next **`BBa_` / `B####` / `J#####`** so “`amilCP` … **`B0034` RBS**” does not label `amilCP` as an RBS. A **narrow before-window** plus **sentence-boundary rejection** avoids stealing “promoter” / “terminator” from neighboring sentences. When the window is capped because another formal ID follows, **`max_offset`** on the type-keyword match prevents binding “**RBS**” in “… terminator **follows** the **RBS** **B0034**” to **B0015**.

Fallback when **no** named parts match: **free-text-lines** extraction from headings / bullets (still drops prompt-skeleton echoes).

### 5.5 Slot assembly (`apply_rag_substitution`)

Used **only** on legacy candidates after **`generate`** channel parsing (not on `circuit_synth`, `rag_first`, or `slot_template` rows).

1. Build **queries** = ordered list **(type_hint, query_text)** (see §5.4).
2. If **no** queries → return model sequence unchanged (with audit metadata).
3. Split the **flattened uppercase** model sequence into **N** equal-ish contiguous **chunks** (N = len(queries)). Chunk *i* is the tentative DNA for slot *i*.
4. **Per slot (in order):**
   - **iGEM retrieval** as in §5.3. Slots whose **`part_type` hint is Promoter** must reach **`similarity ≥ max(DGENE_RAG_MIN_SIM, DGENE_RAG_MIN_SIM_PROMOTER)`** (`min_similarity_for_slot`): **`DGENE_RAG_MIN_SIM_PROMOTER`** defaults to **0.80** to cut weak **`pL*`** collisions; other types use **`DGENE_RAG_MIN_SIM`** (default **0.6**) alone → registry substitution sets **`sequence_source`** = `registry`; **`verified`** = true.
   - Else → **§5.6 NCBI** when applicable (gene-shaped symbol, not `BBa_/B####/J#####`, not stripped as RBS/terminator-only slot).
   - If NCBI succeeds → **`sequence_source`** = `ncbi`; **`verified`** = true; **`match_kind`** = `ncbi-gene`.
   - Else → keep **model** chunk; **`sequence_source`** = `model`; **`verified`** = false (optionally **`reject_reason`** if a sub-threshold registry hit existed).
5. Concatenate slot sequences → **final** DNA for passes / API.

Slots are never **dropped**, but total length changes when substituted sequences differ in length from the original chunks.

**Disable RAG entirely:** `DGENE_RAG=0` / `false`.

### 5.6 NCBI Gene fallback — `ncbi_gene.py`

When iGEM **does not verify** a slot, the compiler can fill **CDS-shaped** symbols from **NCBI Entrez** (bacterial locus / intronless assumption):

1. **`esearch`** `db=gene` with **`{Symbol}[Gene Name] AND "{organism}"[Organism]`**.
2. Organism **order**: prompt-aware heuristics (e.g. *Pyocyanin* / *Phz* mentions bump **Pseudomonas aeruginosa** first; *E. coli* bumps ***Escherichia coli*** first), overrideable via **`DGENE_NCBI_ORGANISMS`** (comma-separated scientific names).
3. **`esummary`** Gene → **`genomicinfo`** `chraccver`, `chrstart`, `chrstop`.
4. **`efetch`** `db=nuccore` genomic **FASTA** slice on the reported strand.

**Caching:** Hits and explicit misses stored under **`.chroma_igem/ncbi_gene_cache.json`** so repeat compiles do not hammer NCBI.

**Promoter slots (default off):** Gene summaries expose genomic CDS spans — **not** cis‑regulatory promoter DNA. By default **`DGENE_NCBI_PROMOTER_SLOTS=0`** blocks Entrez on **Promoter**-typed legacy slots; set **`1`** only if you consciously accept that semantics mismatch.

**Environment**

| Variable | Purpose |
|----------|---------|
| `NCBI_API_KEY` or `DGENE_NCBI_API_KEY` | Optional; raises Entrez throughput (10 vs 3 req/s). Free at NCBI account settings. |
| `DGENE_NCBI_EMAIL` | Recommended for polite-use policy |
| `DGENE_NCBI` | Set `0` / `false` to disable fallback |
| `DGENE_NCBI_ORGANISMS` | Comma-separated default search order |
| `DGENE_NCBI_PROMOTER_SLOTS` | `0` (default) — skip Entrez on promoter slots; `1` — allow (see caveat above) |

**Caveats:** Symbols absent or renamed in Gene (some community names like **`PhzI`** may need a manual JSONL row or synonym). Returned interval is **NCBI Gene’s genomic span** on the assembly — appropriate for bacterial ORFs in most hackathon demos, not a substitute for lab validation.

### 5.7 API audit fields, design QA, snapshots (`server.py`, `web/`)

- **`rag.parts[]`:** Legacy/post-substitution mirrors **`verified`** / **`sequence_source`** per slot. **`circuit_synth`** / **`rag_first`** / **`slot_template`** populate **`parts`** (and **`assembly_trace`**) from the deterministic stitch — **`circuit_synth`** uses catalog-derived **`circuit_synth·vetted_catalog`** queries with **`similarity: 1.0`**.
- **`rag.map_slots[]`:** Intervals for the circular map; server aligns **`verified`** / **`sequence_source`** from **`parts`** where applicable.
- **`rag.intent`:** Raw **`circuit_intent`** or **`circuit_rag_first`** JSON intent embedded on structured pipelines for **`prompt_alignment`** and UI copy.
- **`rag.verification` (circuit_synth only):** **`passes`**, textual **`summary`**, and a **`truth_table`** listing **`inputs`** bits vs **`expected`** vs **`actual`** regulatory outputs from **`circuit_verify`**.
- **`rag.backbone` (circuit_synth, optionally mirrored semantics on slot-template):** Backbone **`name`**, **`bba_id`**, **`ori_bba_id`**, **`resistance_bba_id`**, MCS prefix/suffix metadata so auditors see which RFC10 scaffold closed the construct.
- **`rag.compiler_raw` / `compiler_temperature` (rag_first):** Full Gemma compiler output (truncated only for streaming previews elsewhere); **`compiler_temperature`** records which ladder step produced that variant.
- **`rag.expert_lint`:** Output of **`design_expert_lint.lint_ordered_construct`** — checks **`circuit_parts`** cognate promoter↔TF pairing rules over **`ordered_part_names`** / assembly trace ( **`missing_regulator_cds`**, **`no_common_terminator`** heuristic, **`grade`** **`strong` / `mixed` / `weak`** ).
- **`rag.expert_review`:** Present when **`DGENE_EXPERT_REVIEW=1`** — hosted Gemma returns **`verdict`** (**`likely_coherent` / `needs_revision` / `unclear`**), **`summary`**, **`concerns[]`** after inspecting brief + ordered **`BBa_`** lines + **`short_desc`** snippets from **`igem_dataset.jsonl`**.
- **`snapshot_id`:** After each successful **`_finalize_compile_result`**, the server may persist JSON under **`.design_snapshots/<32-hex>.json`** (gitignored) and return **`snapshot_id`** — **`GET /api/snapshot?id=<sid>`** reloads it (**`DGENE_SNAPSHOTS=0`** disables writes **and** the loader).

---

### 5.8 Topology compiler — `circuit_ir`, `circuit_intent`, `circuit_parts`, `circuit_synth`, `circuit_verify`, `circuit_pipeline`

When the web **`compile`** runs with **`DGENE_COMPILE_MODE=circuit_synth`** (default):

1. **`circuit_intent.extract_circuit_spec`** — Hosted Gemma returns strict JSON: **`applicable`**, **`inputs`** (canonical keys only — **`supported_inducers()`** in **`circuit_parts`**), **`output`** (**`supported_reporters()`**), **`logic`** (**`BUF` / `NOT` / `AND` / `OR` / `NAND` / `NOR`**). Parsing anchors **`\"applicable\"`** so preamble prose does not break **`JSONDecoder`**. **`canonical_inducer_name`** plus **`NOT(name)`** operand shorthand normalize keys before **`circuit_ir.LogicSpec`** validation.

2. **`circuit_ir.CircuitSpec`** — Holds **`truth_table()`** independent of biology — **`circuit_verify`** compares synthesized cassette graphs against those expectations.

3. **`circuit_synth.synthesize`** — Reads sequences strictly from **`igem_dataset.jsonl`** entries wired in **`circuit_parts`** (cognate promoters/TFs, Hrp **`AND`/`NAND`**, **`OR`** branches, TetR **`NOR`**, lactate/pyocyanin catalog metaphors). Emits **`TopologyError`** when topology unsupported → **`circuit_pipeline`** returns **`None`** for verified candidate slot.

4. **`circuit_verify.verify_plasmid`** — Regulatory-graph simulation → **`truth_table` mismatch ⇒ discard candidate**.

5. **`circuit_pipeline.compile_hybrid_variants`** (**sync**) / **`compile_hybrid_variants_iter`** (**async job**) — If synthesis passes verification: **`Candidate`** with **`rag.pipeline=\"circuit_synth\"`** becomes **`cand_0`** after ID normalization inside **`compile_hybrid_variants`** (`candidate_id` `cand_circuit_0` renamed internally); **`thought`** documents backbone (**`make_backbone_ref`** distribution BBa), MCS framing (**ori + chloramphenicol resistance + MCS → cassettes → MCS**), and verification summary. **`need = n - 1`** additional **`run_rag_first_*`** paths issue **one** **`extract_intent_json`** call **shared across all RAG-first variants** in that compile (separate Gemma round-trip from **`extract_circuit_spec`**); **slot-template** may prepend the first **RAG-first** variant (§5.9).

**Important:** **`circuit_synth` requires hosted Gemma for intent extraction.** Without API keys, **`server.py`** drops to **`legacy`** path entirely for compile jobs.

When **`build_circuit_candidate`** returns **`None`** (brief not **`applicable`**, **`TopologyError`**, or verification failure), **`compile_hybrid_variants`** does **not** emit a topology row — it delegates **all `n`** candidate slots to **`run_rag_first_variants`** so users still receive registry-grounded designs.

Static verification is **not** a wet-lab guarantee — it rules out inconsistent Boolean wiring, not burden or toxicity.

---

### 5.9 Slot-template cassette — `slot_template_compile.py`

**Problem addressed:** Open-ended briefs (e.g. dual-analyte biosensors) often route to **RAG-first**, where the LLM orders many **`BBa_`** IDs from a large menu — producing **very long** concatenations or weak semantic hits (e.g. confounding “pyocyanin” with unrelated registry text). **Post-hoc** `apply_rag_substitution` (§5.5) also risks **equal-chunk** misalignment when the thought channel names too few parts.

**Approach:** After **`circuit_rag_first.extract_intent_json`**, an optional pass builds **one** compact linear insert **without** asking the menu compiler for an ordered list:

1. **Parse** **`gate`** (**`AND` / `OR` / `BUF`** — **NOT** gate templates are **not** implemented; **`gate=NOT`** returns **`None`** → pipeline falls through to menu+LLM only), **`input_analytes`**, **`reporter`** (defaults via heuristics when prose mentions dual cues).
2. **Retrieve per slot** via **`igem_rag.retrieve_parts`** with explicit **`part_type` filter**:
   - **Promoter** — query combines analytes + gate wording; top hits are **re-ranked** (e.g. AND requests down-weight descriptions that look like **OR / NOT** gates; analyte substrings in **`short_desc`** up-weight).
   - **RBS** — B0034-class alias / semantic query.
   - **CDS** — reporter (e.g. exact path toward **BBa_K592009** for amilCP when it resolves).
   - **Terminator** — B0015-class.
3. **Concatenate** sequences in that fixed order. **`rag.pipeline === \"slot_template\"`**; **`assembly_trace`** carries **`source: slot_template`** segments (plus backbone segments when embedded).
4. **Optional full vector (default on)** — **`embed_slot_template_in_ecoli_backbone()`** wraps the cassette in the **same E. coli RFC10 backbone** as **`circuit_synth`** (**ColE1-class ori + chloramphenicol resistance + MCS** from **`make_backbone_ref()`**), shifts **`start_bp` / `end_bp`**, and prepends/appends backbone rows to **`map_slots`** so the circular map closes over a complete plasmid. Disable with **`DGENE_SLOT_TEMPLATE_EMBED_BACKBONE=0`**.

**Integration:** **`run_rag_first_variants_iter`** yields **slot-template first** (when **`slot_template_enabled()`** and **`n > 0`**), then **`llm_budget = n - yielded`** menu-compiler variants at temperatures **`[0.25, 0.4, 0.55, …]`** from **`rag_first_candidate_temps`**. **`run_rag_first_single`** returns **only** the slot-template **`Candidate`** immediately when it succeeds (short-circuit before **`build_part_menu`**). **`server.py`** skips post-hoc **`apply_rag_substitution`** for **`slot_template`** the same way as **`rag_first`** and **`circuit_synth`**. **No truth-table proof** — unlike §5.8 when verification passes, arbitrary analytes in the template path are **not** Boolean-verified unless **`circuit_synth`** also fires.

**Environment**

| Variable | Purpose |
|----------|---------|
| `DGENE_SLOT_TEMPLATE` | `1` (default) / `true` — enable; `0` / `false` / `no` — disable |
| `DGENE_SLOT_TEMPLATE_MIN_SIM` | Minimum **raw** embedding similarity for the chosen promoter (default **0.52**); lower if borderline composites never match |
| `DGENE_SLOT_TEMPLATE_MAX_PROMOTER_BP` | Skip promoter hits longer than this (default **4000**) to avoid megabase composite “promoters” |
| `DGENE_SLOT_TEMPLATE_EMBED_BACKBONE` | `1` (default) — wrap slot-template cassette in ori+CmR+MCS scaffold matching **`circuit_synth`**; `0` / `false` — cassette-only DNA |

Intent schema additions live in **`circuit_rag_first._INTENT_SYSTEM`**: the model is asked to emit **`gate`**, **`input_analytes`**, and **`reporter`** alongside legacy **`roles`** / **`logic_summary`**.

---

### 5.10 RAG-first compiler pipeline — step-by-step (`circuit_rag_first.py`)

This path powers **`DGENE_COMPILE_MODE=rag_first`** entirely and **`circuit_synth` padding** after the verified candidate.

| Step | What happens |
|------|----------------|
| **1 · Intent** | **`extract_intent_json`** — Gemma with **`_INTENT_SYSTEM`** emits JSON (`gate`, `input_analytes`, `reporter`, **`roles[]` with `retrieval_queries[]`**, chassis, logic_summary). Missing newer keys back-filled (**`unknown`**, empty lists). |
| **2 · Menu** | **`build_part_menu`** — **`ensure_indexed()`**, **`_flatten_retrieval_queries`** (always adds generic anchors like “strong RBS”, AND-gate English queries when roles sparse), **`retrieve_parts`** per query with **`top_k = DGENE_RAG_FIRST_TOP_K`** (default **15**, clamped **1–50**). Dedupe by **`part_name`**; stable sort Promoter→RBS→…→Terminator for prompt readability. |
| **3–4 · Compiler** | **`run_compiler`** — User message bundles brief + truncated intent JSON + numbered menu. **`_COMPILER_SYSTEM`** demands **`<reasoning>...</reasoning>`**, blank-line **`ORDERED_PART_LIST`**, one **`BBa_`** per line (digit runs **`\\d{4,}`** for long BBa IDs), **`</circuit_design>`** stop. **`stop_sequences`** includes **`</circuit_design>`**. **`DGENE_RAG_FIRST_COMPILER_MAX_TOKENS`** default **3072**. |
| **5 · Parse** | **`parse_ordered_bba`** — Never scans model chatter above **`ORDERED_PART_LIST`** except **`BBa_` extraction fallback when marker exists**. **`_COMPILER_PARSE_RETRY_SUFFIX`** triggers one colder **`run_compiler`** pass if list empty. |
| **6 · Caps** | **`DGENE_RAG_FIRST_MAX_PARTS`** default **48** — **`RuntimeError`** if exceeded (runaway enumeration guard). |
| **7 · DNA** | **`assemble_sequence`** — Concatenate **`sequence`** fields preferring **`menu_by_name`**, else **`igem_dataset.jsonl`** index; **`trace`** rows carry **`start_bp`/`end_bp`**, **`source`** **`menu`/`jsonl`/`missing`**. |
| **8 · Thought** | **`extract_reasoning_for_display`** — Tag-aware excerpt + **`sanitize_thought_for_display`**, chatter scrubbers, sentence/caps via **`DGENE_RAG_FIRST_REASONING_*`**. |

**Compiler prompt hygiene:** `_INTERNAL_DIALOGUE_LINE` regex strips obvious meta narration when clipping reasoning for UI — encourages **`_COMPILER_SYSTEM`** bans (“Wait,” “Option A,” …).

---

## 6. Compiler passes — `passes.py`

Runs on each candidate’s **final** API DNA string (after post-hoc RAG when **`legacy`**; for **`circuit_synth`**, **`rag_first`**, or **`slot_template`**, the same string is already registry-stitched upstream). Examples:

- **Parse:** proportional feature labels (aligns with a default “J23100–lacO–B0034–sfGFP–B0015” style map in UI terms).
- **Lint / score:** ORF scan (ATG→stop), GC band, direct repeats, Type IIS forbidden sites (both strands), common restriction sites, E. coli **CAI**, Shine–Dalgarno heuristic upstream of start, hairpin heuristic, biosecurity stub.

Results are JSON-serializable structs with diagnostics (positions, severities).

---

## 7. Ranking — `ranker.py`

For each candidate, **`score_candidate`** maps pass metrics into four objectives in `[0,1]`:

- **expression** (CAI + RBS proxy)
- **low_burden** (penalize low CAI, length, repeats)
- **gc_balance**
- **cleanliness** (Type IIS warnings)

Those four define the **composite** scalar (weighted sum: **expression 0.40**, **low_burden 0.25**, **gc_balance 0.15**, **cleanliness 0.20** — see **`ranker.WEIGHTS`**) and the **Pareto front** (dominance in objective space only — **no** prompt or pipeline weights inside Pareto).

**Default `best_id` and list order** (what the UI opens on first) sorts candidates by:

1. **`pipeline_tier`** — `circuit_synth` (3) > `slot_template` (2) > `rag_first` (1) > other (0). This prefers **truth-table-verified topology** and **structured** templates over open-ended RAG-first DNA when both appear in the same compile job.
2. **`prompt_alignment`** — share of salient **prompt words** (length-filtered, stopword-stripped) that appear in **`thought`**, **`rag.intent`**, **`rag.parts`**, **`rag.map_slots`**, and **`ordered_part_names`**. Helps surface variants that **mention lactate / pyocyanin / reporter** in metadata when composites would otherwise tie.
3. **`composite`** — breaks remaining ties on lab-plausibility heuristics.

`attach_fidelity_scores` (called from **`server.py`** when building each candidate row) adds **`pipeline_tier`** and **`prompt_alignment`** into the **`scores`** dict alongside the four objectives and composite. **`rank()`** applies the tuple sort above, then assigns **`rank`** and **`is_pareto`**.

---

## 8. Web compiler — `server.py` + `web/`

- **`ThreadingHTTPServer`** so long compiles don’t block health checks or static files. **`PORT`** env (default **8765**); if busy, the server scans **`PORT … PORT+31`** and binds the first free socket (stderr explains when bumped).
- **`POST /api/compile`** — JSON body **`{ "prompt": string, "n": number, "progress"?: boolean }`**. **`n`** clamps to **`[1, 8]`** (default **4**).
  - **`progress: false` or omitted:** synchronous response — full ranked payload + optional **`snapshot_id`** (§5.7).
  - **`progress: true`:** **`202 Accepted`** with **`{ "job_id": "<hex>" }`**. Poll **`GET /api/compile/status?job_id=…`** until **`done: true`**. Poll payloads include **`lines`** (timestamped progress strings), optional **`streams`** (live Gemma/SSE previews keyed by stream id), **`error`** on failure, and **`result`** while running or complete.
  - **Incremental `result`:** While **`done` is false** but variants already exist, **`result`** may carry **`partial: true`**, **`variants_ready`**, **`variants_total`**, **`candidates`**, **`best_id`** — UI can rank partial lists without waiting for all **`n`** variants. Async jobs use **`compile_hybrid_variants_iter`** / **`run_rag_first_variants_iter`** / **`generate_iter`** when available so candidates stream in order.
  - **RAG stderr mirror:** Progress jobs attach **`igem_rag.set_rag_debug_mirror`** so **`DGENE_RAG_DEBUG`** lines also append to the job **`lines`** feed as **`[rag] …`** (when debug enabled).
- **`POST /api/fix`** — Targeted recompile: **`original_prompt`**, **`current_sequence`**, **`fix_type`** (`repeats` \| `type_iis` \| `cai` \| `rbs` \| `repeats_type_iis`), **`candidates`** (full workspace array), optional **`source_candidate_id`**. Builds a constraint-augmented prompt, runs **`_compile(..., n=1)`** with **`user_prompt_for_alignment`** pinned to the original brief, **merges** the new row, **re-ranks**, and returns **`fix.new_candidate_id`**, **`still_flagged`**, **`pass_cleared`**, etc.
- **`GET /api/snapshot?id=<32-hex>`** — Reload JSON saved under **`.design_snapshots/`** when **`DGENE_SNAPSHOTS`** enabled (§5.7); **`503`** if snapshots disabled.
- **`GET /api/health`** — model id / **`backend_kind`** / hosted **`api_model_id`** or GGUF filename.
- **Pipeline** (`_compile` / async job): **`DGENE_COMPILE_MODE`** branch (§1) → per candidate **`_ranked_row_from_candidate`**: skip **`apply_rag_substitution`** when **`rag.pipeline ∈ {circuit_synth, rag_first, slot_template}`** → **`run_passes`** → **`score_candidate`** → **`attach_fidelity_scores`** → global **`rank`** after all candidates (sync) or incremental **`rank`** on prefixes (async partials).
- **Static** files from **`web/`** (HTML/CSS/JS): circular plasmid map with restriction sites, **full annulus base ring** + **gap underlay** for unannotated **`map_slots`** intervals, hover tooltips (leaders use **`pointer-events: none`** so segments receive hover), **⌘/Ctrl+scroll zoom** and **drag-pan** on the SVG viewport (`#plasmidSvg` / `#viewport`), **unverified `*`** markers, candidate table / Pareto chart, **partial compile** respects **selected candidate** (does not reset selection to **`best_id`** on every streaming update), FASTA/GenBank download mirroring **`app.py`** logic in JS.
- **Access log noise:** Default suppresses **`/api/compile/status`** and **`/api/health`** lines — set **`DGENE_HTTP_LOG=all`** for full Apache-style logging. **`DGENE_SERVER_DEBUG=1`** adds **`[dgene/server HH:MM:SS]`** tracing.

---

## 9. Streamlit playground — `app.py`

**Legacy-only today:** **`run_inference`** emits **one** channel-tagged completion → **`parse_thought_and_sequence`** → **`sanitize_thought_for_display`** → **`apply_rag_substitution`** (same iGEM → NCBI → model cascade as **`server.py` legacy**) → **`generate_interactive_plasmid_plot`** (Bokeh + **`dna_features_viewer`** with **quarter-length** feature segments: promoter / RBS / CDS / terminator). It does **not** invoke **`circuit_pipeline`** / **`circuit_rag_first`** — use **`python3 server.py`** for topology or menu compilers. Metrics (length, GC, Wallace Tm), FASTA/GenBank downloads, typewriter-style reasoning expander.

---

## 10. Configuration (summary)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `DGENE_GOOGLE_API_KEY` | Hosted Gemma |
| `DGENE_GEMINI_MODEL` | e.g. `gemma-4-31b-it` |
| `DGENE_GEMINI_HTTP_TIMEOUT`, `DGENE_GEMINI_HTTP_HEARTBEAT_SEC` | Per-request timeout (default **600** s); heartbeat log cadence during generation (**0** disables) |
| `DGENE_MIN_PARSE_DNA_LEN` | Minimum accepted DNA length after `<channel|>` (default **12**) |
| `DGENE_GGUF_PATH` | Local GGUF |
| `DGENE_INFERENCE` | `auto`, `gemini`, `gguf`, … |
| `DGENE_IGEM_JSONL`, `DGENE_CHROMA_PATH`, `DGENE_RAG`, `DGENE_RAG_MIN_SIM`, `DGENE_RAG_MIN_SIM_PROMOTER` | RAG corpus + Chroma dir + master toggle + legacy substitution floors (promoter slot uses **max** of the two sim vars — §5.5); registry token + NCBI JSON caches live under **`DGENE_CHROMA_PATH`** |
| `NCBI_API_KEY` / `DGENE_NCBI_API_KEY`, `DGENE_NCBI_EMAIL`, `DGENE_NCBI`, `DGENE_NCBI_ORGANISMS`, `DGENE_NCBI_PROMOTER_SLOTS` | NCBI Gene fallback (§5.6) |
| `DGENE_GEMINI_STREAM`, `DGENE_GEMINI_STREAM_EARLY_CLOSE`, `DGENE_GEMINI_PARALLEL`, `DGENE_GEMINI_MAX_WORKERS`, `DGENE_GEMINI_MAX_OUTPUT` | Hosted streaming, SSE early close, parallelism, pool size, per-candidate output token cap (default **8192**) |
| `DGENE_COMPILE_MODE` | `circuit_synth` (default): verified boolean topology + RAG-first padding (optional **slot-template** first variant, §5.9); `rag_first`; `legacy` |
| `DGENE_RAG_FIRST_TOP_K`, `DGENE_RAG_FIRST_MAX_PARTS`, `DGENE_RAG_FIRST_COMPILER_MAX_TOKENS`, `DGENE_RAG_FIRST_REASONING_CHARS`, `DGENE_RAG_FIRST_REASONING_SENTENCES` | RAG-first menu size, BBa order cap, compiler token budget, reasoning summary clip (`circuit_rag_first.py`) |
| `DGENE_SLOT_TEMPLATE`, `DGENE_SLOT_TEMPLATE_MIN_SIM`, `DGENE_SLOT_TEMPLATE_MAX_PROMOTER_BP`, `DGENE_SLOT_TEMPLATE_EMBED_BACKBONE` | §5.9 slot-template path |
| `DGENE_SNAPSHOTS` | `1` (default) — persist compile JSON + expose **`GET /api/snapshot`**; `0` — disable |
| `DGENE_EXPERT_REVIEW` | `1`/`true` — extra Gemma reviewer JSON on ordered **`BBa_`** lists (`expert_review.py`) |
| `DGENE_HTTP_LOG`, `DGENE_SERVER_DEBUG`, `DGENE_GEMINI_DEBUG`, `DGENE_RAG_DEBUG` / `DGENE_DEBUG` | Logging / tracing switches (§8, `igem_rag.py`) |
| `PORT` | HTTP bind (**8765** default; server may walk upward **31** ports if busy) |

**Fine-tuned GGUF:** **`circuit_synth`**, **`rag_first`**, slot-template, **`/api/fix`**, **`expert_review`**, and both **`extract_*_json`** intent passes call **`generate_text_gemma4_custom`** against the **Gemini API** — they **do not** route through **`GGUFBackend`** today. A machine can use GGUF for **`DGENE_COMPILE_MODE=legacy`** channel compiles while still setting **`GEMINI_API_KEY`** for topology/menu paths.

**.env loading:** `inference.py` reads repo-root **`.env`** on import (**does not override** variables already present in `os.environ`). **Restart** `python3 server.py` after editing `.env` so subprocess picks up `NCBI_API_KEY` and other additions.

---

## 11. Limitations and hackathon framing

- **Outputs are not validated in the lab** — the stack is tooling / research; safety and wet-lab verification remain human responsibilities.
- **RAG substitution** is **similarity- and slot-based** with **equal-length chunking**. Part discovery has been hardened (§5.4–5.5), but if the **model’s prose omits named parts**, slot count stays low and **chunk alignment is still ambiguous** — treat the **`*` markers** on the map and **`rag.parts`** audit as cues, not proofs of assembly architecture.
- **Slot-template** (§5.9) shortens **RAG-first** results for dual-input-style prompts and (by default) **embeds** the cassette in a full backbone (`DGENE_SLOT_TEMPLATE_EMBED_BACKBONE`); it **does not** substitute for **circuit_verify** truth tables — metabolite-level AND/OR remains **biology + retrieval quality**, not a formal gate proof unless **`circuit_synth`** verifies.
- **iGEM corpus** covers ~30k parts in typical snapshots — not every biological name exists under the same symbol **`PhzI`**/`PhzR` as in papers; **NCBI** closes some gaps; **custom rows** in `igem_dataset.jsonl` remain valid demo extensions (remember **Chroma rebuild** §5.2).
- **NCBI sequences** come from **Gene’s annotated genomic intervals** on a reference assembly — appropriate for bacterial demo builds, **not** a guarantee of Codon optimization, strain background, or wet-lab function.
- **Passes** are heuristics (not a full Salis RBS calculator, not experimental throughput).
- **`expert_lint` / `expert_review`** — The deterministic lint only covers **catalog promoter ↔ TF rules** encoded in **`circuit_parts`**; it misses biology outside that table. The optional Gemma reviewer costs another API round-trip and is **not** wet-lab validation — inspect partial parses (**`_parse`**) when JSON decoding degrades.
- **Streamlit (`app.py`)** — Uses **`run_inference`** + legacy **`apply_rag_substitution`** only; **no** **`circuit_synth`** / **`rag_first`** / slot-template path — run **`python3 server.py`** for those demos.
- **Fine-tune recipe** (exact hyperparameters, merge, quantization command lines) is not versioned in this repo; the repo **does** ship **`generate_gemma_train.py`** and documents the **training objective** (§3). **Runtime** behavior is aligned so **stock Gemma 4 (API)** and **fine-tuned Gemma 4 (GGUF)** share the same parser and UI pipeline.

---

## 12. Repository layout (quick reference)

| Path | Role |
|------|------|
| `igem_dataset.jsonl` | Registry-derived parts corpus (generated + committed or rebuilt) |
| `gemma_train.jsonl` | Generated training JSONL (from `generate_gemma_train.py`) |
| `xml_parts.xml.gz` | Input to `extract_igem_dataset.py` (user-supplied) |
| `ncbi_gene.py` | NCBI Entrez client + on-disk cache for CDS fallback |
| `.chroma_igem/` | Persistent embedding index (gitignored); may also contain `registry_tokens.json` and `ncbi_gene_cache.json` |
| `.design_snapshots/` | Saved **`POST /api/compile`** payloads (**`.json`**, gitignored) for **`snapshot_id`** replay |
| `web/` | Static compiler UI (plasmid map, RAG / Pareto / exports / snapshots client) |
| `design_expert_lint.py`, `expert_review.py` | Ordered **`BBa_`** regulatory sniff-test + optional Gemma reviewer (§5.7) |
| `circuit_ir.py`, `circuit_intent.py`, `circuit_parts.py`, `circuit_synth.py`, `circuit_verify.py`, `circuit_pipeline.py` | Topology IR → synthesis → proof → hybrid orchestration (§5.8) |
| `circuit_rag_first.py`, `slot_template_compile.py` | Intent → menu → **`ORDERED_PART_LIST`** compiler → DNA; optional slot-template prefix (§5.9–5.10) |

---

*Technical writeup for the DGene codebase — Google hackathon submission.*
