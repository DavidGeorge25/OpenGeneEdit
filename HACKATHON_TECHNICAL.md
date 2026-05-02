# DGene — Technical overview (Google hackathon)

**DGene** is a synthetic-biology–oriented DNA “compiler.” A user describes a genetic circuit in natural language; the system returns structured reasoning, a nucleotide sequence, optional substitution of slices with **verified iGEM registry** sequences (RAG), heuristic **compiler-style checks**, **multi-objective ranking**, plasmid visualization, and **FASTA / GenBank** export. Inference is **Google Gemma 4 only**: either via the **Gemini API** (hosted) or a local **GGUF** checkpoint loaded with `llama-cpp-python`.

**Two model modes in practice**

- **Stock Gemma 4 (hosted)** — The default path for most users is **standard instruction-tuned Gemma 4** on the Gemini API (e.g. `gemma-4-31b-it` via `DGENE_GEMINI_MODEL`). No local GPU required; the same channel-tagged prompt and parser are used. Quality follows the base model plus prompting.
- **Fine-tuned Gemma 4 (local GGUF)** — The hackathon build also supports a **domain-specific fine-tune** trained on compiler-shaped examples derived from the iGEM parts corpus (see §3). That checkpoint is distributed as a **quantized `.gguf`** for self-hosted inference through `llama-cpp-python`. The app does not bundle the file; you download it (e.g. from Hugging Face) and point `DGENE_GGUF_PATH` at it.

This document walks through data acquisition, cleaning, model alignment / local weights, Hugging Face hosting, live demo, self-hosting, and how the web and Streamlit apps orchestrate inference, RAG, passes, and ranking.

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

```
User prompt
    → Inference (Gemma 4: Gemini API or GGUF)
        → Parsed (thought + DNA channel format)
            → iGEM RAG (optional Chroma + embeddings; slot-wise substitution)
                → Compiler passes (lint / score)
                    → Ranker (Pareto + composite sort)
                        → UI (web or Streamlit): maps, exports, candidate table
```

**Core modules**

| Module | Role |
|--------|------|
| `inference.py` | Backend selection, Gemma prompting, parsing `<|channel>thought` / `<channel|>` / DNA / `</circuit>` |
| `igem_rag.py` | JSONL → ChromaDB index; retrieval; **slot-based** merge with model DNA |
| `passes.py` | ORF, GC, repeats, Type IIS, restriction map, CAI, RBS heuristic, hairpins, etc. |
| `ranker.py` | Objective vectors + Pareto front + composite ordering |
| `server.py` | `ThreadingHTTPServer`, `/api/compile`, static `web/` |
| `app.py` | Streamlit single-shot UI + Bokeh plasmid map |
| `extract_igem_dataset.py` | Build `igem_dataset.jsonl` from `xml_parts.xml.gz` |
| `generate_gemma_train.py` | Build supervised `gemma_train.jsonl` for fine-tuning using hosted Gemma 4 |

---

## 2. iGEM registry → `igem_dataset.json`

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

## 5. iGEM RAG — `igem_rag.py`

RAG here is **not** “retrieve chunks and stuff into the prompt.” It runs **after** the model emits a full sequence: it tries to **replace contiguous slices** of the model DNA with **registry sequences** when retrieval confidence is high enough.

### 5.1 Index

- **Corpus:** `igem_dataset.jsonl` (path override `DGENE_IGEM_JSONL`).
- **Vector store:** Chroma persistent client (`DGENE_CHROMA_PATH`, default `.chroma_igem`).
- **Embedding model:** `sentence-transformers` **`all-MiniLM-L6-v2`**, L2-normalized; collection metadata sets **`hnsw:space: cosine`**.
- **Indexed document text:** concatenation of `part_name`, `part_type`, `short_desc` (not the raw DNA — DNA lives in metadata).
- **First run:** `ensure_indexed()` loads JSONL in batches and writes embeddings if the collection is empty.

### 5.2 Retrieval

For each query string:

1. **Exact path:** If the query contains **`BBa_…`**, Chroma `where={"part_name": …}` exact lookup (similarity 1.0).
2. **Alias path:** Word-boundary aliases (e.g. `luxR` → `BBa_C0062`) also resolve to exact `part_name` lookup.
3. Else **semantic query:** embed the query string; optional **`part_type` filter** (Promoter / RBS / CDS / Terminator) to reduce cross-type confusion.

Similarity is **`1 - distance`** in `[0, 1]`.

### 5.3 Slot assembly (`apply_rag_substitution`)

1. **Parse the model’s thought** into an ordered list of **(type_hint, query_text)** pairs:
   - Preferred: **named parts** — regex scan for `BBa_*`, `J#####` promoters, `B####` parts, and a curated list (sfGFP, lacO, …) with **local context** used to attach type words (e.g. “B0034 RBS”).
   - Fallback: **free-text lines** from sections like “Parts used” / bullet lists, with filters to drop prompt-skeleton garbage.
2. If **no** queries extracted → return model sequence unchanged (with audit metadata).
3. Split the **model sequence** into **N equal-ish contiguous chunks** (N = number of queries). Chunk *i* is the slot for query *i*.
4. For each slot, **retrieve** the best part. If **similarity ≥ `DGENE_RAG_MIN_SIM`** (default **0.6**), **replace** that chunk with the **registry** sequence; otherwise **keep** the model slice.
5. Concatenate slots → **final** sequence passed to passes / API.

This design **never drops** a slot: weak matches keep model DNA so the construct length stays coherent at the cost of possible hallucinated bases in those slots.

**Disable:** `DGENE_RAG=0` / `false`.

---

## 6. Compiler passes — `passes.py`

Runs on the **post-RAG** DNA string. Examples:

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

**Pareto front** is computed in objective space (no weights). **Composite** score breaks ties for sort order. Candidates are returned sorted by composite, with `rank` and `is_pareto`.

---

## 8. Web compiler — `server.py` + `web/`

- **`ThreadingHTTPServer`** so long compiles don’t block health checks or static files.
- **`POST /api/compile`** with JSON `{ "prompt", "n", "progress"? }`:
  - **`progress: false`:** synchronous JSON result.
  - **`progress: true`:** `202` + `job_id`; client polls **`GET /api/compile/status?job_id=`** for `lines`, optional `streams`, then `result` when `done`.
- **Pipeline** (`_compile`): `backend.generate` → per-candidate RAG → `run_passes` → `score_candidate` → `rank`.
- **`GET /api/health`:** model id / backend kind / GGUF filename for fine-tuned runs.
- **Static** files from `web/` (HTML/CSS/JS plasmid renderer, candidate table, FASTA/GenBank download mirroring `app.py` logic in JS).

---

## 9. Streamlit playground — `app.py`

Single-shot flow: **form** → `run_inference` (one candidate) → `parse_thought_and_sequence` → `sanitize_thought_for_display` → `apply_rag_substitution` → `generate_interactive_plasmid_plot` (Bokeh + `dna_features_viewer` with **quarter-length** feature segments: promoter / RBS / CDS / terminator). Metrics (length, GC, Wallace Tm), FASTA/GenBank downloads, typewriter-style reasoning expander.

---

## 10. Configuration (summary)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `DGENE_GOOGLE_API_KEY` | Hosted Gemma |
| `DGENE_GEMINI_MODEL` | e.g. `gemma-4-31b-it` |
| `DGENE_GGUF_PATH` | Local GGUF |
| `DGENE_INFERENCE` | `auto`, `gemini`, `gguf`, … |
| `DGENE_IGEM_JSONL`, `DGENE_CHROMA_PATH`, `DGENE_RAG`, `DGENE_RAG_MIN_SIM` | RAG |
| `DGENE_GEMINI_STREAM`, `DGENE_GEMINI_STREAM_EARLY_CLOSE`, `DGENE_GEMINI_PARALLEL`, `DGENE_GEMINI_MAX_WORKERS`, `DGENE_GEMINI_MAX_OUTPUT` | Hosted streaming, SSE early close, parallelism, pool size, per-candidate output token cap (default **8192**) |

Stock **Gemini API** vs **fine-tuned GGUF:** `auto` prefers the API when keys are set; force local weights with `DGENE_INFERENCE=gguf` and `DGENE_GGUF_PATH` (details in §3c).

`.env` is read on import without overriding existing environment variables.

---

## 11. Limitations and hackathon framing

- **Outputs are not validated in the lab** — the stack is tooling / research; safety and wet-lab verification remain human responsibilities.
- **RAG substitution** is similarity- and slot-based; misaligned part counts or model reasoning errors can still yield biological nonsense even when some slots are “verified.”
- **Passes** are heuristics (not a full Salis RBS calculator, not experimental throughput).
- **Fine-tune recipe** (exact hyperparameters, merge, quantization command lines) is not versioned in this repo; the repo **does** ship **`generate_gemma_train.py`** and documents the **training objective** (§3). **Runtime** behavior is aligned so **stock Gemma 4 (API)** and **fine-tuned Gemma 4 (GGUF)** share the same parser and UI pipeline.

---

## 12. Repository layout (quick reference)

| Path | Role |
|------|------|
| `igem_dataset.jsonl` | Registry-derived parts corpus (generated + committed or rebuilt) |
| `gemma_train.jsonl` | Generated training JSONL (from `generate_gemma_train.py`) |
| `xml_parts.xml.gz` | Input to `extract_igem_dataset.py` (user-supplied) |
| `.chroma_igem/` | Persistent embedding index (gitignored pattern) |

---

*Technical writeup for the DGene codebase — Google hackathon submission.*
