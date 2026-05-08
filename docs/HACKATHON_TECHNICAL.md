# OpenGeneEdit — Technical reference

**OpenGeneEdit** is a natural-language → DNA “compiler” for synthetic biology: the user describes a genetic circuit; the system returns reasoning, one or more candidate sequences, iGEM-grounded audits (when applicable), heuristic compiler **passes**, **Pareto**-aware **ranking**, plasmid visualization, and **FASTA / GenBank** export.

**Branding:** user-facing **OpenGeneEdit**; stderr tags **`[oge/…]`**; env vars **`DGENE_*`** (legacy-compatible).

**Inference:** **Gemma 4 only** — **Google Generative Language API** (`urllib` in `inference.py`) **or** local **`.gguf`** via **`llama-cpp-python`** (llama.cpp bindings). No mock backend.

---

## 1. Model backends and API-key gating

| Backend | Env | Role |
|--------|-----|------|
| Hosted | `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `DGENE_GOOGLE_API_KEY`, `DGENE_GEMINI_MODEL` | Default when keys exist (`DGENE_INFERENCE=auto`). **Mid-compile** **`search_igem_registry`** tool loop is **hosted-only**. |
| Local GGUF | `DGENE_GGUF_PATH`, optional `DGENE_GGUF_CTX`, `DGENE_GGUF_GPU_LAYERS`, `DGENE_GGUF_MAX_TOKENS`, `DGENE_GGUF_CHAT_MAX_TOKENS` | **`GGUFBackend`** — legacy channel sampling **and** **`generate_text_gemma4`** / **`generate_text_gemma4_custom`** (intent JSON, RAG-first compiler, expert review) when **`DGENE_INFERENCE=gguf`** or **`auto`** without API keys. Compiler tools are disabled on GGUF. |

**Important:** `circuit_synth` and `rag_first` require **`hosted_generation_ready()`** (`rag_first_configured()`): either an API key **or** an initialized GGUF backend. If neither applies, **`server.py`** falls back to **legacy** `backend.generate` + post-hoc RAG (**stderr warning**). **`circuit_pipeline.compile_hybrid_variants_iter`** raises if there is **no** topology candidate **and** hybrid LLM steps cannot run.

---

## 2. Hackathon prize alignment (Gemma 4 tracks)

Condensed from official track copy:

| Track | Prize | Fit for OpenGeneEdit |
|-------|--------|----------------------|
| **Health & Sciences** | $10,000 | *Bridge the gap between humans and data; accelerate discovery / democratize knowledge.* Aligns with NL→structured designs, registry RAG, audits, topology verification, exports. |
| **llama.cpp** | $10,000 | *Best innovative Gemma 4 on **resource-constrained hardware**.* Story: quantized **GGUF** + **`llama-cpp-python`**, CPU/partial GPU (`DGENE_GGUF_GPU_LAYERS`). |
| **Ollama** | $10,000 | *Gemma 4 **locally via Ollama**.* This repo’s default local stack is **`llama-cpp-python`**, not Ollama; align judging narrative with a documented Ollama workflow or integration if targeting this prize. |
| **Unsloth** | $10,000 | *Best **Unsloth** fine-tune of Gemma 4 for an impactful task.* **`data/gemma_train.jsonl`** from **`scripts/generate_gemma_train.py`** feeds external Unsloth SFT/LoRA → merge → GGUF. |

---

## 3. Compile modes (`DGENE_COMPILE_MODE`)

| Mode | Behaviour |
|------|-----------|
| **`circuit_synth`** (default) | **`build_circuit_candidate`**: hosted Gemma extracts boolean **`CircuitSpec`** (`circuit_intent` → `circuit_ir`). **`circuit_synth`** + **`circuit_verify`** yields **at most one** truth-table-checked candidate (`rag.pipeline === "circuit_synth"`). Remaining **`n−1`** slots: **`run_rag_first_variants`** (shared **`extract_intent_json`** + menu + compiler). |
| **`rag_first`** | No topology path — **`run_rag_first_variants`** only (optional **slot-template** first variant inside `circuit_rag_first`). |
| **`legacy`** | **`backend.generate(prompt, n)`** → `<|channel>thought` / DNA / `</circuit>` → **`parse_thought_and_sequence`** → **`apply_rag_substitution`** (equal chunks + Chroma + optional NCBI). |

Post-hoc **`apply_rag_substitution`** runs **only** for **legacy** candidates (and **`circuit_synth` / `rag_first` / `slot_template`** paths skip it — DNA is already registry-stitched).

---

## 4. End-to-end flow (web compiler)

1. **`POST /api/compile`** → `_compile` chooses **`compile_hybrid_variants`** vs **`run_rag_first_variants`** vs **`backend.generate`** (`server.py`).
2. Per candidate: optional **RAG substitution** (legacy only) → **`run_passes`** (`passes.py`) → **`score_candidate`** → **`attach_fidelity_scores`** (prompt alignment + pipeline tier).
3. **`rank`** — Pareto on four objectives; default order **`pipeline_tier`** → **`prompt_alignment`** → **`composite`**.
4. **`_finalize_compile_result`** — optional **`snapshot_id`** (`.design_snapshots/`).

Async jobs (`progress: true`): **`compile_hybrid_variants_iter`** / **`run_rag_first_variants_iter`** / **`generate_iter`** yield partial **`result`** with **`partial: true`** until all variants finish.

---

## 5. Module map

| Module | Responsibility |
|--------|----------------|
| `inference.py` | `.env` load; **`get_backend`**; **`GeminiBackend`** (`generateContent` / SSE `streamGenerateContent`; **`search_igem_registry`** tool loop via **`generate_text_gemma4_custom(..., igem_tools=True)`**); **`GGUFBackend`**; **`parse_thought_and_sequence`** / **`sanitize_thought_for_display`**. |
| `igem_rag.py` | JSONL → Chroma (`sentence-transformers` **`all-MiniLM-L6-v2`**); **`retrieve_parts`**; **`apply_rag_substitution`**; **`build_part_menu`** helpers; tool **`search_igem_registry_for_llm_tool`**. |
| `ncbi_gene.py` | Entrez gene → genomic slice → CDS FASTA; cache under chroma path; promoter slots gated by **`DGENE_NCBI_PROMOTER_SLOTS`**. |
| `circuit_ir.py` | **`CircuitSpec`**, **`LogicSpec`**, **`truth_table()`**. |
| `circuit_intent.py` | Hosted Gemma → strict JSON → **`CircuitSpec`** or skip. |
| `circuit_parts.py` | Curated promoters/TFs/backbone for **`circuit_synth`**. |
| `circuit_synth.py` | Deterministic assembly from catalog / **`data/igem_dataset.jsonl`**. |
| `circuit_verify.py` | Regulatory graph vs truth table. |
| `circuit_pipeline.py` | **`build_circuit_candidate`**, **`compile_hybrid_variants`** / **`_iter`**. |
| `circuit_rag_first.py` | **`extract_intent_json`**, **`build_part_menu`**, **`run_compiler`** (tools on), **`parse_ordered_bba`**, **`assemble_sequence`**, variant iterators + temps. |
| `slot_template_compile.py` | Deterministic Promoter/RBS/CDS/Terminator cassette when **`gate` / `input_analytes` / `reporter`** parse (**AND/OR/BUF**; not NOT). |
| `design_expert_lint.py` | Catalog promoter ↔ regulator CDS rules → **`rag.expert_lint`**. |
| `expert_review.py` | Optional Gemma JSON reviewer (**`DGENE_EXPERT_REVIEW`**) → **`rag.expert_review`** (hosted or GGUF). |
| `passes.py` | ORF, GC, repeats, Type IIS, restriction map, CAI, RBS, hairpins, biosecurity stub, parse labels. |
| `ranker.py` | Objectives, **`WEIGHTS`**, Pareto, **`pipeline_tier`**, **`prompt_alignment`**. |
| `server.py` | **`ThreadingHTTPServer`**; static **`web/`**; **`/api/compile`**, **`/api/compile/status`**, **`/api/fix`**, **`/api/snapshot`**, **`/api/health`**. |
| `app.py` | Streamlit: legacy **`run_inference`** + RAG + Bokeh map only. |
| `scripts/extract_igem_dataset.py` | `data/xml_parts.xml.gz` → **`data/igem_dataset.jsonl`** (filtered types, ACGT, length ≥ 40). |
| `scripts/generate_gemma_train.py` | Balanced sample → hosted Gemma reasoning → **`data/gemma_train.jsonl`** (channel-tagged assistant targets). |

---

## 6. Data pipeline

**Registry corpus**

- Input: **`data/xml_parts.xml.gz`** (iGEM parts table XML).
- **`scripts/extract_igem_dataset.py`**: tolerant gzip stream; rows → **`Promoter` / `RBS` / `CDS` / `Terminator`**; sequence hygiene (**≥40 bp**, no **`N`**, **[ACGT]** only).
- Output lines: `part_id`, `part_name`, `part_type`, `short_desc`, `sequence`.

**Fine-tune helper**

- **`scripts/generate_gemma_train.py`**: **`--sample-size`** divisible by 4; per-row **`generate_text_gemma4`** (PhD-style 2-sentence rationale); assistant format matches **`parse_thought_and_sequence`** (`<|channel>thought` … `<channel|>` … sequence).
- Repo stops at **`data/gemma_train.jsonl`**; training merge and **GGUF** export are **out-of-repo** (llama.cpp converters, Unsloth, etc.).

---

## 7. Retrieval and substitution

**Three mechanisms**

1. **Legacy post-hoc** — **`apply_rag_substitution`**: ordered **(type_hint, query)** from thought/part discovery → sequence split into **N equal chunks** → per-slot Chroma (+ optional NCBI); thresholds **`DGENE_RAG_MIN_SIM`**, **`DGENE_RAG_MIN_SIM_PROMOTER`** (promoters use the **max** of the two via **`min_similarity_for_slot`**).
2. **RAG-first menu** — **`build_part_menu`**: embed queries from **`intent`** (+flattened **`retrieval_queries`**); top‑**k** per query (**`DGENE_RAG_FIRST_TOP_K`**); dedupe → numbered menu.
3. **Mid-compile tools** — **`run_compiler`** calls **`generate_text_gemma4_custom(..., igem_tools=True)`** (unless **`DGENE_GEMINI_IGEM_TOOLS=0`**): Gemini **`functionCall`** → **`search_igem_registry`** → **`functionResponse`** (cap **`DGENE_GEMINI_TOOL_ROUNDS`**). Extra rows merge into **`menu_by_name`** for **`assemble_sequence`**.

**Chroma:** persistence **`DGENE_CHROMA_PATH`** (default `.chroma_igem`); corpus path **`DGENE_IGEM_JSONL`** (default repo-relative **`data/igem_dataset.jsonl`**). Registry token cache / NCBI cache may live under the same tree.

---

## 8. Topology compiler (`circuit_synth`)

- **`extract_circuit_spec`** → **`CircuitSpec`** with **`LOGIC_OPS`** = BUF, NOT, AND, OR, NAND, NOR over catalog-backed inputs/reporters.
- **`synthesize`** builds **`Plasmid`** from **`circuit_parts`** + **`data/igem_dataset.jsonl`**.
- **`verify_plasmid`** compares simulated outputs to **`truth_table()`**; failure ⇒ **no** topology candidate.
- **`rag` payload:** `verification.truth_table`, `backbone`, `circuit_spec`, **`pipeline`: `"circuit_synth"`**.

---

## 9. RAG-first and slot-template

**Steps (`circuit_rag_first`):**

1. **`extract_intent_json`** (Gemma via ``generate_text_gemma4_custom``, `_INTENT_SYSTEM`).
2. **`build_part_menu`** (+ **`ensure_indexed`**).
3. Optional **slot-template** variant first (**`slot_template_compile`**, toggles **`DGENE_SLOT_TEMPLATE`**, backbone embed **`DGENE_SLOT_TEMPLATE_EMBED_BACKBONE`**).
4. **`run_compiler`** at temperature ladder **`rag_first_candidate_temps`** → parse **`ORDERED_PART_LIST`** / **`parse_ordered_bba`** (guards against scanning free‑form **`BBa_`** in reasoning).
5. **`assemble_sequence`** with hard cap **`DGENE_RAG_FIRST_MAX_PARTS`**.

**Design QA:** **`lint_ordered_construct`** on ordered **`BBa_`**; optional **`expert_gemma_review`**.

---

## 10. Configuration reference

Authoritative comments and defaults: **`.env.example`**.

**API / generation**

- Keys & model: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `DGENE_GOOGLE_API_KEY`, `DGENE_GEMINI_MODEL`
- Timeouts / streaming: `DGENE_GEMINI_HTTP_TIMEOUT`, `DGENE_GEMINI_HTTP_HEARTBEAT_SEC`, `DGENE_GEMINI_STREAM`, `DGENE_GEMINI_STREAM_EARLY_CLOSE`, `DGENE_GEMINI_MAX_OUTPUT`, `DGENE_GEMINI_PARALLEL`, `DGENE_GEMINI_MAX_WORKERS`
- Tools: `DGENE_GEMINI_IGEM_TOOLS`, `DGENE_GEMINI_TOOL_ROUNDS`
- Parsing: `DGENE_MIN_PARSE_DNA_LEN`

**Compile pipeline**

- `DGENE_COMPILE_MODE` — `circuit_synth` | `rag_first` | `legacy`
- RAG-first: `DGENE_RAG_FIRST_TOP_K`, `DGENE_RAG_FIRST_MAX_PARTS`, `DGENE_RAG_FIRST_COMPILER_MAX_TOKENS`, `DGENE_RAG_FIRST_REASONING_CHARS`, `DGENE_RAG_FIRST_REASONING_SENTENCES`
- Slot-template: `DGENE_SLOT_TEMPLATE`, `DGENE_SLOT_TEMPLATE_MIN_SIM`, `DGENE_SLOT_TEMPLATE_MAX_PROMOTER_BP`, `DGENE_SLOT_TEMPLATE_EMBED_BACKBONE`

**RAG / corpus**

- `DGENE_IGEM_JSONL`, `DGENE_CHROMA_PATH`, `DGENE_RAG`, `DGENE_RAG_MIN_SIM`, `DGENE_RAG_MIN_SIM_PROMOTER`

**Local GGUF**

- `DGENE_GGUF_PATH`, `DGENE_INFERENCE`, `DGENE_GGUF_CTX`, `DGENE_GGUF_GPU_LAYERS`, `DGENE_GGUF_MAX_TOKENS`

**NCBI**

- `NCBI_API_KEY`, `DGENE_NCBI_API_KEY`, `DGENE_NCBI_EMAIL`, `DGENE_NCBI`, `DGENE_NCBI_ORGANISMS`, `DGENE_NCBI_PROMOTER_SLOTS`

**Server / UX**

- `PORT` — if unset locally, server tries **8765…8765+31**; if set (e.g. Railway), binds that port only.
- `DGENE_SNAPSHOTS`, `DGENE_EXPERT_REVIEW`, `DGENE_HTTP_LOG`, `DGENE_SERVER_DEBUG`, `DGENE_GEMINI_DEBUG`, `DGENE_RAG_DEBUG`, `DGENE_DEBUG`

---

## 11. Limitations and hackathon framing

- **Not wet-lab validated** — research tooling only.
- **Legacy RAG** uses **proportional chunking**; low named-part count ⇒ misaligned slots — use **`rag.parts`** / map **`*`** markers as hints.
- **Slot-template** lacks **`circuit_verify`** truth-table proof unless **`circuit_synth`** also succeeds.
- **NCBI** CDS spans ≠ promoter biology; **`DGENE_NCBI_PROMOTER_SLOTS`** defaults **off** for promoters.
- **`expert_lint` / `expert_review`** cover catalog rules / heuristic review, not full biology.
- **`app.py`** does **not** run **`circuit_pipeline`** / **`circuit_rag_first`** — use **`python3 server.py`** for full compilers.
- Hosted structured paths **do not** use **`GGUFBackend`** today; GGUF is primarily **legacy** generation (see **`get_backend`** / `_compile` branches).

---

## 12. Passes, ranking, and scores

**Passes** (`pass_ids`): `parse`, `orf`, `gc`, `repeats`, `type_iis`, `restrict`, `cai`, `rbs`, `hairpin`, …

**Objectives** (each in **[0,1]**): `expression`, `low_burden`, `gc_balance`, `cleanliness`.

**Composite:** `ranker.WEIGHTS` — expression **0.40**, low_burden **0.25**, gc_balance **0.15**, cleanliness **0.20**.

**`pipeline_tier`:** `circuit_synth` **3** > `slot_template` **2** > `rag_first` **1** > legacy **0**.

**Pareto:** dominance **only** on the four objectives (not on pipeline tier).

---

## 13. HTTP API (`server.py`)

| Method | Path | Notes |
|--------|------|------|
| POST | `/api/compile` | Body: `prompt`, `n` ∈ **[1,8]** (default 4), optional `progress`. Sync JSON **or** **`202`** + `{job_id}` if `progress: true`. |
| GET | `/api/compile/status` | `job_id` — `done`, `lines`, optional `streams`, `result` (may include **`partial`**). |
| GET | `/api/health` | Backend kind + model id / GGUF name. |
| POST | `/api/fix` | `original_prompt`, `current_sequence`, `fix_type` (`repeats` \| `type_iis` \| `cai` \| `rbs` \| `repeats_type_iis`), `candidates`, optional `source_candidate_id`. |
| GET | `/api/snapshot` | `id` — replay snapshot JSON (**503** if disabled). |

**`/api/fix`** builds constraint text from **`FIX_PROMPTS`**, runs **`_compile(..., n=1)`** with **`user_prompt_for_alignment`** pinned to the original brief, merges + re-ranks.

Static assets: **`/`** → `web/index.html`; same-origin **`/api/*`**.

---

## 14. Repository layout

| Path | Role |
|------|------|
| `server.py`, `web/` | Production compiler UI + APIs |
| `inference.py` | All Gemma I/O and backends |
| `igem_rag.py`, `ncbi_gene.py` | Retrieval + substitution |
| `circuit_*.py`, `slot_template_compile.py` | IR → synthesis → verify → hybrid / RAG-first |
| `design_expert_lint.py`, `expert_review.py` | QA passes |
| `passes.py`, `ranker.py` | Metrics + ordering |
| `app.py` | Streamlit legacy demo |
| `data/igem_dataset.jsonl` | Parts corpus |
| `data/gemma_train.jsonl` | Generated training data (optional artifact) |
| `scripts/extract_igem_dataset.py`, `scripts/generate_gemma_train.py` | Dataset builders |
| `Procfile`, `railway.toml`, `nixpacks.toml`, `runtime.txt` | Deploy hints (Railway: **`python server.py`**, health **`GET /api/health`**, RAM **≥ ~2 GB** for first embed). |

Gitignored / runtime: `.chroma_igem/`, `.design_snapshots/`, caches.

---

## Demo URL and model artifact

**Live demo:** *(paste deployment URL here.)*

**Fine-tuned GGUF:** **[davidgeorge25/opengenedit-gemma-4-31b](https://huggingface.co/davidgeorge25/opengenedit-gemma-4-31b)** — download **`dgene-q4km.gguf`** (Q4_K_M); base **`google/gemma-4-31B-it`**. Set **`DGENE_GGUF_PATH`** to the downloaded file; **`DGENE_INFERENCE=gguf`** to bypass hosted API when both keys and GGUF exist. The same file runs under standalone **`llama-server -m …`** / **`-hf davidgeorge25/opengenedit-gemma-4-31b`** (recent llama.cpp; Gemma 4 support). OpenGeneEdit loads it via **`llama-cpp-python`** in-process — see **README → Quick start**.
