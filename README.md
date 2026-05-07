# OpenGeneEdit

## Gemma 4 implementation (for reviewers)

Trace **hosted Gemma 4** and **optional local GGUF** in **[`inference.py`](inference.py)**:

| Entry point | Role |
|-------------|------|
| **`generate_text_gemma4`** / **`generate_text_gemma4_custom`** | Google Generative Language API (`DGENE_GEMINI_MODEL`, default **`gemma-4-31b-it`**); retries, streaming, tool payloads. |
| **`_gemini_generate_custom_with_igem_tools`** | Multi-turn **`functionCall`** / **`functionResponse`** loop with **`search_igem_registry`** during compile. |
| **`get_backend`** → **`run_inference`** | Resolves **`DGENE_INFERENCE`** (`auto` chooses Gemini vs **`llama-cpp-python`** when **`DGENE_GGUF_PATH`** is set). |
| **`parse_thought_and_sequence`** | Parses channel-tagged model output (`<|channel>thought` … `</circuit>`) used in legacy and training formats. |

Supervised JSONL for external SFT / LoRA → GGUF: **[`scripts/generate_gemma_train.py`](scripts/generate_gemma_train.py)** (same **`generate_text_gemma4`** API path as the live compiler).

**Architecture & APIs:** [`docs/HACKATHON_TECHNICAL.md`](docs/HACKATHON_TECHNICAL.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Quick start: fine-tuned model only (no Gemini API)

Use the OpenGeneEdit **GGUF** on Hugging Face instead of `GEMINI_API_KEY`. Steps:

1. **Download the weights**  
   Open **[davidgeorge25/opengenedit-gemma-4-31b](https://huggingface.co/davidgeorge25/opengenedit-gemma-4-31b)** → **Files and versions** → download **`dgene-q4km.gguf`** (Q4_K_M, based on `google/gemma-4-31B-it`).

2. **Install Python deps** (repo root)

   ```bash
   python3 -m pip install -r requirements.txt
   python3 -m pip install llama-cpp-python
   ```

3. **Create `.env`** next to `server.py` with:

   ```bash
   DGENE_GGUF_PATH=/absolute/path/to/dgene-q4km.gguf
   DGENE_INFERENCE=gguf
   DGENE_COMPILE_MODE=legacy
   ```

   - **`DGENE_INFERENCE=gguf`** forces local inference even if a Gemini/Google API key is set elsewhere on your machine.  
   - **`DGENE_COMPILE_MODE=legacy`** runs the path where **your GGUF** generates `<|channel>thought` + DNA + `</circuit>`, then iGEM RAG can substitute slots (`apply_rag_substitution`).  
     Topology/RAG-first modes need the **hosted** API; without it they fall back to legacy anyway — setting **`legacy`** avoids extra warnings.

4. **Start the compiler UI**

   ```bash
   python3 server.py
   ```

   Open the printed URL (often **`http://127.0.0.1:8765/`**).

**Hardware.** A **31B** quantized model is large; on CPU-only boxes expect slow generations. If `llama-cpp-python` is built with CUDA or Metal, you can offload layers via **`DGENE_GGUF_GPU_LAYERS`** (see **[`.env.example`](.env.example)**).

### Same GGUF with upstream llama.cpp (`llama-server` / `llama-cli`)

The file **`dgene-q4km.gguf`** you downloaded is a standard **GGUF** — it is the same artifact **[llama.cpp](https://github.com/ggerganov/llama.cpp)** loads. You can host it locally **without Python**:

```bash
# Install llama.cpp (pick one): brew, winget, or a release binary from the llama.cpp repo
# Then, using your downloaded file:
llama-server -m /absolute/path/to/dgene-q4km.gguf

# Or let llama.cpp fetch from Hugging Face (same repo / weights as above):
llama-server -hf davidgeorge25/opengenedit-gemma-4-31b
```

Use a **recent** llama.cpp build so **Gemma 4** is supported. The terminal prints a **local URL** (often with a small web UI and an OpenAI-compatible HTTP API) for chatting or testing the fine-tune outside OpenGeneEdit.

**OpenGeneEdit compiler:** `python3 server.py` uses **`llama-cpp-python`** (llama.cpp under the hood) with **`DGENE_GGUF_PATH`** — it loads that **same `.gguf` in-process**. It does **not** call `llama-server` today; use **`DGENE_GGUF_PATH` + `DGENE_INFERENCE=gguf`** for compiles in the web UI, and use **`llama-server`** separately if you want a standalone local host for the identical weights.

---

**OpenGeneEdit** is a synthetic-biology–oriented DNA “compiler”: you describe a genetic circuit in natural language and get structured reasoning, candidate sequences, iGEM-aware retrieval and audits, heuristic compiler passes, multi-objective ranking (including a Pareto-style front), plasmid visualization, and FASTA / GenBank export.

**Full technical reference (architecture, RAG, APIs, env vars, limitations):** [`docs/HACKATHON_TECHNICAL.md`](docs/HACKATHON_TECHNICAL.md)

**Naming.** Product branding is **OpenGeneEdit**. Stderr tags use **`oge`** (e.g. `[oge/server]`). Configuration keys keep the **`DGENE_*`** prefix so existing `.env` files stay valid.

---

## Architecture summary

Inference is **Google Gemma 4 only**: **Gemini API** (stdlib `urllib` in `inference.py`) or local **GGUF** via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). Hosted Gemma is required for **boolean intent extraction**, **RAG-first intent/menu compilers**, **`/api/fix`**, and optional **`expert_review`**; GGUF applies to **legacy** channel DNA generation when configured (`DGENE_INFERENCE`).

### Architecture diagrams

**Tool calling** — compile path with `search_igem_registry`, declared Gemini tools, Chroma-backed hits, and iterative `functionResponse` rounds until the model returns final reasoning (`ORDERED_PART_LIST`, etc.):

![OpenGeneEdit: Gemini tool-calling loop with iGEM registry tools](docs/diagrams/dgene_tool_calling.png)

**RAG retrieval** — natural-language brief → intent JSON (`circuit_rag_first.extract_intent_json`) → search phrases → `igem_rag.retrieve_parts` / embedding search over **`data/igem_dataset.jsonl`** (sentence-transformers + Chroma) → numbered parts menu for the planner:

![OpenGeneEdit: retrieval and embedding search pipeline](docs/diagrams/dgene_retrieval_search.png)

### Compile modes (`DGENE_COMPILE_MODE`)

| Mode | Behaviour |
|------|-----------|
| **`circuit_synth`** (default) | **Hybrid:** Gemma extracts boolean **`circuit_intent`** JSON → when **`applicable`**, **`circuit_ir`** + **`circuit_synth`** + **`circuit_verify`** emit a **truth-table-checked** linear plasmid (`rag.pipeline === circuit_synth`). Remaining slots use **RAG-first** (shared biological intent JSON + Chroma menu + Gemma **`ORDERED_PART_LIST`** compiler). Optional **`slot_template`** cassette may lead RAG-first variants when **`gate` / `input_analytes` / `reporter`** parse. |
| **`rag_first`** | Intent JSON → **`build_part_menu`** → menu-constrained compiler → **`assemble_sequence`** (registry DNA only). |
| **`legacy`** | Gemma emits `<|channel>thought` + DNA + `</circuit>` → **`parse_thought_and_sequence`** → **`apply_rag_substitution`** (equal-chunk slots + Chroma + optional **NCBI Gene**). |

If **`circuit_synth`** or **`rag_first`** is selected but **no** `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set, **`server.py` falls back to legacy** and logs a warning.

### iGEM data & RAG

- **Corpus:** [`data/igem_dataset.jsonl`](data/igem_dataset.jsonl) (from [`scripts/extract_igem_dataset.py`](scripts/extract_igem_dataset.py) + optional [`data/xml_parts.xml.gz`](data/xml_parts.xml.gz)).
- **Embeddings:** ChromaDB + **`sentence-transformers`** (`all-MiniLM-L6-v2`), persisted under **`DGENE_CHROMA_PATH`** (default `.chroma_igem`).
- **Two retrieval paths:** (1) **Legacy post-hoc** **`apply_rag_substitution`** — proportional chunks + similarity (**promoters** use **`DGENE_RAG_MIN_SIM_PROMOTER`** vs **`DGENE_RAG_MIN_SIM`**). (2) **RAG-first** — retrieve **before** the compiler; DNA from menu/`ORDERED_PART_LIST` only (details in §5–5.10 of the technical doc).
- **NCBI fallback:** **`ncbi_gene.py`** (Entrez) for CDS-shaped symbols when iGEM does not verify; promoter slots default **`DGENE_NCBI_PROMOTER_SLOTS=0`**.

### Passes, ranking, QA

- **`passes.py`** — ORF, GC, repeats, Type IIS, restriction map, E. coli CAI, RBS heuristic, hairpins, etc.
- **`ranker.py`** — Four objectives + **Pareto**; default **`best_id`** order: **`pipeline_tier`** (`circuit_synth` > `slot_template` > `rag_first` > legacy) → **`prompt_alignment`** → **composite** (weights in **`ranker.WEIGHTS`**).
- **`design_expert_lint.py`** — Promoter ↔ cognate regulator rules on ordered **`BBa_`** lists → **`rag.expert_lint`**.
- **`expert_review.py`** — Optional second Gemma pass when **`DGENE_EXPERT_REVIEW=1`** → **`rag.expert_review`**.
- **Snapshots:** **`GET /api/snapshot?id=…`** when **`DGENE_SNAPSHOTS`** enabled (`.design_snapshots/`, gitignored).

### Web compiler (`server.py` + `web/`)

- **Not Flask/FastAPI** — **`ThreadingHTTPServer`** serves **`/`** → `web/index.html`, **`/css/*`**, **`/js/*`**, other static assets under `web/`, plus **`/api/*`** on the **same origin**.
- **Endpoints:** `POST /api/compile` (sync or **`progress: true`** → **`202`** + poll **`/api/compile/status`**), **`GET /api/health`**, **`POST /api/fix`**, **`GET /api/snapshot`**, CORS headers on responses.
- **`PORT`:** If **`PORT`** is set (Railway, etc.), the server binds **only** that port. If **unset**, local default **`8765`** with fallback to the next free port.

### Streamlit (`app.py`)

**Legacy path only** — single **`run_inference`** + **`apply_rag_substitution`** + Bokeh map. Does **not** run **`circuit_pipeline`** / **`circuit_rag_first`**. Use **`python3 server.py`** for topology or menu compilers.

### Fine-tuning helper

[`scripts/generate_gemma_train.py`](scripts/generate_gemma_train.py) builds **[`data/gemma_train.jsonl`](data/gemma_train.jsonl)** from **`data/igem_dataset.jsonl`** for external SFT/LoRA → GGUF workflows (see technical doc §3).

```bash
python3 scripts/extract_igem_dataset.py   # data/xml_parts.xml.gz → data/igem_dataset.jsonl
python3 scripts/generate_gemma_train.py    # hosted Gemma 4 → data/gemma_train.jsonl (needs API key)
```

---

## Requirements

- **Python 3.10+** recommended (Streamlit/Bokeh path); **`runtime.txt`** pins **3.11.8** for Railway/Nixpacks.

Install RAG / embedding dependencies for the full **`server.py`** pipeline:

```bash
python3 -m pip install -r requirements.txt
```

| Goal | Packages |
|------|----------|
| Web compiler (`server.py`) | `requirements.txt` (Chroma + sentence-transformers; pulls large transitive deps e.g. **torch** — plan **~2 GB+ RAM** for first embed index) |
| Streamlit (`app.py`) | `streamlit`, `bokeh`, `pandas`, `dna-features-viewer` |
| Hosted Gemma | API key only (no Gemini SDK; **`urllib`** in `inference.py`) |
| Local GGUF | `llama-cpp-python` for your platform |

```bash
python3 -m pip install streamlit bokeh pandas dna-features-viewer
```

---

## Configuration

Create **`.env`** at the repo root (optional; loaded in `inference.py`, **does not override** existing environment variables). See **[`.env.example`](.env.example)** and **§10 of [`docs/HACKATHON_TECHNICAL.md`](docs/HACKATHON_TECHNICAL.md)** for the full variable list (`DGENE_COMPILE_MODE`, RAG-first knobs, slot-template, NCBI, snapshots, streaming, etc.).

**Minimum for hosted demo:** `GEMINI_API_KEY` or `GOOGLE_API_KEY`, and usually `DGENE_GEMINI_MODEL` (e.g. `gemma-4-31b-it`).

Restart **`server.py`** after changing `.env`.

---

## Running

**Compiler server (recommended — full pipeline + static UI)**

```bash
python3 server.py
```

When **`PORT` is unset**, opens at **`http://127.0.0.1:8765/`** (or next free port). When **`PORT` is set**, listens on **`0.0.0.0:PORT`** only.

### Railway (single URL for judges)

Ships **`Procfile`**, **`railway.toml`**, **`nixpacks.toml`**, and **`runtime.txt`**. Connect the GitHub repo and set **`GEMINI_API_KEY`** / **`GOOGLE_API_KEY`** (and optional **`DGENE_GEMINI_MODEL`**) in Railway Variables.

- **Process:** `python server.py` (**not** `app.py` / Streamlit — Nixpacks otherwise auto-starts `app.py`). If logs show `ModuleNotFoundError: streamlit`, open **Service → Settings → Deploy → Custom Start Command** and set **`python server.py`**.
- **Stack:** stdlib HTTP (**not** gunicorn — not WSGI).
- **Health check:** `GET /api/health`
- **RAM:** Prefer **≥ 2 GB**; first Chroma index build is heavy.
- **Disk:** `.chroma_igem/` and snapshots are **ephemeral** across redeploys — first compile after deploy may be slower.

Public URL: **`https://<service>.up.railway.app/`** serves both **`/`** and **`/api/*`**.

### Streamlit playground

```bash
streamlit run app.py
```

---

## Repository layout

| Path | Role |
|------|------|
| [`server.py`](server.py) | `ThreadingHTTPServer`, static `web/`, `/api/*` |
| [`web/`](web/) | Compiler UI (map, candidates, RAG audit, exports) |
| [`inference.py`](inference.py) | Gemma 4 backends, parsing, `.env` load |
| [`igem_rag.py`](igem_rag.py) | Chroma index, retrieval, legacy substitution, menu retrieval |
| [`ncbi_gene.py`](ncbi_gene.py) | NCBI Entrez CDS fallback + cache |
| [`circuit_ir.py`](circuit_ir.py) … [`circuit_pipeline.py`](circuit_pipeline.py) | IR, intent, parts catalog, synthesis, verification, hybrid orchestration |
| [`circuit_rag_first.py`](circuit_rag_first.py), [`slot_template_compile.py`](slot_template_compile.py) | RAG-first menu compiler + optional slot-template cassette |
| [`design_expert_lint.py`](design_expert_lint.py), [`expert_review.py`](expert_review.py) | Regulatory lint + optional Gemma reviewer |
| [`passes.py`](passes.py), [`ranker.py`](ranker.py) | Diagnostics + Pareto / ranking |
| [`app.py`](app.py) | Streamlit legacy playground |
| [`data/igem_dataset.jsonl`](data/igem_dataset.jsonl), [`data/gemma_train.jsonl`](data/gemma_train.jsonl), [`data/xml_parts.xml.gz`](data/xml_parts.xml.gz) | Registry corpus, optional training JSONL, optional raw iGEM XML export |
| [`scripts/extract_igem_dataset.py`](scripts/extract_igem_dataset.py), [`scripts/generate_gemma_train.py`](scripts/generate_gemma_train.py) | Dataset / Gemma 4 training JSONL builders |
| [`docs/HACKATHON_TECHNICAL.md`](docs/HACKATHON_TECHNICAL.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/diagrams/`](docs/diagrams/) | Technical spec, architecture write-up, diagram sources + renders |
| [`railway.toml`](railway.toml), [`nixpacks.toml`](nixpacks.toml), [`Procfile`](Procfile), [`runtime.txt`](runtime.txt) | Railway / Nixpacks deploy hints |

Gitignored / generated: `.chroma_igem/`, `.design_snapshots/`, `finetune_results/`, etc.

---

## Limitations

Outputs are **not** wet-lab validated. Legacy RAG uses **similarity and proportional chunks** — treat map **`*`** markers and **`rag.parts`** as cues. **Slot-template** does not replace **`circuit_verify`** truth-table proofs unless **`circuit_synth`** also applies. See **§11** in [`docs/HACKATHON_TECHNICAL.md`](docs/HACKATHON_TECHNICAL.md).

---

## License / data

Use of the Gemini API, local model weights, and iGEM-derived data must comply with their respective terms. This repo is tooling and research-oriented; it is **not** a substitute for lab validation or safety review.
