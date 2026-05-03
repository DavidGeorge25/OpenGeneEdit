# OpenGeneEdit

**OpenGeneEdit** is a synthetic-biology–oriented DNA “compiler”: you describe a genetic circuit in natural language and get structured reasoning, candidate sequences, iGEM-aware retrieval and audits, heuristic compiler passes, multi-objective ranking (including a Pareto-style front), plasmid visualization, and FASTA / GenBank export.

**Full technical reference (architecture, RAG, APIs, env vars, limitations):** [`HACKATHON_TECHNICAL.md`](HACKATHON_TECHNICAL.md)

**Naming.** Product branding is **OpenGeneEdit**. Stderr tags use **`oge`** (e.g. `[oge/server]`). Configuration keys keep the **`DGENE_*`** prefix so existing `.env` files stay valid.

---

## Architecture summary

Inference is **Google Gemma 4 only**: **Gemini API** (stdlib `urllib` in `inference.py`) or local **GGUF** via [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). Hosted Gemma is required for **boolean intent extraction**, **RAG-first intent/menu compilers**, **`/api/fix`**, and optional **`expert_review`**; GGUF applies to **legacy** channel DNA generation when configured (`DGENE_INFERENCE`).

### Compile modes (`DGENE_COMPILE_MODE`)

| Mode | Behaviour |
|------|-----------|
| **`circuit_synth`** (default) | **Hybrid:** Gemma extracts boolean **`circuit_intent`** JSON → when **`applicable`**, **`circuit_ir`** + **`circuit_synth`** + **`circuit_verify`** emit a **truth-table-checked** linear plasmid (`rag.pipeline === circuit_synth`). Remaining slots use **RAG-first** (shared biological intent JSON + Chroma menu + Gemma **`ORDERED_PART_LIST`** compiler). Optional **`slot_template`** cassette may lead RAG-first variants when **`gate` / `input_analytes` / `reporter`** parse. |
| **`rag_first`** | Intent JSON → **`build_part_menu`** → menu-constrained compiler → **`assemble_sequence`** (registry DNA only). |
| **`legacy`** | Gemma emits `<|channel>thought` + DNA + `</circuit>` → **`parse_thought_and_sequence`** → **`apply_rag_substitution`** (equal-chunk slots + Chroma + optional **NCBI Gene**). |

If **`circuit_synth`** or **`rag_first`** is selected but **no** `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set, **`server.py` falls back to legacy** and logs a warning.

### iGEM data & RAG

- **Corpus:** `igem_dataset.jsonl` (from [`extract_igem_dataset.py`](extract_igem_dataset.py) + optional `xml_parts.xml.gz`).
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

[`generate_gemma_train.py`](generate_gemma_train.py) builds **`gemma_train.jsonl`** from `igem_dataset.jsonl` for external SFT/LoRA → GGUF workflows (see technical doc §3).

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

Create **`.env`** at the repo root (optional; loaded in `inference.py`, **does not override** existing environment variables). See **[`.env.example`](.env.example)** and **§10 of [`HACKATHON_TECHNICAL.md`](HACKATHON_TECHNICAL.md)** for the full variable list (`DGENE_COMPILE_MODE`, RAG-first knobs, slot-template, NCBI, snapshots, streaming, etc.).

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
| [`igem_dataset.jsonl`](igem_dataset.jsonl) | Registry-derived parts corpus |
| [`extract_igem_dataset.py`](extract_igem_dataset.py), [`generate_gemma_train.py`](generate_gemma_train.py) | Dataset / training JSONL builders |
| [`railway.toml`](railway.toml), [`nixpacks.toml`](nixpacks.toml), [`Procfile`](Procfile), [`runtime.txt`](runtime.txt) | Railway / Nixpacks deploy hints |

Gitignored / generated: `.chroma_igem/`, `.design_snapshots/`, `finetune_results/`, etc.

---

## Limitations

Outputs are **not** wet-lab validated. Legacy RAG uses **similarity and proportional chunks** — treat map **`*`** markers and **`rag.parts`** as cues. **Slot-template** does not replace **`circuit_verify`** truth-table proofs unless **`circuit_synth`** also applies. See **§11** in [`HACKATHON_TECHNICAL.md`](HACKATHON_TECHNICAL.md).

---

## License / data

Use of the Gemini API, local model weights, and iGEM-derived data must comply with their respective terms. This repo is tooling and research-oriented; it is **not** a substitute for lab validation or safety review.
