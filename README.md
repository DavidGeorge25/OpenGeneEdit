# OpenGeneEdit

**OpenGeneEdit** is a synthetic-biology oriented DNA “compiler”: you describe a genetic circuit in natural language, and the stack produces structured reasoning plus a nucleotide sequence, optional verification against real [iGEM](https://igem.org) registry parts (RAG), compiler-style checks, ranking, plasmid visualization, and FASTA / GenBank export.

Configuration env vars still use the **`DGENE_*`** prefix for backwards compatibility.

Inference is powered by **Google Gemma 4 only** — either via the Gemini API or a local **GGUF** model.

## Features

- **Hosted or local inference** — `GEMINI_API_KEY` / `GOOGLE_API_KEY` (Gemini API) or `DGENE_GGUF_PATH` + [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) for quantized Gemma on your machine.
- **Web compiler** — `python3 server.py` serves a UI under `web/` and a **`/api/compile`** pipeline (multi-candidate inference → RAG → passes → Pareto-style ranking). Long compiles support **async progress** via job polling (`progress: true`).
- **Streamlit playground** — `streamlit run app.py` for a quick single-shot flow with an interactive circular plasmid map (Bokeh + DNA Features Viewer).
- **iGEM RAG** — Retrieval over `igem_dataset.jsonl`; ChromaDB + sentence embeddings can substitute verified registry sequences when similarity exceeds a threshold.

## Requirements

- **Python 3.10+** recommended (Streamlit + Bokeh stack; see `app.py` for compatibility notes).

Install RAG dependencies (also used when the compile server enables substitution):

```bash
python3 -m pip install -r requirements.txt
```

Additional installs depend on how you run the app:

| Goal | Typical packages |
|------|------------------|
| Web server (`server.py`) | stdlib-only for HTTP; inference deps per backend below |
| Streamlit (`app.py`) | `streamlit`, `bokeh`, `pandas`, `dna-features-viewer` |
| Hosted Gemma | API key only (HTTP via stdlib `urllib` in `inference.py`) |
| Local GGUF | `llama-cpp-python` matching your Python / platform |

Example one-liner for the Streamlit UI:

```bash
python3 -m pip install streamlit bokeh pandas dna-features-viewer
```

## Configuration

Create a `.env` in the repo root (optional; loaded on import, does not override existing env vars):

**Inference**

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `DGENE_GOOGLE_API_KEY` | Gemini API authentication |
| `DGENE_GEMINI_MODEL` | e.g. `gemma-4-31b-it` |
| `DGENE_GGUF_PATH` | Path to a Gemma `.gguf` file for local inference |
| `DGENE_INFERENCE` | `auto` (default), `gemini` / `hosted`, or `gguf` / `local` |

**Debugging**

| Variable | Purpose |
|----------|---------|
| `DGENE_GEMINI_DEBUG` | Gemini HTTP / retry traces on stderr |
| `DGENE_DEBUG`, `DGENE_RAG_DEBUG` | Verbose RAG logging (see `igem_rag.py`) |

**iGEM RAG** (optional; see `igem_rag.py`)

| Variable | Purpose |
|----------|---------|
| `DGENE_IGEM_JSONL` | Override path to registry JSONL (default: `./igem_dataset.jsonl`) |
| `DGENE_CHROMA_PATH` | Chroma persistence directory (default: `.chroma_igem`) |
| `DGENE_RAG` | Set `0` / `false` to disable substitution |
| `DGENE_RAG_MIN_SIM` | Minimum cosine similarity for substitution (default `0.6`) |

Restart the server after changing `.env`.

## Running

**Compiler server (recommended full pipeline)**

```bash
python3 server.py
```

Opens at `http://127.0.0.1:8765/` by default (or next free port). Override with `PORT`.

**Streamlit**

```bash
streamlit run app.py
```

## Repository layout

| Path | Role |
|------|------|
| `server.py` | Threading HTTP server, `/api/compile`, static `web/` |
| `inference.py` | Gemma 4 backends and parsing |
| `igem_rag.py` | ChromaDB + embeddings RAG layer |
| `passes.py` / `ranker.py` | Candidate diagnostics and scoring |
| `app.py` | Streamlit frontend |
| `igem_dataset.jsonl` | iGEM-derived parts corpus for RAG |
| `extract_igem_dataset.py` | Optional: build JSONL from `xml_parts.xml.gz` |

Transient / generated paths are gitignored (`finetune_results/`, `.chroma_igem/`).

## License / data

Ensure your use of the Gemini API, local model weights, and iGEM-derived data complies with their respective terms. This repo is tooling and research-oriented; outputs are **not** a substitute for lab validation or safety review.
