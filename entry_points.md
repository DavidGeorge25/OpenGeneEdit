# Entry points (reproduction)

All path roots are defined in **`SETTINGS.json`** at the repository root. Do **not**
hard-code corpus or output directories in custom wrappers; extend **`SETTINGS.json`**
and, if needed, **`submission_settings.py`**.

## 1. Environment

```bash
cd /path/to/DGene
python3 -m venv .venv && source .venv/bin/activate   # optional
python3 -m pip install -r requirements.txt
cp .env.example .env
# Edit .env: set GEMINI_API_KEY or GOOGLE_API_KEY and DGENE_GEMINI_MODEL, and/or
# DGENE_GGUF_PATH + DGENE_INFERENCE=gguf (see README.md).
```

## 2. Prepare data

Build **`data/igem_dataset.jsonl`** when **`data/xml_parts.xml.gz`** is present; ensure
test prompt file exists (see **`data/test_prompts.example.jsonl`**).

```bash
python3 prepare_data.py
```

Underlying script (also runnable directly): **`python3 scripts/extract_igem_dataset.py`**.

## 3. Train / training artifacts

**Hosted JSONL for external SFT** (calls the same Gemma API stack as production inference):

```bash
python3 train.py
```

Skip API calls (documentation-only dry run message):

```bash
python3 train.py --skip-gemma-jsonl
```

Direct invocation: **`python3 scripts/generate_gemma_train.py --output <path>`**.

**LoRA checkpoints and merged weights** are expected under **`MODEL_DIR`** in
**`SETTINGS.json`** (default **`finetune_results/gemma_4_lora/`**) when you reproduce an
external Unsloth/PEFT workflow; those large binaries are **gitignored** — obtain from
your training run or from published **GGUF** artifacts (see **`README.md`**).

## 4. Predict / batch compile

Input path: **`TEST_DATA_CLEAN_PATH`** in **`SETTINGS.json`** (default
**`data/test_prompts.jsonl`**). Output: **`SUBMISSION_DIR`** (default **`outputs/`**).

```bash
python3 predict.py
```

## 5. Interactive web compiler (alternate prediction path)

```bash
python3 server.py
```

Then use **`POST /api/compile`** as documented in **`docs/HACKATHON_TECHNICAL.md`**.

## 6. Legacy Streamlit demo

```bash
streamlit run app.py
```

**Note:** **`app.py`** exercises the **legacy** path only, not the full hybrid compiler.
