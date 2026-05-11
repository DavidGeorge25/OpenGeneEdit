# Configuration files (Kaggle submission component B3)

This folder holds **documentation** for configuration that is normally **not** committed
because it contains secrets or machine-local paths.

## Environment contract

- **Primary runtime configuration** is via a **`.env`** file at the repository root
  (see **`.env.example`** in the repo root for every supported variable). Copy
  **`.env.example`** to **`.env`** and set at least **`GEMINI_API_KEY`** or
  **`GOOGLE_API_KEY`**, or point **`DGENE_GGUF_PATH`** at a local **`.gguf`** file.

- **`SETTINGS.json`** (repo root) is the **only** canonical place for **data and I/O
  directory paths** used by **`prepare_data.py`**, **`train.py`**, and **`predict.py`**.
  Those scripts call **`submission_settings.apply_settings()`**, which maps JSON keys
  into **`DGENE_IGEM_JSONL`**, **`DGENE_CHROMA_PATH`**, etc., before other modules load.

## No `keras.json`

OpenGeneEdit does **not** use Keras/TensorFlow for inference. There is no
**`$HOME/.keras/keras.json`** requirement.

## Optional: deploy / platform files

Railway-oriented hints live at the repo root (**`railway.toml`**, **`nixpacks.toml`**,
**`Procfile`**, **`runtime.txt`**) and are described in the main **`README.md`**.
