#!/usr/bin/env python3
"""Training-data and fine-tune artifact layout (Kaggle-style ``train`` entry).

This repository’s **primary** generative model weights are **not** trained inside this
tree by default: hosted **Gemma 4** is called over HTTPS, or a **GGUF** is loaded from
``SETTINGS.json`` / ``DGENE_GGUF_PATH``. Optional **LoRA** experiment outputs may live
under ``MODEL_DIR`` from external **Unsloth** / Hugging Face workflows.

This script:

1. Applies ``SETTINGS.json`` paths to the environment.

2. Optionally builds ``TRAIN_DATA_CLEAN_PATH`` by invoking ``scripts/generate_gemma_train.py``
   (requires ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` and network access).

Pass ``--skip-gemma-jsonl`` to only validate paths and print instructions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from submission_settings import apply_settings, load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenGeneEdit train / data-prep entry.")
    parser.add_argument(
        "--skip-gemma-jsonl",
        action="store_true",
        help="Do not call generate_gemma_train.py (no API usage).",
    )
    args = parser.parse_args()

    apply_settings()
    cfg = load_settings()
    root = Path(__file__).resolve().parent
    train_out = cfg.get("TRAIN_DATA_CLEAN_PATH", "data/gemma_train.jsonl")
    train_path = Path(train_out)
    if not train_path.is_absolute():
        train_path = root / train_path

    if args.skip_gemma_jsonl:
        print(
            "[train] Skipped JSONL generation. For supervised JSONL run:\n"
            f"  {sys.executable} {root / 'scripts' / 'generate_gemma_train.py'} "
            f"--output {train_path}\n"
            "External LoRA → merge → GGUF: see docs/MODEL_SUMMARY.md §A5 and README."
        )
        return 0

    gen = root / "scripts" / "generate_gemma_train.py"
    if not gen.is_file():
        print(f"[train] Missing {gen}", file=sys.stderr)
        return 1
    train_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(gen), "--output", str(train_path)]
    print("[train] ", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(root))
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
