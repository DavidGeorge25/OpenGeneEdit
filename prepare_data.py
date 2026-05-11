#!/usr/bin/env python3
"""Prepare corpus and optional test prompts (Kaggle-style ``prepare_data`` entry).

Reads paths only from ``SETTINGS.json`` (via :mod:`submission_settings`), then:

1. Runs ``scripts/extract_igem_dataset.py`` when ``data/xml_parts.xml.gz`` exists
   (builds ``data/igem_dataset.jsonl`` in the repo layout used by extract script).

2. Ensures ``TEST_DATA_CLEAN_PATH`` parent directory exists.

Chroma embeddings are built lazily on first compile when the server imports ``igem_rag``;
no separate batch step is required for retrieval to function.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from submission_settings import apply_settings, load_settings


def main() -> int:
    apply_settings()
    cfg = load_settings()
    root = Path(__file__).resolve().parent
    raw_gz = root / "data" / "xml_parts.xml.gz"
    extract = root / "scripts" / "extract_igem_dataset.py"
    if raw_gz.is_file():
        print(f"[prepare_data] Running {extract} …")
        r = subprocess.run([sys.executable, str(extract)], cwd=str(root))
        if r.returncode != 0:
            return r.returncode
    else:
        print(
            "[prepare_data] Skip extract: data/xml_parts.xml.gz not found "
            "(using committed data/igem_dataset.jsonl if present).",
            file=sys.stderr,
        )

    test_rel = cfg.get("TEST_DATA_CLEAN_PATH", "data/test_prompts.jsonl")
    test_path = Path(test_rel)
    if not test_path.is_absolute():
        test_path = root / test_path
    test_path.parent.mkdir(parents=True, exist_ok=True)
    if not test_path.is_file():
        sample = root / "data" / "test_prompts.example.jsonl"
        if sample.is_file():
            test_path.write_text(sample.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[prepare_data] Wrote default prompts to {test_path} from example.")
        else:
            print(
                f"[prepare_data] Create {test_path} (JSONL: one object per line with "
                f'"prompt" key) or add data/test_prompts.example.jsonl.',
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
