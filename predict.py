#!/usr/bin/env python3
"""Batch compile prompts → JSON predictions (Kaggle-style ``predict`` entry).

Reads ``TEST_DATA_CLEAN_PATH`` from ``SETTINGS.json`` (JSONL: one JSON object per line
with a ``"prompt"`` string). Writes one JSON file per line into ``SUBMISSION_DIR`` named
``predict_000001.json``, … plus ``predict_manifest.jsonl`` with paths and ``best_id``.

Requires the same inference credentials as ``server.py`` (API key and/or GGUF / LM Studio).

Example input line::

    {"prompt": "Build an AND gate with pBAD and GFP reporter."}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from submission_settings import apply_settings, submission_dir


def main() -> int:
    cfg = apply_settings()
    root = Path(__file__).resolve().parent
    test_rel = cfg.get("TEST_DATA_CLEAN_PATH", "data/test_prompts.jsonl")
    test_path = Path(test_rel)
    if not test_path.is_absolute():
        test_path = root / test_path
    if not test_path.is_file():
        print(f"[predict] Missing test file: {test_path}", file=sys.stderr)
        return 1

    out_root = submission_dir(cfg)
    lines = [
        ln.strip()
        for ln in test_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    manifest = []

    # Import after env is set (backend reads keys on first use).
    from server import _compile

    for i, ln in enumerate(lines, start=1):
        try:
            row = json.loads(ln)
        except json.JSONDecodeError as e:
            print(f"[predict] Line {i}: invalid JSON: {e}", file=sys.stderr)
            return 1
        prompt = (row.get("prompt") or row.get("text") or "").strip()
        if not prompt:
            print(f"[predict] Line {i}: missing prompt", file=sys.stderr)
            return 1
        n = int(row.get("n", 4))
        result = _compile(prompt, n=n)
        out_file = out_root / f"predict_{i:06d}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        manifest.append(
            {
                "index": i,
                "prompt": prompt,
                "out": str(out_file.relative_to(root)),
                "best_id": result.get("best_id"),
            }
        )
        print(f"[predict] {i}/{len(lines)} → {out_file.name}")

    man_path = out_root / "predict_manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")
    print(f"[predict] Wrote manifest {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
