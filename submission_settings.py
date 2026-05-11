"""Load SETTINGS.json and map paths into ``DGENE_*`` env vars used by ``igem_rag`` and friends.

Kaggle-style submission layout: all train/test/model/output roots are declared only in
``SETTINGS.json`` at the repository root. Call :func:`apply_settings` before importing
modules that read ``os.environ`` for corpus or Chroma paths.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def load_settings(path: Path | None = None) -> Dict[str, Any]:
    root = _repo_root()
    cfg_path = path or (root / "SETTINGS.json")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("SETTINGS.json must contain a JSON object")
    return data


def _abs(root: Path, p: str) -> str:
    p = (p or "").strip()
    if not p:
        return str(root)
    q = Path(p)
    if q.is_absolute():
        return str(q.resolve())
    return str((root / q).resolve())


def apply_settings(path: Path | None = None) -> Dict[str, Any]:
    """Apply SETTINGS.json paths to the environment. Returns the parsed dict."""
    root = _repo_root()
    cfg = load_settings(path)

    os.environ["DGENE_IGEM_JSONL"] = _abs(root, str(cfg.get("IGEM_JSONL", "data/igem_dataset.jsonl")))
    os.environ["DGENE_CHROMA_PATH"] = _abs(root, str(cfg.get("CHROMA_PATH", ".chroma_igem")))

    model_dir = _abs(root, str(cfg.get("MODEL_DIR", "finetune_results/gemma_4_lora")))
    gguf = (cfg.get("GGUF_PATH") or "").strip()
    if gguf:
        os.environ["DGENE_GGUF_PATH"] = _abs(root, gguf) if not Path(gguf).is_absolute() else gguf

    sub = cfg.get("SUBMISSION_DIR", "outputs")
    out_dir = Path(_abs(root, str(sub)))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg["_resolved"] = {
        "REPO_ROOT": str(root),
        "SUBMISSION_DIR": str(out_dir),
        "MODEL_DIR": model_dir,
    }
    return cfg


def submission_dir(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["_resolved"]["SUBMISSION_DIR"])
