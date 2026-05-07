#!/usr/bin/env python3
"""Build Gemma 4 supervised JSONL training data using hosted Gemma 4 (same API path as inference)."""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from inference import generate_text_gemma4

TARGET_TYPES = ("Promoter", "RBS", "CDS", "Terminator")
SYSTEM_PROMPT = (
    "You are a PhD Synthetic Biologist. Given a DNA part name, type, and description, "
    "explain the specific biochemical logic of how this part functions within a genetic circuit. "
    "Mention design constraints (e.g., 'must be upstream of a CDS' or 'reacts to Lead ions'). "
    "Keep it to 2 concise sentences."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build Gemma dataset from iGEM JSONL via hosted Gemma 4 "
            "(GEMINI_API_KEY / GOOGLE_API_KEY + DGENE_GEMINI_MODEL)."
        )
    )
    parser.add_argument(
        "--input",
        default=str(_REPO_ROOT / "data" / "igem_dataset.jsonl"),
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=str(_REPO_ROOT / "data" / "gemma_train.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Exactly how many records to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def validate_sequence(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    cleaned = re.sub(r"[^A-Za-z]", "", seq).upper()
    if len(cleaned) < 40:
        return False
    if "N" in cleaned:
        return False
    return bool(re.fullmatch(r"[ACGT]+", cleaned))


def load_valid_records(path: str) -> List[Dict[str, str]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_part_type = str(obj.get("part_type", "")).strip()
            categories = str(obj.get("categories", "")).strip().lower()
            raw_lower = raw_part_type.lower()
            if raw_lower in {"regulatory", "generator"} or "generator" in categories:
                part_type = "Promoter"
            else:
                part_type = raw_part_type
            if part_type not in TARGET_TYPES:
                continue
            seq = str(obj.get("sequence", ""))
            if not validate_sequence(seq):
                continue
            records.append(
                {
                    "part_name": str(obj.get("part_name", "")).strip(),
                    "part_type": part_type,
                    "short_desc": str(obj.get("short_desc", "")).strip(),
                    "sequence": re.sub(r"[^A-Za-z]", "", seq).upper(),
                }
            )
    return records


def sample_diverse(records: List[Dict[str, str]], sample_size: int, seed: int):
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for record in records:
        by_type[record["part_type"]].append(record)

    for part_type in TARGET_TYPES:
        rng.shuffle(by_type[part_type])

    total_available = sum(len(by_type[t]) for t in TARGET_TYPES)
    if total_available < sample_size:
        raise ValueError(
            f"Not enough valid records for sample size {sample_size}. "
            f"Only {total_available} available."
        )

    missing_types = [t for t in TARGET_TYPES if len(by_type[t]) == 0]
    if missing_types:
        raise ValueError(
            "Cannot build balanced sample. Missing part types: "
            + ", ".join(missing_types)
        )

    if sample_size % len(TARGET_TYPES) != 0:
        raise ValueError(
            f"sample_size ({sample_size}) must be divisible by {len(TARGET_TYPES)} "
            "for equal per-type sampling."
        )

    per_type = sample_size // len(TARGET_TYPES)
    chosen = []
    for part_type in TARGET_TYPES:
        if len(by_type[part_type]) < per_type:
            raise ValueError(
                f"Cannot sample {per_type} {part_type} entries; only "
                f"{len(by_type[part_type])} available."
            )
        chosen.extend(by_type[part_type][:per_type])

    rng.shuffle(chosen)
    return chosen


def build_user_prompt(record: Dict[str, str]) -> str:
    return (
        f"Part: {record['part_name']}. "
        f"Type: {record['part_type']}. "
        f"Description: {record['short_desc']}."
    )


def get_reasoning_gemma(record: Dict[str, str], max_retries: int = 8) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            text = generate_text_gemma4(
                build_user_prompt(record),
                system_message=SYSTEM_PROMPT,
                temperature=0.2,
            ).strip()
            if not text:
                raise RuntimeError("Model returned empty reasoning text.")
            return text
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            msg = str(exc).lower()
            if not ("429" in msg or "resource exhausted" in msg or "rate" in msg):
                raise
            sleep_for = delay + random.uniform(0.0, 0.5)
            print(
                f"[retry] rate-limited for {record['part_name']}, "
                f"attempt={attempt + 1}, sleeping={sleep_for:.2f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2.0, 120.0)


def format_gemma_example(record: Dict[str, str], thought: str) -> Dict[str, object]:
    human = f"Design a {record['part_type']} for the following purpose: {record['short_desc']}"
    gpt = f"<|channel>thought\n{thought}\n<channel|>\n{record['sequence']}"
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ]
    }


def main():
    args = parse_args()

    records = load_valid_records(args.input)
    selected = sample_diverse(records, args.sample_size, args.seed)
    counts = defaultdict(int)
    for r in selected:
        counts[r["part_type"]] += 1

    print(f"Loaded valid records: {len(records)}", flush=True)
    print(
        "Selected counts: "
        + ", ".join(f"{k}={counts[k]}" for k in TARGET_TYPES),
        flush=True,
    )

    with open(args.output, "w", encoding="utf-8") as out:
        for idx, record in enumerate(selected, start=1):
            thought = get_reasoning_gemma(record)

            example = format_gemma_example(record, thought)
            out.write(json.dumps(example, ensure_ascii=True) + "\n")

            if idx % 50 == 0:
                print(f"Processed {idx}/{len(selected)}", flush=True)

    print(f"Done. Wrote {len(selected)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
