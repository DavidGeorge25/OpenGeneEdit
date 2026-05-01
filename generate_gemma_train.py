#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List

from openai import BadRequestError
from openai import APIStatusError
from openai import AzureOpenAI
from openai import RateLimitError


TARGET_TYPES = ("Promoter", "RBS", "CDS", "Terminator")
SYSTEM_PROMPT = (
    "You are a PhD Synthetic Biologist. Given a DNA part name, type, and description, "
    "explain the specific biochemical logic of how this part functions within a genetic circuit. "
    "Mention design constraints (e.g., 'must be upstream of a CDS' or 'reacts to Lead ions'). "
    "Keep it to 2 concise sentences."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Gemma 4 thinking dataset from iGEM JSONL via Azure OpenAI."
    )
    parser.add_argument("--input", default="igem_dataset.jsonl", help="Input JSONL path.")
    parser.add_argument("--output", default="gemma_train.jsonl", help="Output JSONL path.")
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
    parser.add_argument(
        "--deployment",
        default="gpt-4o-mini",
        help="Azure OpenAI deployment name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and emit deterministic placeholder thoughts for format testing.",
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


def init_client():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_ENDPOINT", endpoint),
        ]
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def build_user_prompt(record: Dict[str, str]) -> str:
    return (
        f"Part: {record['part_name']}. "
        f"Type: {record['part_type']}. "
        f"Description: {record['short_desc']}."
    )


def fallback_thought(record: Dict[str, str]) -> str:
    part_type = record["part_type"]
    desc = record["short_desc"] or "the requested circuit behavior"
    if part_type == "Promoter":
        return (
            f"This promoter should convert the stated signal in '{desc}' into transcriptional output with context-appropriate strength. "
            "It must be placed upstream of a CDS and paired with host-compatible regulation to minimize leak and unintended activation."
        )
    if part_type == "RBS":
        return (
            f"This RBS should tune translation initiation for '{desc}' so protein expression is balanced with upstream transcription. "
            "It must be directly upstream of the CDS with proper spacing and sequence context to avoid weak initiation or burden."
        )
    if part_type == "CDS":
        return (
            f"This CDS should encode the functional effector described in '{desc}' and preserve a coherent reading frame for reliable protein output. "
            "It must remain in-frame with start/stop context and be matched to host expression constraints such as codon usage and toxicity."
        )
    return (
        f"This terminator should stop transcription linked to '{desc}' to prevent read-through into downstream modules. "
        "It must be placed downstream of the transcribed unit and be strong enough for the host context to improve circuit insulation."
    )


def get_reasoning_with_backoff(
    client: AzureOpenAI, deployment: str, record: Dict[str, str], max_retries: int = 8
) -> str:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(record)},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Model returned empty reasoning text.")
            return text
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0.0, 0.5)
            print(
                f"[retry] rate-limited for {record['part_name']}, "
                f"attempt={attempt + 1}, sleeping={sleep_for:.2f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            delay *= 2
        except APIStatusError as exc:
            if exc.status_code != 429 or attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0.0, 0.5)
            print(
                f"[retry] HTTP 429 for {record['part_name']}, "
                f"attempt={attempt + 1}, sleeping={sleep_for:.2f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            delay *= 2
        except BadRequestError as exc:
            msg = str(exc).lower()
            if "content_filter" in msg or "responsibleaipolicyviolation" in msg:
                print(
                    f"[warn] content-filtered prompt for {record['part_name']}; "
                    "using fallback thought.",
                    flush=True,
                )
                return fallback_thought(record)
            raise


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

    client = None if args.dry_run else init_client()

    with open(args.output, "w", encoding="utf-8") as out:
        for idx, record in enumerate(selected, start=1):
            if args.dry_run:
                thought = (
                    f"This {record['part_type'].lower()} should be placed according to its regulatory role "
                    f"in the circuit and matched to host context for predictable expression. "
                    f"It must satisfy placement and compatibility constraints implied by the part description."
                )
            else:
                thought = get_reasoning_with_backoff(client, args.deployment, record)

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
