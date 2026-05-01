#!/usr/bin/env python3
import gzip
import html
import json
import re
from pathlib import Path


INPUT_PATH = Path("xml_parts.xml.gz")
OUTPUT_PATH = Path("igem_dataset.jsonl")


TABLE_START_RE = re.compile(r'<table_data name="([^"]+)">')
FIELD_RE = re.compile(
    r'<field name="([^"]+)"(?: xsi:nil="true"\s*/>|>(.*?)</field>)',
    re.S,
)


def normalize_part_type(value: str, categories: str = ""):
    v = value.strip().lower()
    c = categories.strip().lower()
    if not v:
        return None
    if v.startswith("promoter") or v == "regulatory" or v == "generator":
        return "Promoter"
    if "generator" in c:
        return "Promoter"
    if v == "rbs":
        return "RBS"
    if v in {"cds", "coding", "coding sequence"}:
        return "CDS"
    if v.startswith("terminator"):
        return "Terminator"
    return None


def clean_sequence(value: str) -> str:
    seq = re.sub(r"[^A-Za-z]", "", value).upper()
    return seq


def parse_row_fields(row_xml: str):
    row = {}
    for m in FIELD_RE.finditer(row_xml):
        name = m.group(1)
        val = m.group(2) if m.group(2) is not None else ""
        row[name] = html.unescape(val)
    return row


def iter_lines_with_tolerant_gzip(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        while True:
            try:
                line = f.readline()
            except gzip.BadGzipFile:
                # Some registry exports have trailing junk after valid gzip content.
                break
            if not line:
                break
            yield line


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    kept = 0
    seen_rows = 0
    in_parts_table = False
    in_row = False
    row_lines = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for line in iter_lines_with_tolerant_gzip(INPUT_PATH):
            if not in_parts_table:
                m = TABLE_START_RE.search(line)
                if m and m.group(1) == "parts":
                    in_parts_table = True
                continue

            if "</table_data>" in line:
                break

            if "<row>" in line:
                in_row = True
                row_lines = [line]
                continue

            if in_row:
                row_lines.append(line)
                if "</row>" not in line:
                    continue

                in_row = False
                seen_rows += 1
                row = parse_row_fields("".join(row_lines))

                part_type = normalize_part_type(
                    row.get("part_type", ""),
                    row.get("categories", ""),
                )
                if part_type is None:
                    continue

                seq = clean_sequence(row.get("sequence", ""))
                if len(seq) < 40:
                    continue
                if "N" in seq:
                    continue
                if re.search(r"[^ACGT]", seq):
                    continue

                record = {
                    "part_id": row.get("part_id", ""),
                    "part_name": row.get("part_name", "").strip(),
                    "part_type": part_type,
                    "short_desc": row.get("short_desc", "").strip(),
                    "sequence": seq,
                }
                out.write(json.dumps(record, ensure_ascii=True) + "\n")
                kept += 1

    print(f"Parsed rows in parts table: {seen_rows}")
    print(f"Records written: {kept}")
    print(f"Output: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
