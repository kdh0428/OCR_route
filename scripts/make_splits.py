#!/usr/bin/env python3
"""Utility to flatten routing JSONL logs and create train/valid/test splits."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_training_row(raw: Dict, use_trigger_label: bool = False) -> Dict:
    """Extract the fields needed for calibration/gating training."""
    meta = raw.get("meta") or {}
    answer = raw.get("answer_first", "") or ""
    length = int(meta.get("answer_len", len(str(answer))))
    row = {
        "qid": raw.get("qid"),
        "out1_entropy": raw.get("mean_entropy_first"),
        "out1_margin": raw.get("min_margin_first"),
        "out1_len": length,
        "meta": {
            "has_digits": bool(meta.get("has_digits")),
            "has_units": bool(meta.get("has_units")),
            "is_yesno": bool(meta.get("is_yesno")),
            "language": meta.get("language", "other"),
            "text_density": float(meta.get("text_density", -1.0)),
        },
    }
    if use_trigger_label:
        row["label_call_ocr_good"] = bool(raw.get("used_ocr") and raw.get("triggered"))
    else:
        row["label_call_ocr_good"] = raw.get("label_call_ocr_good")
    return row


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split routing results into train/valid/test JSONL files.")
    parser.add_argument("--source", required=True, help="Source JSONL file produced by run_dataset_eval.py")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--use-trigger-label", action="store_true", help="Use used_ocr & triggered as label")
    args = parser.parse_args()

    source = Path(args.source)
    rows_raw = load_rows(source)
    if not rows_raw:
        raise ValueError(f"No rows loaded from {source}")

    rows = [
        build_training_row(row, use_trigger_label=args.use_trigger_label)
        for row in rows_raw
    ]

    random.seed(args.seed)
    random.shuffle(rows)

    n = len(rows)
    train_end = int(n * args.train_ratio)
    valid_end = train_end + int(n * args.valid_ratio)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", rows[:train_end])
    write_jsonl(out_dir / "valid.jsonl", rows[train_end:valid_end])
    write_jsonl(out_dir / "test.jsonl", rows[valid_end:])

    print(
        f"[make_splits] total={n} "
        f"train={train_end} valid={max(valid_end - train_end, 0)} test={max(n - valid_end, 0)} "
        f"-> {out_dir}"
    )


if __name__ == "__main__":
    main()
