#!/usr/bin/env python3
"""Stub evaluation loop for TextVQA-style datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from ocr_route.cli import build_pipeline, parse_args, setup_logging  # reuse CLI helpers
from ocr_route.utils import iter_csv_records


def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OCR routing on a TextVQA-style CSV")
    parser.add_argument("--csv", required=True, help="CSV file with columns image,question,answer")
    parser.add_argument("--out", help="Optional JSON report path")
    args, remaining = parser.parse_known_args()

    cli_args = parse_args(remaining)
    setup_logging(cli_args.log_level)
    pipeline = build_pipeline(cli_args)

    total = 0
    correct = 0
    for row in tqdm(iter_csv_records(args.csv), desc="Evaluating"):
        image = row.get("image")
        question = row.get("question")
        answer = row.get("answer")
        if not all([image, question, answer]):
            continue
        result = pipeline.run(image=image, question=question)
        pred = result.answer
        total += 1
        if exact_match(pred, answer):
            correct += 1

    accuracy = (correct / total) if total else 0.0
    report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
