#!/usr/bin/env python3
"""Grid search utility for entropy/margin thresholds using saved JSONL results."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

STRUCTURED_KEYWORDS = ("latex", "table", "tabular", "equation", "matrix")
STRUCTURED_PATTERNS = ("\\begin{", "\\frac", "\\section", "\\tabular", "\\matrix")


def parse_grid(arg: str) -> List[float]:
    if ":" in arg:
        start, stop, step = (float(x) for x in arg.split(":"))
        values = []
        cur = start
        while cur <= stop + 1e-9:
            values.append(round(cur, 6))
            cur += step
        return values
    return [float(x) for x in arg.split(",")]


def load_results(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def structured_flag(question: str, answer: str | None) -> bool:
    corpus = " ".join(filter(None, [question or "", answer or ""]))
    lower = corpus.lower()
    if any(keyword in lower for keyword in STRUCTURED_KEYWORDS):
        return True
    return any(pattern in corpus for pattern in STRUCTURED_PATTERNS)


def evaluate_combo(
    samples: List[Dict[str, object]],
    entropy_a: float,
    entropy_b: float,
    margin_tau: float,
) -> Tuple[float, Dict[str, float]]:
    metric_keys = ["exact_match", "cer", "wer", "precision", "recall", "f1", "rouge_l"]
    totals = defaultdict(float)
    triggered = 0
    total = len(samples)
    for sample in samples:
        answer_first = sample.get("answer_first") or ""
        question = sample.get("question") or ""
        mean_entropy = sample.get("mean_entropy_first", 0.0)
        min_margin = sample.get("min_margin_first", 0.0)
        length_tokens = max(len(str(answer_first).split()), 1)
        tau_h = entropy_a + entropy_b * math.log1p(length_tokens)
        force_structured = structured_flag(question, answer_first)
        trigger = force_structured or (mean_entropy > tau_h) or (min_margin < margin_tau)
        metrics_key = "routed_metrics" if trigger else "baseline_metrics"
        metrics = sample.get(metrics_key) or sample.get("baseline_metrics")
        if not metrics:
            continue
        triggered += int(trigger)
        for key in metric_keys:
            totals[key] += float(metrics.get(key, 0.0))
    coverage = triggered / total if total else 0.0
    averages = {k: (totals[k] / total if total else 0.0) for k in metric_keys}
    return coverage, averages


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold tuning via risk-coverage sweep")
    parser.add_argument("--results", required=True, help="JSONL file from evaluation script")
    parser.add_argument("--entropy-a", required=True, help="Grid for entropy intercept (e.g., 0.2,0.4 or 0.2:0.8:0.1)")
    parser.add_argument("--entropy-b", required=True, help="Grid for entropy slope")
    parser.add_argument("--margin-tau", required=True, help="Grid for margin threshold")
    parser.add_argument("--coverage-min", type=float, default=0.6)
    parser.add_argument("--coverage-max", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=5, help="Print top K combos by F1 within coverage range")
    args = parser.parse_args()

    entropy_a_grid = parse_grid(args.entropy_a)
    entropy_b_grid = parse_grid(args.entropy_b)
    margin_grid = parse_grid(args.margin_tau)
    samples = load_results(Path(args.results))

    candidates: List[Tuple[float, float, float, float, Dict[str, float]]] = []
    for a in entropy_a_grid:
        for b in entropy_b_grid:
            for tau in margin_grid:
                coverage, metrics = evaluate_combo(samples, a, b, tau)
                if args.coverage_min <= coverage <= args.coverage_max:
                    candidates.append((coverage, a, b, tau, metrics))

    if not candidates:
        print("No combinations within coverage range. Increase grid or adjust range.")
        return

    candidates.sort(key=lambda item: item[4].get("f1", 0.0), reverse=True)
    print(
        f"Top {min(args.top_k, len(candidates))} combinations within coverage "
        f"[{args.coverage_min},{args.coverage_max}]"
    )
    for coverage, a, b, tau, metrics in candidates[: args.top_k]:
        print(
            json.dumps(
                {
                    "coverage": coverage,
                    "entropy_a": a,
                    "entropy_b": b,
                    "margin_tau": tau,
                    "metrics": metrics,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
