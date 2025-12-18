#!/usr/bin/env python3
"""Temperature/Platt calibration for OCR routing uncertainty."""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.jsonl import get_label, read_jsonl
from src.gating.calibrator import PlattCalibrator, TemperatureCalibrator

BIN_NUM = 15


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = BIN_NUM) -> Tuple[float, List[Tuple[float, float, int, float, float, float]]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows: List[Tuple[float, float, int, float, float, float]] = []
    n = len(y_true)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            idx = (y_prob >= lo) & (y_prob <= hi)
        else:
            idx = (y_prob >= lo) & (y_prob < hi)
        count = int(idx.sum())
        if count == 0:
            rows.append((lo, hi, 0, float("nan"), float("nan"), 0.0))
            continue
        acc = float(y_true[idx].mean())
        conf = float(y_prob[idx].mean())
        weight = count / max(n, 1)
        ece += weight * abs(acc - conf)
        rows.append((lo, hi, count, acc, conf, weight))
    return ece, rows


def build_score(row: Dict[str, object], use_margin: bool, w1: float, w2: float) -> float:
    ent_val = row.get("out1_entropy")
    ent = float(ent_val) if ent_val is not None else 0.0
    score = w1 * ent
    if use_margin:
        margin_val = row.get("out1_margin")
        margin = float(margin_val) if margin_val is not None else 0.0
        score += w2 * (-margin)
    return score


def fit_temperature(scores: np.ndarray, labels: np.ndarray, T_min: float = 0.1, T_max: float = 5.0, num: int = 60) -> float:
    labels = labels.astype(float)
    scores = scores.astype(float)

    def bce(temp: float) -> float:
        probs = sigmoid(scores / temp)
        eps = 1e-8
        return -np.mean(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps))

    temps = np.linspace(T_min, T_max, num)
    losses = np.array([bce(t) for t in temps])
    best = float(temps[int(np.argmin(losses))])
    return best


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    labels = labels.astype(float)
    scores = scores.astype(float)
    # simple gradient descent
    a, b = 1.0, 0.0
    lr = 0.01
    for _ in range(500):
        logits = a * scores + b
        probs = sigmoid(logits)
        grad_a = np.mean((probs - labels) * scores)
        grad_b = np.mean(probs - labels)
        a -= lr * grad_a
        b -= lr * grad_b
    return float(a), float(b)


def save_bins(path: Path, rows: List[Tuple[float, float, int, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("bin_lo,bin_hi,count,accuracy,confidence,weight\n")
        for lo, hi, count, acc, conf, weight in rows:
            f.write(f"{lo},{hi},{count},{acc},{conf},{weight}\n")


def draw_report_image(path: Path, before: Dict[str, float], after: Dict[str, float]) -> None:
    width, height = 480, 200
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Calibration Summary", fill="black")
    lines = [
        f"Train ECE before: {before['train']:.4f}",
        f"Train ECE after : {after['train']:.4f}",
        f"Valid ECE before: {before['valid']:.4f}",
        f"Valid ECE after : {after['valid']:.4f}",
    ]
    y = 60
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 30
    img.save(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_labels(rows: List[Dict[str, object]]) -> np.ndarray:
    labels: List[int] = []
    for row in rows:
        lab = get_label(row)
        if lab is None:
            raise ValueError("label_call_ocr_good missing; cannot calibrate")
        labels.append(lab)
    return np.asarray(labels, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate uncertainty scores via temperature/platt scaling")
    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--use_margin", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--platt", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report_dir", required=True)
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    valid_rows = read_jsonl(args.valid)
    if not train_rows or not valid_rows:
        raise ValueError("Train/valid JSONL must contain at least one record")

    y_train = extract_labels(train_rows)
    y_valid = extract_labels(valid_rows)

    score_train = np.array([build_score(r, args.use_margin, args.w1, args.w2) for r in train_rows], dtype=float)
    score_valid = np.array([build_score(r, args.use_margin, args.w1, args.w2) for r in valid_rows], dtype=float)

    # baseline probabilities
    p_train_before = sigmoid(score_train)
    p_valid_before = sigmoid(score_valid)
    ece_train_before, bins_train_before = expected_calibration_error(y_train, p_train_before)
    ece_valid_before, bins_valid_before = expected_calibration_error(y_valid, p_valid_before)

    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)
    save_bins(report_dir / "bins_train_before.csv", bins_train_before)
    save_bins(report_dir / "bins_valid_before.csv", bins_valid_before)

    feature_names = ["out1_entropy"]
    if args.use_margin:
        feature_names.append("neg_out1_margin")
    combo = f"w1={args.w1},w2={args.w2}" if args.use_margin else f"w1={args.w1}"

    if args.platt:
        a, b = fit_platt(score_train, y_train)
        calibrator = PlattCalibrator(a=a, b=b, feature_names=feature_names, combo=combo)
        p_train_after = sigmoid(calibrator.transform_scores(score_train))
        p_valid_after = sigmoid(calibrator.transform_scores(score_valid))
    else:
        T = fit_temperature(score_train, y_train)
        calibrator = TemperatureCalibrator(T=T, feature_names=feature_names, combo=combo)
        p_train_after = sigmoid(calibrator.transform_scores(score_train))
        p_valid_after = sigmoid(calibrator.transform_scores(score_valid))

    ece_train_after, bins_train_after = expected_calibration_error(y_train, p_train_after)
    ece_valid_after, bins_valid_after = expected_calibration_error(y_valid, p_valid_after)
    save_bins(report_dir / "bins_train_after.csv", bins_train_after)
    save_bins(report_dir / "bins_valid_after.csv", bins_valid_after)

    summary = {
        "train_before": float(ece_train_before),
        "train_after": float(ece_train_after),
        "valid_before": float(ece_valid_before),
        "valid_after": float(ece_valid_after),
    }
    with open(report_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    draw_report_image(
        report_dir / "ece_before_after.png",
        {"train": ece_train_before, "valid": ece_valid_before},
        {"train": ece_train_after, "valid": ece_valid_after},
    )

    out_path = Path(args.out)
    if out_path.parent:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    calibrator.save(str(out_path))
    print(f"[calibration] saved calibrator -> {args.out}")
    print(
        "[ece] train {:.4f} -> {:.4f} | valid {:.4f} -> {:.4f}".format(
            ece_train_before, ece_train_after, ece_valid_before, ece_valid_after
        )
    )


if __name__ == "__main__":
    main()
