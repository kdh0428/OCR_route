"""Feature vector builders for learned gating."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from src.gating.calibrator import TemperatureCalibrator
from src.features.jsonl import safe_get

LANG_BUCKETS = ("en", "ko", "mixed", "other")


def normalize_language(value: Optional[str]) -> str:
    if not value:
        return "other"
    lowered = value.lower()
    if lowered in LANG_BUCKETS:
        return lowered
    if "en" in lowered:
        return "en"
    if "ko" in lowered or "kor" in lowered or "kr" in lowered:
        return "ko"
    if "mix" in lowered:
        return "mixed"
    return "other"


def load_feature_spec(path: str | Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_raw_scores(row: Dict[str, object], use_margin: bool, w1: float, w2: float) -> Tuple[float, float, float]:
    ent_val = row.get("out1_entropy")
    entropy = float(ent_val) if ent_val is not None else 0.0
    margin_val = row.get("out1_margin")
    margin = float(margin_val) if margin_val is not None else 0.0
    neg_margin = -margin
    score = w1 * entropy + (w2 * neg_margin if use_margin else 0.0)
    return entropy, neg_margin, score


def build_example(
    row: Dict[str, object],
    calibrator: TemperatureCalibrator,
    feature_spec: Dict[str, object],
    *,
    use_margin: bool = True,
    w1: float = 1.0,
    w2: float = 1.0,
) -> Tuple[np.ndarray, Optional[int]]:
    entropy, neg_margin, _ = compute_raw_scores(row, use_margin, w1, w2)
    calib_entropy = calibrator.transform_scores(np.array([entropy]))[0]
    calib_neg_margin = calibrator.transform_scores(np.array([neg_margin]))[0]

    out_len = float(row.get("out1_len", 0.0))
    text_density = float(safe_get(row, "meta.text_density", -1.0) or -1.0)

    language = normalize_language(safe_get(row, "meta.language"))
    lang_onehot = {bucket: 1.0 if bucket == language else 0.0 for bucket in LANG_BUCKETS}

    features = {
        "calib_entropy": calib_entropy,
        "calib_neg_margin": calib_neg_margin,
        "out1_len": out_len,
        "meta.has_digits": 1.0 if bool(safe_get(row, "meta.has_digits", False)) else 0.0,
        "meta.has_units": 1.0 if bool(safe_get(row, "meta.has_units", False)) else 0.0,
        "meta.is_yesno": 1.0 if bool(safe_get(row, "meta.is_yesno", False)) else 0.0,
        "lang_en": lang_onehot["en"],
        "lang_ko": lang_onehot["ko"],
        "lang_mixed": lang_onehot["mixed"],
        "lang_other": lang_onehot["other"],
        "text_density": text_density,
    }

    defaults: Dict[str, float] = feature_spec.get("defaults", {})  # type: ignore[arg-type]
    ordered = []
    for name in feature_spec["features"]:  # type: ignore[index]
        if name in features:
            ordered.append(features[name])
        else:
            ordered.append(defaults.get(name, -1.0))

    label = row.get("label_call_ocr_good")
    y: Optional[int]
    if label is None:
        y = None
    else:
        y = 1 if bool(label) else 0
    return np.array(ordered, dtype=float), y
