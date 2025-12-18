"""Temperature (and Platt) calibrators for uncertainty scores."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class TemperatureCalibrator:
    """Simple temperature scaling calibrator."""

    T: float = 1.0
    feature_names: Optional[List[str]] = None
    combo: Optional[str] = None

    def transform_scores(self, scores: np.ndarray) -> np.ndarray:
        """Scale raw uncertainty scores by temperature."""
        eps = 1e-8
        denom = max(float(self.T), eps)
        return np.asarray(scores, dtype=float) / denom

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "temperature",
                    "T": float(self.T),
                    "features": list(self.feature_names or []),
                    "combo": self.combo,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(path: str) -> "TemperatureCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return TemperatureCalibrator(
            T=float(obj.get("T", 1.0)),
            feature_names=obj.get("features") or [],
            combo=obj.get("combo"),
        )


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator p = sigmoid(a * score + b)."""

    a: float = 1.0
    b: float = 0.0
    feature_names: Optional[List[str]] = None
    combo: Optional[str] = None

    def transform_scores(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float) * float(self.a) + float(self.b)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "platt",
                    "a": float(self.a),
                    "b": float(self.b),
                    "features": list(self.feature_names or []),
                    "combo": self.combo,
                },
                f,
                indent=2,
            )

    @staticmethod
    def load(path: str) -> "PlattCalibrator":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return PlattCalibrator(
            a=float(obj.get("a", 1.0)),
            b=float(obj.get("b", 0.0)),
            feature_names=obj.get("features") or [],
            combo=obj.get("combo"),
        )

