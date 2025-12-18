import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build import build_example
from src.gating.calibrator import TemperatureCalibrator


def test_build_example_vector_length():
    row = {
        "out1_entropy": 1.2,
        "out1_margin": 0.4,
        "out1_len": 7,
        "meta": {
            "has_digits": True,
            "has_units": False,
            "is_yesno": True,
            "language": "en",
            "text_density": 0.15,
        },
        "label_call_ocr_good": True,
    }
    calibrator = TemperatureCalibrator(T=2.0, feature_names=["out1_entropy", "neg_out1_margin"])
    spec = {
        "features": [
            "calib_entropy",
            "calib_neg_margin",
            "out1_len",
            "meta.has_digits",
            "meta.has_units",
            "meta.is_yesno",
            "lang_en",
            "lang_ko",
            "lang_mixed",
            "lang_other",
            "text_density",
        ],
        "defaults": {"text_density": -1.0},
    }
    vec, label = build_example(row, calibrator, spec)
    assert vec.shape[0] == len(spec["features"])
    assert label == 1
    # ensure calibrated values scaled by T
    assert np.isclose(vec[0], row["out1_entropy"] / calibrator.T)


def test_missing_values_defaults():
    row = {"out1_entropy": 0.9}
    calibrator = TemperatureCalibrator(T=1.0)
    spec = {"features": ["text_density"], "defaults": {"text_density": -1.0}}
    vec, label = build_example(row, calibrator, spec)
    assert label is None
    assert np.allclose(vec, np.array([-1.0]))
