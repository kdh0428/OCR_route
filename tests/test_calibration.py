import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gating.calibrator import PlattCalibrator, TemperatureCalibrator


def test_temperature_identity():
    calibrator = TemperatureCalibrator(T=1.0)
    scores = np.array([0.5, 1.0, -0.7])
    np.testing.assert_allclose(calibrator.transform_scores(scores), scores)


def test_temperature_scale():
    calibrator = TemperatureCalibrator(T=2.0)
    scores = np.array([1.0, -1.0])
    expected = np.array([0.5, -0.5])
    np.testing.assert_allclose(calibrator.transform_scores(scores), expected)


def test_platt_transform():
    calibrator = PlattCalibrator(a=2.0, b=-1.0)
    scores = np.array([0.5])
    np.testing.assert_allclose(calibrator.transform_scores(scores), np.array([0.0]))
