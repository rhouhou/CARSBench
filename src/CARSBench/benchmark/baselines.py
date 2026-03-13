from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

from CARSBench.datasets.schema import SampleBatch


def zero_baseline(spectrum: np.ndarray) -> np.ndarray:
    """
    Predict zero Raman signal.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    return np.zeros_like(spectrum)


def identity_baseline(spectrum: np.ndarray) -> np.ndarray:
    """
    Return the measured spectrum itself.
    Useful as a sanity-check upper proxy, not a valid retrieval method.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    return spectrum.copy()


def moving_average_baseline(
    spectrum: np.ndarray,
    window: int = 21,
) -> np.ndarray:
    """
    Smoothed baseline predictor.
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)

    if window <= 1:
        return spectrum.copy()

    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(spectrum, kernel, mode="same")


def highpass_baseline(
    spectrum: np.ndarray,
    window: int = 31,
) -> np.ndarray:
    """
    High-pass prediction:
        spectrum - moving_average(spectrum)
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    smooth = moving_average_baseline(spectrum, window=window)
    return spectrum - smooth


@dataclass
class BaselineModel:
    """
    Lightweight wrapper for baseline retrieval models.
    """

    name: str
    predict_fn: Callable[[np.ndarray], np.ndarray]

    def predict(self, spectrum: np.ndarray) -> np.ndarray:
        return self.predict_fn(spectrum)


def get_default_baselines() -> Dict[str, BaselineModel]:
    """
    Standard benchmark baselines.
    """
    return {
        "zero": BaselineModel(
            name="zero",
            predict_fn=zero_baseline,
        ),
        "identity": BaselineModel(
            name="identity",
            predict_fn=identity_baseline,
        ),
        "smoothed": BaselineModel(
            name="smoothed",
            predict_fn=lambda s: moving_average_baseline(s, window=21),
        ),
        "highpass": BaselineModel(
            name="highpass",
            predict_fn=lambda s: highpass_baseline(s, window=31),
        ),
    }


def evaluate_baseline(
    model: BaselineModel,
    batch: SampleBatch,
):
    """
    Apply one baseline model to a batch.
    """
    preds = []
    targets = []

    for sample in batch.samples:
        preds.append(model.predict(sample.spectrum))
        targets.append(sample.raman_target)

    return preds, targets