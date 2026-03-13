from __future__ import annotations

from typing import Callable

from .metrics import compute_all_metrics


def run_model_on_batch(
    model: Callable,
    batch,
):
    """
    Run a callable model on a batch.

    The model is expected to take one spectrum and return one prediction.
    """
    preds = [model(sample.spectrum) for sample in batch.samples]
    targets = [sample.raman_target for sample in batch.samples]

    return preds, targets


def evaluate_predictions(
    preds,
    targets,
):
    """
    Aggregate benchmark metrics over prediction/target pairs.
    """
    metric_list = [
        compute_all_metrics(y_true=target, y_pred=pred)
        for pred, target in zip(preds, targets)
    ]

    keys = metric_list[0].keys()
    return {
        key: sum(m[key] for m in metric_list) / len(metric_list)
        for key in keys
    }