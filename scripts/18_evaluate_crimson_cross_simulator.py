
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from CARSBench.ml import (
    CARSBaselineCNN,
    ChunkedNPZDataset,
    select_device,
)

from CARSBench.ml.preprocessing import (
    NormalizationStats,
)


ROOT = Path(__file__).resolve().parents[1]

SCRIPT08_PATH = (
    ROOT
    / "scripts"
    / "08_run_acquisition_scale_study.py"
)

SCRIPT17_PATH = (
    ROOT
    / "scripts"
    / "17_train_fingerprint_source_models.py"
)

MODEL_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "fingerprint_source"
)

CRIMSON_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
    / "crimson_case5b_n5000_seed42_carsbench_fingerprint.npz"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "evaluation"
)

ID_DATA_ROOT = (
    ROOT
    / "data"
    / "cross_simulator_id_test"
)

MODEL_SEEDS = [
    101,
    202,
    303,
]

ID_START = 110_000
ID_SIZE = 5_000

BATCH_SIZE = 256


# ============================================================
# Dynamic imports
# ============================================================

def load_module(
    path: Path,
    module_name: str,
):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# External adapted CRIMSON dataset
# ============================================================

class ExternalDataset:

    def __init__(
        self,
        path: Path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path)

        self.axis = np.asarray(
            data["axis"],
            dtype=np.float32,
        )

        self.spectrum = np.asarray(
            data["spectrum"],
            dtype=np.float32,
        )

        self.target = np.asarray(
            data["raman_target"],
            dtype=np.float32,
        )

        self.mask = np.asarray(
            data["observation_mask"],
            dtype=np.float32,
        )

        if (
            self.spectrum.shape
            != self.target.shape
        ):
            raise ValueError(
                "External spectrum/target shapes differ."
            )

        if (
            self.spectrum.shape
            != self.mask.shape
        ):
            raise ValueError(
                "External spectrum/mask shapes differ."
            )

        if (
            self.spectrum.shape[1]
            != len(self.axis)
        ):
            raise ValueError(
                "External spectral length "
                "does not match axis."
            )

        if not np.all(
            np.isfinite(self.spectrum)
        ):
            raise FloatingPointError(
                "Non-finite external spectra."
            )

        if not np.all(
            np.isfinite(self.target)
        ):
            raise FloatingPointError(
                "Non-finite external targets."
            )

    def __len__(
        self,
    ):
        return self.spectrum.shape[0]

    def __getitem__(
        self,
        index,
    ):
        return {
            "spectrum": self.spectrum[index],
            "observation_mask": self.mask[index],
            "target": self.target[index][
                np.newaxis,
                :
            ],
            "axis": self.axis,
            "dataset_index": int(index),
            "sample_index": int(index),
            "domain_name": (
                "CRIMSON_cross_simulator"
            ),
        }


# ============================================================
# ID test generation
# ============================================================

def valid_id_dataset(
    path: Path,
) -> bool:

    try:
        dataset = ChunkedNPZDataset(
            path
        )

        if len(dataset) != ID_SIZE:
            return False

        first = int(
            dataset[0][
                "sample_index"
            ]
        )

        last = int(
            dataset[
                ID_SIZE - 1
            ][
                "sample_index"
            ]
        )

        return (
            first == ID_START
            and last
            == ID_START + ID_SIZE - 1
        )

    except Exception:
        return False


def ensure_id_test(
    acquisition_protocol,
) -> Path:

    path = (
        ID_DATA_ROOT
        / "A_typical"
    )

    if valid_id_dataset(
        path
    ):
        print(
            "Reusing matched A_typical ID test."
        )

        return path

    if path.exists():
        shutil.rmtree(
            path
        )

    ID_DATA_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Generating independent matched "
        "A_typical ID test..."
    )

    acquisition_protocol.run_generator(
        output_root=ID_DATA_ROOT,
        domain="A_typical",
        samples=ID_SIZE,
        start_index=ID_START,
    )

    if not valid_id_dataset(
        path
    ):
        raise RuntimeError(
            "ID test generation failed."
        )

    return path


# ============================================================
# Model loading
# ============================================================

def load_normalization(
    run_root: Path,
) -> NormalizationStats:

    path = (
        run_root
        / "normalization.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(
            handle
        )

    return NormalizationStats.from_dict(
        data
    )


def load_state_dict(
    path: Path,
):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_model(
    run_root: Path,
    device: torch.device,
):

    model_path = (
        run_root
        / "model.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    model = CARSBaselineCNN(
        in_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
    )

    state = load_state_dict(
        model_path
    )

    model.load_state_dict(
        state
    )

    model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# Dataset preparation
# ============================================================

def collect_dataset_arrays(
    dataset,
):

    n = len(dataset)

    first = dataset[0]

    length = len(
        first["spectrum"]
    )

    spectra = np.empty(
        (n, length),
        dtype=np.float32,
    )

    masks = np.empty(
        (n, length),
        dtype=np.float32,
    )

    targets = np.empty(
        (n, length),
        dtype=np.float32,
    )

    axis = np.asarray(
        first["axis"],
        dtype=np.float32,
    )

    sample_indices = np.empty(
        n,
        dtype=np.int64,
    )

    for index in range(n):

        item = dataset[index]

        current_axis = np.asarray(
            item["axis"],
            dtype=np.float32,
        )

        if not np.allclose(
            current_axis,
            axis,
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RuntimeError(
                "Dataset axes differ between samples."
            )

        spectra[index] = np.asarray(
            item["spectrum"],
            dtype=np.float32,
        )

        masks[index] = np.asarray(
            item[
                "observation_mask"
            ],
            dtype=np.float32,
        )

        target = np.asarray(
            item["target"],
            dtype=np.float32,
        )

        if target.ndim == 2:
            target = target[0]

        targets[index] = target

        sample_indices[index] = int(
            item["sample_index"]
        )

    return {
        "axis": axis,
        "spectrum": spectra,
        "mask": masks,
        "target": targets,
        "sample_index": sample_indices,
    }


# ============================================================
# Source normalization
# ============================================================

def normalized_inputs(
    spectra: np.ndarray,
    masks: np.ndarray,
    stats: NormalizationStats,
):

    normalized = (
        (
            spectra
            - stats.spectrum_mean
        )
        / stats.spectrum_std
    ).astype(
        np.float32
    )

    normalized = np.where(
        masks > 0.5,
        normalized,
        0.0,
    ).astype(
        np.float32
    )

    inputs = np.stack(
        [
            normalized,
            masks,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    return (
        inputs,
        normalized,
    )


# ============================================================
# Input-shift diagnostic
# ============================================================

def input_shift_summary(
    normalized_spectra: np.ndarray,
    masks: np.ndarray,
):

    values = normalized_spectra[
        masks > 0.5
    ].astype(
        np.float64
    )

    return {
        "count": int(
            values.size
        ),

        "mean": float(
            np.mean(values)
        ),

        "std": float(
            np.std(values)
        ),

        "median": float(
            np.median(values)
        ),

        "p01": float(
            np.percentile(
                values,
                1,
            )
        ),

        "p05": float(
            np.percentile(
                values,
                5,
            )
        ),

        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),

        "fraction_abs_gt_3": float(
            np.mean(
                np.abs(values)
                > 3.0
            )
        ),

        "fraction_abs_gt_5": float(
            np.mean(
                np.abs(values)
                > 5.0
            )
        ),
    }


# ============================================================
# Inference
# ============================================================

def predict(
    model,
    inputs: np.ndarray,
    stats: NormalizationStats,
    device: torch.device,
):

    n = inputs.shape[0]

    length = inputs.shape[-1]

    predictions = np.empty(
        (n, length),
        dtype=np.float32,
    )

    with torch.no_grad():

        for start in range(
            0,
            n,
            BATCH_SIZE,
        ):

            end = min(
                start + BATCH_SIZE,
                n,
            )

            batch = torch.from_numpy(
                inputs[
                    start:end
                ]
            ).to(
                device=device,
                dtype=torch.float32,
            )

            prediction_normalized = (
                model(
                    batch
                )
            )

            prediction_normalized = (
                prediction_normalized
                .detach()
                .cpu()
                .numpy()
            )

            if (
                prediction_normalized.ndim
                == 3
            ):
                prediction_normalized = (
                    prediction_normalized[
                        :,
                        0,
                        :
                    ]
                )

            prediction = (
                prediction_normalized
                * stats.target_std
                + stats.target_mean
            )

            predictions[
                start:end
            ] = prediction.astype(
                np.float32
            )

    return predictions


# ============================================================
# Metrics
# ============================================================

def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    target = np.asarray(
        target,
        dtype=np.float64,
    )

    difference = (
        prediction
        - target
    )

    rmse = np.sqrt(
        np.mean(
            difference ** 2,
            axis=1,
        )
    )

    mae = np.mean(
        np.abs(
            difference
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Spectral angle:
    # insensitive to a positive multiplicative scale factor.
    # --------------------------------------------------------

    dot = np.sum(
        prediction * target,
        axis=1,
    )

    prediction_norm = np.linalg.norm(
        prediction,
        axis=1,
    )

    target_norm = np.linalg.norm(
        target,
        axis=1,
    )

    denominator = (
        prediction_norm
        * target_norm
    )

    cosine = np.full(
        len(target),
        np.nan,
        dtype=np.float64,
    )

    valid_angle = (
        denominator
        > 1.0e-15
    )

    cosine[
        valid_angle
    ] = (
        dot[
            valid_angle
        ]
        / denominator[
            valid_angle
        ]
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    spectral_angle = np.arccos(
        cosine
    )

    # --------------------------------------------------------
    # Pearson correlation:
    # insensitive to affine offset + positive scaling.
    # --------------------------------------------------------

    pred_centered = (
        prediction
        - np.mean(
            prediction,
            axis=1,
            keepdims=True,
        )
    )

    target_centered = (
        target
        - np.mean(
            target,
            axis=1,
            keepdims=True,
        )
    )

    corr_num = np.sum(
        pred_centered
        * target_centered,
        axis=1,
    )

    corr_den = (
        np.linalg.norm(
            pred_centered,
            axis=1,
        )
        * np.linalg.norm(
            target_centered,
            axis=1,
        )
    )

    pearson = np.full(
        len(target),
        np.nan,
        dtype=np.float64,
    )

    valid_corr = (
        corr_den
        > 1.0e-15
    )

    pearson[
        valid_corr
    ] = (
        corr_num[
            valid_corr
        ]
        / corr_den[
            valid_corr
        ]
    )

    # --------------------------------------------------------
    # Relative amplitude diagnostics
    # --------------------------------------------------------

    target_rms = np.sqrt(
        np.mean(
            target ** 2,
            axis=1,
        )
    )

    prediction_rms = np.sqrt(
        np.mean(
            prediction ** 2,
            axis=1,
        )
    )

    nrmse_target_rms = np.full(
        len(target),
        np.nan,
        dtype=np.float64,
    )

    rms_ratio = np.full(
        len(target),
        np.nan,
        dtype=np.float64,
    )

    valid_target = (
        target_rms
        > 1.0e-15
    )

    nrmse_target_rms[
        valid_target
    ] = (
        rmse[
            valid_target
        ]
        / target_rms[
            valid_target
        ]
    )

    rms_ratio[
        valid_target
    ] = (
        prediction_rms[
            valid_target
        ]
        / target_rms[
            valid_target
        ]
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "spectral_angle": spectral_angle,
        "pearson": pearson,
        "nrmse_target_rms": (
            nrmse_target_rms
        ),
        "prediction_to_target_rms_ratio": (
            rms_ratio
        ),
    }


def summarize(
    values: np.ndarray,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(
            values
        )
    ]

    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }

    return {
        "n": int(
            finite.size
        ),

        "mean": float(
            np.mean(
                finite
            )
        ),

        "std": float(
            np.std(
                finite
            )
        ),

        "median": float(
            np.median(
                finite
            )
        ),

        "p05": float(
            np.percentile(
                finite,
                5,
            )
        ),

        "p95": float(
            np.percentile(
                finite,
                95,
            )
        ),
    }


def summarize_metrics(
    metrics,
):
    return {
        name: summarize(
            values
        )
        for name, values
        in metrics.items()
    }


# ============================================================
# Normalization consistency
# ============================================================

def normalization_signature(
    stats,
):
    return np.asarray(
        [
            stats.spectrum_mean,
            stats.spectrum_std,
            stats.target_mean,
            stats.target_std,
            stats.num_samples,
            stats.num_observed_spectrum_values,
            stats.num_target_values,
        ],
        dtype=np.float64,
    )


# ============================================================
# Results across model seeds
# ============================================================

def aggregate_seed_means(
    per_seed,
    dataset_name,
):

    metric_names = [
        "rmse",
        "mae",
        "spectral_angle",
        "pearson",
        "nrmse_target_rms",
        "prediction_to_target_rms_ratio",
    ]

    result = {}

    for metric in metric_names:

        values = np.asarray(
            [
                per_seed[
                    str(seed)
                ][
                    dataset_name
                ][
                    "metrics"
                ][
                    metric
                ][
                    "mean"
                ]
                for seed
                in MODEL_SEEDS
            ],
            dtype=np.float64,
        )

        result[
            metric
        ] = {
            "mean_across_model_seeds": float(
                np.mean(values)
            ),

            "std_across_model_seeds": float(
                np.std(values)
            ),

            "per_seed_means": {
                str(seed): float(
                    value
                )
                for seed, value
                in zip(
                    MODEL_SEEDS,
                    values,
                )
            },
        }

    return result


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="cuda",
    )

    args = parser.parse_args()

    device = select_device(
        args.device
    )

    print("=" * 76)
    print(
        "CARSBENCH CROSS-SIMULATOR "
        "ZERO-SHOT EVALUATION"
    )
    print("=" * 76)

    print()
    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Load authoritative source protocol
    # --------------------------------------------------------

    acquisition_protocol = load_module(
        SCRIPT08_PATH,
        "carsbench_acquisition_protocol",
    )

    fingerprint_protocol = load_module(
        SCRIPT17_PATH,
        "carsbench_fingerprint_protocol",
    )

    (
        fingerprint_mask,
        fingerprint_axis,
    ) = (
        fingerprint_protocol
        .load_fingerprint_mask()
    )

    # --------------------------------------------------------
    # Independent source ID test
    # --------------------------------------------------------

    id_path = ensure_id_test(
        acquisition_protocol
    )

    id_dataset = (
        fingerprint_protocol
        .FingerprintDataset(
            id_path,
            fingerprint_mask,
        )
    )

    if len(id_dataset) != ID_SIZE:
        raise RuntimeError(
            "Incorrect ID test size."
        )

    print()
    print("Matched source ID test")
    print("----------------------")
    print(
        "Samples:",
        len(id_dataset),
    )
    print(
        "Indices:",
        id_dataset[0]["sample_index"],
        "to",
        id_dataset[
            len(id_dataset) - 1
        ][
            "sample_index"
        ],
    )

    # --------------------------------------------------------
    # CRIMSON external test
    # --------------------------------------------------------

    crimson_dataset = ExternalDataset(
        CRIMSON_PATH
    )

    print()
    print("CRIMSON external test")
    print("---------------------")
    print(
        "Samples:",
        len(crimson_dataset),
    )

    print(
        "Points:",
        len(
            crimson_dataset.axis
        ),
    )

    print(
        "Range:",
        float(
            crimson_dataset.axis[0]
        ),
        "to",
        float(
            crimson_dataset.axis[-1]
        ),
        "cm^-1",
    )

    # Exact axis agreement after adapter.
    if not np.allclose(
        crimson_dataset.axis,
        fingerprint_axis,
        rtol=0.0,
        atol=1.0e-4,
    ):
        raise RuntimeError(
            "CRIMSON adapted axis does not "
            "match CARSBench fingerprint axis."
        )

    # --------------------------------------------------------
    # Materialize test arrays once
    # --------------------------------------------------------

    print()
    print(
        "Loading matched evaluation arrays..."
    )

    id_arrays = collect_dataset_arrays(
        id_dataset
    )

    crimson_arrays = collect_dataset_arrays(
        crimson_dataset
    )

    # --------------------------------------------------------
    # Check model artifacts + normalizations
    # --------------------------------------------------------

    stats_by_seed = {}

    for seed in MODEL_SEEDS:

        run_root = (
            MODEL_ROOT
            / (
                "A_fingerprint_"
                f"train100000_seed{seed}"
            )
        )

        stats_by_seed[
            seed
        ] = load_normalization(
            run_root
        )

    reference_stats = (
        stats_by_seed[
            MODEL_SEEDS[0]
        ]
    )

    reference_signature = (
        normalization_signature(
            reference_stats
        )
    )

    for seed in MODEL_SEEDS[1:]:

        current = (
            normalization_signature(
                stats_by_seed[
                    seed
                ]
            )
        )

        if not np.array_equal(
            current,
            reference_signature,
        ):
            raise RuntimeError(
                "Source normalization differs "
                "between model seeds."
            )

    print()
    print(
        "Source normalization identical "
        "across all three seeds: PASS"
    )

    # --------------------------------------------------------
    # Input-shift diagnostics
    # --------------------------------------------------------

    (
        id_inputs,
        id_normalized,
    ) = normalized_inputs(
        id_arrays["spectrum"],
        id_arrays["mask"],
        reference_stats,
    )

    (
        crimson_inputs,
        crimson_normalized,
    ) = normalized_inputs(
        crimson_arrays["spectrum"],
        crimson_arrays["mask"],
        reference_stats,
    )

    id_input_shift = (
        input_shift_summary(
            id_normalized,
            id_arrays["mask"],
        )
    )

    crimson_input_shift = (
        input_shift_summary(
            crimson_normalized,
            crimson_arrays["mask"],
        )
    )

    print()
    print("SOURCE-NORMALIZED INPUT DIAGNOSTIC")
    print("----------------------------------")

    print()
    print("A_typical ID:")
    print(
        json.dumps(
            id_input_shift,
            indent=2,
        )
    )

    print()
    print("CRIMSON:")
    print(
        json.dumps(
            crimson_input_shift,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Frozen-model evaluation
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed = {}

    for seed in MODEL_SEEDS:

        print()
        print("=" * 76)
        print(
            f"EVALUATING FROZEN MODEL SEED {seed}"
        )
        print("=" * 76)

        run_root = (
            MODEL_ROOT
            / (
                "A_fingerprint_"
                f"train100000_seed{seed}"
            )
        )

        stats = (
            stats_by_seed[
                seed
            ]
        )

        model = load_model(
            run_root,
            device,
        )

        # ---------------- ID ----------------

        id_prediction = predict(
            model,
            id_inputs,
            stats,
            device,
        )

        id_metrics = compute_metrics(
            id_prediction,
            id_arrays["target"],
        )

        id_summary = (
            summarize_metrics(
                id_metrics
            )
        )

        # ---------------- CRIMSON ----------------

        crimson_prediction = predict(
            model,
            crimson_inputs,
            stats,
            device,
        )

        crimson_metrics = compute_metrics(
            crimson_prediction,
            crimson_arrays["target"],
        )

        crimson_summary = (
            summarize_metrics(
                crimson_metrics
            )
        )

        per_seed[
            str(seed)
        ] = {
            "source_id": {
                "metrics": (
                    id_summary
                )
            },

            "crimson": {
                "metrics": (
                    crimson_summary
                )
            },
        }

        # Save per-sample outputs for later analysis.
        np.savez_compressed(
            OUTPUT_ROOT
            / f"seed{seed}_source_id.npz",

            axis=id_arrays[
                "axis"
            ],

            sample_index=id_arrays[
                "sample_index"
            ],

            prediction=id_prediction,

            target=id_arrays[
                "target"
            ],

            **{
                name: values.astype(
                    np.float32
                )
                for name, values
                in id_metrics.items()
            },
        )

        np.savez_compressed(
            OUTPUT_ROOT
            / f"seed{seed}_crimson.npz",

            axis=crimson_arrays[
                "axis"
            ],

            sample_index=crimson_arrays[
                "sample_index"
            ],

            prediction=crimson_prediction,

            target=crimson_arrays[
                "target"
            ],

            **{
                name: values.astype(
                    np.float32
                )
                for name, values
                in crimson_metrics.items()
            },
        )

        print()
        print("SOURCE ID")
        print(
            "RMSE:",
            f"{id_summary['rmse']['mean']:.6f}",
        )
        print(
            "MAE:",
            f"{id_summary['mae']['mean']:.6f}",
        )
        print(
            "Angle:",
            f"{id_summary['spectral_angle']['mean']:.6f}",
        )
        print(
            "Pearson:",
            f"{id_summary['pearson']['mean']:.6f}",
        )

        print()
        print("CRIMSON ZERO-SHOT")
        print(
            "RMSE:",
            f"{crimson_summary['rmse']['mean']:.6f}",
        )
        print(
            "MAE:",
            f"{crimson_summary['mae']['mean']:.6f}",
        )
        print(
            "Angle:",
            f"{crimson_summary['spectral_angle']['mean']:.6f}",
        )
        print(
            "Pearson:",
            f"{crimson_summary['pearson']['mean']:.6f}",
        )
        print(
            "NRMSE(target RMS):",
            f"{crimson_summary['nrmse_target_rms']['mean']:.6f}",
        )
        print(
            "Pred/target RMS ratio:",
            f"{crimson_summary['prediction_to_target_rms_ratio']['mean']:.6f}",
        )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Aggregate model-seed results
    # --------------------------------------------------------

    source_aggregate = (
        aggregate_seed_means(
            per_seed,
            "source_id",
        )
    )

    crimson_aggregate = (
        aggregate_seed_means(
            per_seed,
            "crimson",
        )
    )

    # --------------------------------------------------------
    # Cross-simulator degradation
    # --------------------------------------------------------

    degradation = {}

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "nrmse_target_rms",
    ]:

        source_value = (
            source_aggregate[
                metric
            ][
                "mean_across_model_seeds"
            ]
        )

        crimson_value = (
            crimson_aggregate[
                metric
            ][
                "mean_across_model_seeds"
            ]
        )

        degradation[
            metric
        ] = {
            "source_id": float(
                source_value
            ),

            "crimson": float(
                crimson_value
            ),

            "crimson_over_source_ratio": float(
                crimson_value
                / source_value
            ),
        }

    source_corr = (
        source_aggregate[
            "pearson"
        ][
            "mean_across_model_seeds"
        ]
    )

    crimson_corr = (
        crimson_aggregate[
            "pearson"
        ][
            "mean_across_model_seeds"
        ]
    )

    degradation[
        "pearson"
    ] = {
        "source_id": float(
            source_corr
        ),

        "crimson": float(
            crimson_corr
        ),

        "absolute_drop": float(
            source_corr
            - crimson_corr
        ),
    }

    # --------------------------------------------------------
    # Full result
    # --------------------------------------------------------

    result = {
        "protocol": (
            "CARSBench_CRIMSON_"
            "zero_shot_cross_simulator_v1"
        ),

        "model_seeds": (
            MODEL_SEEDS
        ),

        "source_training_domain": (
            "A_typical"
        ),

        "source_training_samples": (
            100_000
        ),

        "source_validation_samples": (
            5_000
        ),

        "source_id_test": {
            "start_index": (
                ID_START
            ),

            "num_samples": (
                ID_SIZE
            ),
        },

        "external_test": {
            "name": (
                "CRIMSON NRB_removal"
            ),

            "num_samples": int(
                len(crimson_dataset)
            ),

            "training_samples_used": 0,

            "validation_samples_used": 0,

            "normalization_samples_used": 0,

            "model_selection_samples_used": 0,
        },

        "matched_spectral_window": {
            "name": (
                "fingerprint"
            ),

            "min_cm1": float(
                fingerprint_axis[0]
            ),

            "max_cm1": float(
                fingerprint_axis[-1]
            ),

            "num_points": int(
                len(
                    fingerprint_axis
                )
            ),
        },

        "preprocessing": {
            "external_axis_adapter": (
                "deterministic physical-axis "
                "linear interpolation"
            ),

            "external_amplitude_rescaling": False,

            "normalization": (
                "CARSBench source-training "
                "statistics only"
            ),
        },

        "metric_interpretation": {
            "rmse": (
                "Scale-sensitive absolute retrieval error."
            ),

            "mae": (
                "Scale-sensitive absolute retrieval error."
            ),

            "nrmse_target_rms": (
                "Scale-sensitive error divided by "
                "the target RMS amplitude."
            ),

            "spectral_angle": (
                "Shape-oriented metric invariant to "
                "positive multiplicative scaling."
            ),

            "pearson": (
                "Shape-oriented metric invariant to "
                "additive offset and positive scaling."
            ),

            "prediction_to_target_rms_ratio": (
                "Amplitude diagnostic; 1 means matched RMS amplitude."
            ),
        },

        "source_normalized_input_shift": {
            "source_id": (
                id_input_shift
            ),

            "crimson": (
                crimson_input_shift
            ),
        },

        "per_seed": (
            per_seed
        ),

        "aggregate_source_id": (
            source_aggregate
        ),

        "aggregate_crimson": (
            crimson_aggregate
        ),

        "cross_simulator_degradation": (
            degradation
        ),
    }

    json_path = (
        OUTPUT_ROOT
        / "cross_simulator_crimson_evaluation.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    # --------------------------------------------------------
    # CSV summary
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_ROOT
        / "cross_simulator_crimson_summary.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "seed",
                "dataset",
                "rmse",
                "mae",
                "spectral_angle",
                "pearson",
                "nrmse_target_rms",
                "prediction_to_target_rms_ratio",
            ]
        )

        for seed in MODEL_SEEDS:

            for dataset_name in [
                "source_id",
                "crimson",
            ]:

                metrics = (
                    per_seed[
                        str(seed)
                    ][
                        dataset_name
                    ][
                        "metrics"
                    ]
                )

                writer.writerow(
                    [
                        seed,
                        dataset_name,
                        metrics[
                            "rmse"
                        ][
                            "mean"
                        ],
                        metrics[
                            "mae"
                        ][
                            "mean"
                        ],
                        metrics[
                            "spectral_angle"
                        ][
                            "mean"
                        ],
                        metrics[
                            "pearson"
                        ][
                            "mean"
                        ],
                        metrics[
                            "nrmse_target_rms"
                        ][
                            "mean"
                        ],
                        metrics[
                            "prediction_to_target_rms_ratio"
                        ][
                            "mean"
                        ],
                    ]
                )

    # --------------------------------------------------------
    # Final concise report
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "CROSS-SIMULATOR SUMMARY"
    )
    print("=" * 76)

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "pearson",
        "nrmse_target_rms",
        "prediction_to_target_rms_ratio",
    ]:

        source = (
            source_aggregate[
                metric
            ]
        )

        crimson = (
            crimson_aggregate[
                metric
            ]
        )

        print()
        print(metric)

        print(
            "  Source ID:",
            f"{source['mean_across_model_seeds']:.6f}",
            "±",
            f"{source['std_across_model_seeds']:.6f}",
        )

        print(
            "  CRIMSON:  ",
            f"{crimson['mean_across_model_seeds']:.6f}",
            "±",
            f"{crimson['std_across_model_seeds']:.6f}",
        )

    print()
    print("Degradation ratios")

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "nrmse_target_rms",
    ]:
        print(
            f"  {metric}:",
            f"{degradation[metric]['crimson_over_source_ratio']:.3f}x",
        )

    print()
    print(
        "Pearson absolute drop:",
        f"{degradation['pearson']['absolute_drop']:.6f}",
    )

    print()
    print("Saved:")
    print(json_path)
    print(csv_path)

    print()
    print(
        "ZERO-SHOT CROSS-SIMULATOR "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
