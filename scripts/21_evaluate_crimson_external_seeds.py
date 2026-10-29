
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from CARSBench.ml import select_device


ROOT = Path(__file__).resolve().parents[1]

SCRIPT18_PATH = (
    ROOT
    / "scripts"
    / "18_evaluate_crimson_cross_simulator.py"
)

MODEL_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "fingerprint_source"
)

CRIMSON_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "crimson"
)

EVAL_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "evaluation"
)

SOURCE_EVAL_JSON = (
    EVAL_ROOT
    / "cross_simulator_crimson_evaluation.json"
)

OUTPUT_JSON = (
    EVAL_ROOT
    / "cross_simulator_multiseed_evaluation.json"
)

MODEL_SEEDS = [
    101,
    202,
    303,
]

EXTERNAL_SEEDS = [
    42,
    43,
    44,
]

METRICS = [
    "rmse",
    "mae",
    "spectral_angle",
    "pearson",
    "nrmse_target_rms",
    "prediction_to_target_rms_ratio",
]


def load_module(
    path: Path,
    name: str,
):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name,
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


def adapted_path(
    external_seed: int,
) -> Path:

    return (
        CRIMSON_ROOT
        / (
            "crimson_case5b_n5000_"
            f"seed{external_seed}_"
            "carsbench_fingerprint.npz"
        )
    )


def summarize_across(
    values,
):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(x)
        ),
        "std": float(
            np.std(x)
        ),
        "minimum": float(
            np.min(x)
        ),
        "maximum": float(
            np.max(x)
        ),
        "values": [
            float(v)
            for v in x
        ],
    }


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

    print("=" * 78)
    print(
        "CARSBENCH MULTI-SEED "
        "CROSS-SIMULATOR EVALUATION"
    )
    print("=" * 78)

    print()
    print("Device:", device)

    # ========================================================
    # Reuse the already frozen evaluation implementation.
    # ========================================================

    eval18 = load_module(
        SCRIPT18_PATH,
        "carsbench_cross_simulator_eval18",
    )

    # ========================================================
    # Source ID reference from the completed strict evaluation.
    # ========================================================

    if not SOURCE_EVAL_JSON.exists():
        raise FileNotFoundError(
            SOURCE_EVAL_JSON
        )

    with SOURCE_EVAL_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        source_eval = json.load(
            handle
        )

    source_id = (
        source_eval[
            "aggregate_source_id"
        ]
    )

    # ========================================================
    # Load and verify source normalizations.
    # ========================================================

    stats_by_model_seed = {}

    for model_seed in MODEL_SEEDS:

        run_root = (
            MODEL_ROOT
            / (
                "A_fingerprint_"
                f"train100000_seed{model_seed}"
            )
        )

        stats_by_model_seed[
            model_seed
        ] = (
            eval18.load_normalization(
                run_root
            )
        )

    reference_stats = (
        stats_by_model_seed[
            MODEL_SEEDS[0]
        ]
    )

    reference_signature = (
        eval18.normalization_signature(
            reference_stats
        )
    )

    for model_seed in MODEL_SEEDS[1:]:

        signature = (
            eval18.normalization_signature(
                stats_by_model_seed[
                    model_seed
                ]
            )
        )

        if not np.array_equal(
            signature,
            reference_signature,
        ):
            raise RuntimeError(
                "Normalization differs "
                "between model seeds."
            )

    print()
    print(
        "Source normalization identical "
        "across model seeds: PASS"
    )

    # ========================================================
    # Results containers.
    # ========================================================

    per_combination = {}

    input_diagnostics = {}

    # ========================================================
    # Loop over independent external simulator datasets.
    # ========================================================

    for external_seed in EXTERNAL_SEEDS:

        print()
        print("=" * 78)
        print(
            f"CRIMSON EXTERNAL SEED {external_seed}"
        )
        print("=" * 78)

        path = adapted_path(
            external_seed
        )

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        dataset = (
            eval18.ExternalDataset(
                path
            )
        )

        if len(dataset) != 5000:
            raise RuntimeError(
                f"Seed {external_seed}: "
                "expected 5000 samples."
            )

        arrays = (
            eval18.collect_dataset_arrays(
                dataset
            )
        )

        if arrays[
            "spectrum"
        ].shape != (
            5000,
            512,
        ):
            raise RuntimeError(
                f"Seed {external_seed}: "
                "unexpected spectrum shape."
            )

        if not np.all(
            arrays["mask"] == 1.0
        ):
            raise RuntimeError(
                f"Seed {external_seed}: "
                "mask is not fully observed."
            )

        # ----------------------------------------------------
        # Apply exactly the same SOURCE normalization.
        # ----------------------------------------------------

        (
            inputs,
            normalized_spectra,
        ) = eval18.normalized_inputs(
            arrays["spectrum"],
            arrays["mask"],
            reference_stats,
        )

        input_diagnostic = (
            eval18.input_shift_summary(
                normalized_spectra,
                arrays["mask"],
            )
        )

        input_diagnostics[
            str(external_seed)
        ] = input_diagnostic

        print()
        print(
            "Source-normalized input diagnostic:"
        )

        print(
            json.dumps(
                input_diagnostic,
                indent=2,
            )
        )

        per_combination[
            str(external_seed)
        ] = {}

        # ----------------------------------------------------
        # Evaluate all three frozen source models.
        # ----------------------------------------------------

        for model_seed in MODEL_SEEDS:

            print()
            print(
                f"Model seed {model_seed}"
            )

            run_root = (
                MODEL_ROOT
                / (
                    "A_fingerprint_"
                    f"train100000_seed{model_seed}"
                )
            )

            model = eval18.load_model(
                run_root,
                device,
            )

            stats = (
                stats_by_model_seed[
                    model_seed
                ]
            )

            prediction = (
                eval18.predict(
                    model,
                    inputs,
                    stats,
                    device,
                )
            )

            metrics = (
                eval18.compute_metrics(
                    prediction,
                    arrays["target"],
                )
            )

            metric_summary = (
                eval18.summarize_metrics(
                    metrics
                )
            )

            per_combination[
                str(external_seed)
            ][
                str(model_seed)
            ] = {
                "metrics": (
                    metric_summary
                )
            }

            # Save frozen-model predictions so later
            # diagnostics do not require rerunning inference.
            np.savez_compressed(
                EVAL_ROOT
                / (
                    f"model_seed{model_seed}_"
                    f"crimson_seed{external_seed}.npz"
                ),

                axis=arrays["axis"],

                sample_index=arrays[
                    "sample_index"
                ],

                prediction=prediction,

                **{
                    name: np.asarray(
                        values,
                        dtype=np.float32,
                    )
                    for name, values
                    in metrics.items()
                },
            )

            print(
                "  RMSE:",
                f"{metric_summary['rmse']['mean']:.6f}",
            )

            print(
                "  MAE:",
                f"{metric_summary['mae']['mean']:.6f}",
            )

            print(
                "  Angle:",
                f"{metric_summary['spectral_angle']['mean']:.6f}",
            )

            print(
                "  Pearson:",
                f"{metric_summary['pearson']['mean']:.6f}",
            )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ========================================================
    # External-seed summaries:
    # average across MODEL seeds first.
    #
    # These are the important values for estimating variation
    # due to independently generated external datasets.
    # ========================================================

    external_seed_summary = {}

    for external_seed in EXTERNAL_SEEDS:

        external_seed_summary[
            str(external_seed)
        ] = {}

        for metric in METRICS:

            model_values = [
                per_combination[
                    str(external_seed)
                ][
                    str(model_seed)
                ][
                    "metrics"
                ][
                    metric
                ][
                    "mean"
                ]
                for model_seed
                in MODEL_SEEDS
            ]

            external_seed_summary[
                str(external_seed)
            ][
                metric
            ] = summarize_across(
                model_values
            )

    # ========================================================
    # Model-seed summaries:
    # average each model across independent external sets.
    # ========================================================

    model_seed_summary = {}

    for model_seed in MODEL_SEEDS:

        model_seed_summary[
            str(model_seed)
        ] = {}

        for metric in METRICS:

            external_values = [
                per_combination[
                    str(external_seed)
                ][
                    str(model_seed)
                ][
                    "metrics"
                ][
                    metric
                ][
                    "mean"
                ]
                for external_seed
                in EXTERNAL_SEEDS
            ]

            model_seed_summary[
                str(model_seed)
            ][
                metric
            ] = summarize_across(
                external_values
            )

    # ========================================================
    # Primary external-sampling summary.
    #
    # Each external dataset contributes ONE value:
    # its metric averaged across the three source models.
    #
    # Therefore std here describes external-sampling
    # variability rather than treating all 9 combinations
    # as independent replicates.
    # ========================================================

    primary_summary = {}

    for metric in METRICS:

        external_seed_means = [
            external_seed_summary[
                str(external_seed)
            ][
                metric
            ][
                "mean"
            ]
            for external_seed
            in EXTERNAL_SEEDS
        ]

        primary_summary[
            metric
        ] = summarize_across(
            external_seed_means
        )

    # ========================================================
    # Descriptive model-seed variability.
    # ========================================================

    model_variability = {}

    for metric in METRICS:

        model_means = [
            model_seed_summary[
                str(model_seed)
            ][
                metric
            ][
                "mean"
            ]
            for model_seed
            in MODEL_SEEDS
        ]

        model_variability[
            metric
        ] = summarize_across(
            model_means
        )

    # ========================================================
    # Compare external result to matched CARSBench source ID.
    # ========================================================

    degradation = {}

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "nrmse_target_rms",
    ]:

        source_value = float(
            source_id[
                metric
            ][
                "mean_across_model_seeds"
            ]
        )

        external_value = float(
            primary_summary[
                metric
            ][
                "mean"
            ]
        )

        degradation[
            metric
        ] = {
            "source_id": source_value,

            "crimson_multiseed": (
                external_value
            ),

            "ratio": float(
                external_value
                / source_value
            ),
        }

    source_pearson = float(
        source_id[
            "pearson"
        ][
            "mean_across_model_seeds"
        ]
    )

    external_pearson = float(
        primary_summary[
            "pearson"
        ][
            "mean"
        ]
    )

    degradation[
        "pearson"
    ] = {
        "source_id": (
            source_pearson
        ),

        "crimson_multiseed": (
            external_pearson
        ),

        "absolute_drop": float(
            source_pearson
            - external_pearson
        ),
    }

    # ========================================================
    # Save formal result.
    # ========================================================

    result = {
        "protocol": (
            "CARSBench_CRIMSON_"
            "cross_simulator_multiseed_v1"
        ),

        "model_seeds": (
            MODEL_SEEDS
        ),

        "external_generation_seeds": (
            EXTERNAL_SEEDS
        ),

        "num_model_seeds": (
            len(MODEL_SEEDS)
        ),

        "num_external_seeds": (
            len(EXTERNAL_SEEDS)
        ),

        "num_crossed_evaluations": (
            len(MODEL_SEEDS)
            * len(EXTERNAL_SEEDS)
        ),

        "samples_per_external_seed": 5000,

        "external_samples_total": (
            5000
            * len(EXTERNAL_SEEDS)
        ),

        "training_or_tuning_on_crimson": False,

        "preprocessing_changed_after_seed42_result": False,

        "input_diagnostics": (
            input_diagnostics
        ),

        "per_combination": (
            per_combination
        ),

        "external_seed_summary": (
            external_seed_summary
        ),

        "model_seed_summary": (
            model_seed_summary
        ),

        "primary_external_sampling_summary": (
            primary_summary
        ),

        "descriptive_model_seed_variability": (
            model_variability
        ),

        "source_id_reference": (
            source_id
        ),

        "cross_simulator_degradation": (
            degradation
        ),

        "statistical_note": (
            "The primary external-sampling summary "
            "first averages each external generation "
            "seed across the three model seeds and then "
            "summarizes the three independent external "
            "seed means. The nine crossed combinations "
            "are not treated as nine independent "
            "experimental replicates."
        ),
    }

    EVAL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
            sort_keys=True,
        )

    # ========================================================
    # Console report.
    # ========================================================

    print()
    print("=" * 78)
    print(
        "EXTERNAL-SEED RESULTS "
        "(MEAN ACROSS 3 MODEL SEEDS)"
    )
    print("=" * 78)

    for external_seed in EXTERNAL_SEEDS:

        print()
        print(
            f"CRIMSON seed {external_seed}"
        )

        for metric in METRICS:

            summary = (
                external_seed_summary[
                    str(external_seed)
                ][
                    metric
                ]
            )

            print(
                f"  {metric}: "
                f"{summary['mean']:.6f} "
                f"± {summary['std']:.6f}"
            )

    print()
    print("=" * 78)
    print(
        "PRIMARY MULTI-SEED "
        "CROSS-SIMULATOR SUMMARY"
    )
    print("=" * 78)

    for metric in METRICS:

        summary = (
            primary_summary[
                metric
            ]
        )

        print()
        print(
            metric
        )

        print(
            "  Mean across external seeds:",
            f"{summary['mean']:.6f}",
        )

        print(
            "  External-seed std:",
            f"{summary['std']:.6f}",
        )

        print(
            "  External-seed values:",
            ", ".join(
                f"{value:.6f}"
                for value
                in summary["values"]
            ),
        )

    print()
    print("=" * 78)
    print(
        "DEGRADATION VS MATCHED SOURCE ID"
    )
    print("=" * 78)

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "nrmse_target_rms",
    ]:

        print(
            f"{metric}: "
            f"{degradation[metric]['ratio']:.3f}x"
        )

    print(
        "Pearson absolute drop:",
        f"{degradation['pearson']['absolute_drop']:.6f}",
    )

    print()
    print(
        "Saved:",
        OUTPUT_JSON,
    )

    print()
    print(
        "MULTI-SEED CROSS-SIMULATOR "
        "EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
