from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from CARSBench.ml import (
    CARSBaselineCNN,
    ChunkedNPZDataset,
    NormalizationStats,
    evaluate_model,
    select_device,
)


ROOT = Path(__file__).resolve().parents[1]

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)


DEFAULT_MODEL_ROOT = (
    ROOT
    / "results"
    / "chemistry_smoke"
    / "H_train256_seed101"
)


DIAGNOSTIC_DOMAINS = {
    "H_biochemical_source": 110000,
    "I_biochemical_target": 0,
    "K_joint_ood": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the strength of the CARSBench chemistry OOD shift "
            "using a fixed trained baseline model."
        )
    )

    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            ROOT
            / "data"
            / "carsbench_chemistry_diagnostic"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "results"
            / "chemistry_diagnostic"
        ),
    )

    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "mps",
            "cuda",
        ],
        default="auto",
    )

    parser.add_argument(
        "--skip-generation",
        action="store_true",
    )

    return parser.parse_args()


def generate_domain(
    *,
    output_root: Path,
    domain: str,
    samples: int,
    start_index: int,
    chunk_size: int,
    seed: int,
) -> None:
    command = [
        sys.executable,
        str(GENERATOR),

        "--output-root",
        str(output_root),

        "--samples-per-domain",
        str(samples),

        "--start-index",
        str(start_index),

        "--chunk-size",
        str(chunk_size),

        "--seed",
        str(seed),

        "--overwrite",

        "--domains",
        domain,
    ]

    print()
    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "median": float(
            np.median(values)
        ),
        "std": float(
            np.std(values)
        ),
        "p05": float(
            np.percentile(
                values,
                5.0,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95.0,
            )
        ),
    }


def target_statistics(
    dataset: ChunkedNPZDataset,
) -> dict:
    """
    Characterize Raman-target scale/energy.

    This helps determine whether one domain appears easier simply
    because its targets have systematically smaller amplitudes.
    """
    rms_values = []
    max_values = []
    mean_values = []
    l1_values = []

    for index in range(
        len(dataset)
    ):
        target = np.asarray(
            dataset[index][
                "target"
            ],
            dtype=np.float64,
        ).reshape(
            -1
        )

        rms_values.append(
            float(
                np.sqrt(
                    np.mean(
                        target
                        * target
                    )
                )
            )
        )

        max_values.append(
            float(
                np.max(
                    np.abs(
                        target
                    )
                )
            )
        )

        mean_values.append(
            float(
                np.mean(
                    target
                )
            )
        )

        l1_values.append(
            float(
                np.mean(
                    np.abs(
                        target
                    )
                )
            )
        )

    return {
        "target_rms": summarize(
            np.asarray(
                rms_values
            )
        ),

        "target_max_abs": summarize(
            np.asarray(
                max_values
            )
        ),

        "target_mean": summarize(
            np.asarray(
                mean_values
            )
        ),

        "target_mean_abs": summarize(
            np.asarray(
                l1_values
            )
        ),
    }


def metric_mean(
    evaluation: dict,
    metric: str,
) -> float:
    return float(
        evaluation[
            "metrics"
        ][
            metric
        ][
            "mean"
        ]
    )


def print_result(
    domain_name: str,
    evaluation: dict,
    target_stats: dict,
) -> None:
    print()
    print(
        domain_name
    )

    print(
        "-" * len(
            domain_name
        )
    )

    print(
        "Retrieval:"
    )

    print(
        f"  RMSE           "
        f"{metric_mean(evaluation, 'rmse'):.6f}"
    )

    print(
        f"  MAE            "
        f"{metric_mean(evaluation, 'mae'):.6f}"
    )

    print(
        f"  spectral angle "
        f"{metric_mean(evaluation, 'spectral_angle'):.6f}"
    )

    print(
        "Target distribution:"
    )

    print(
        f"  RMS mean       "
        f"{target_stats['target_rms']['mean']:.6f}"
    )

    print(
        f"  max|target|    "
        f"{target_stats['target_max_abs']['mean']:.6f}"
    )

    print(
        f"  mean|target|   "
        f"{target_stats['target_mean_abs']['mean']:.6f}"
    )


def main() -> None:
    args = parse_args()

    if args.samples_per_domain <= 0:
        raise ValueError(
            "--samples-per-domain must be positive."
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be positive."
        )

    # =========================================================
    # Generate larger independent evaluation sets
    # =========================================================
    if not args.skip_generation:
        for (
            domain_name,
            start_index,
        ) in DIAGNOSTIC_DOMAINS.items():
            generate_domain(
                output_root=args.data_root,
                domain=domain_name,
                samples=args.samples_per_domain,
                start_index=start_index,
                chunk_size=args.chunk_size,
                seed=args.dataset_seed,
            )

    # =========================================================
    # Load normalization from the ORIGINAL H training run
    # =========================================================
    normalization_path = (
        args.model_root
        / "normalization.json"
    )

    if not normalization_path.exists():
        raise FileNotFoundError(
            normalization_path
        )

    with normalization_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        normalization = (
            NormalizationStats.from_dict(
                json.load(
                    handle
                )
            )
        )

    # =========================================================
    # Load fixed trained model
    # =========================================================
    model_path = (
        args.model_root
        / "model.pt"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    device = select_device(
        args.device
    )

    print()
    print(
        f"Device: {device}"
    )

    model = CARSBaselineCNN(
        in_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
    )

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    # =========================================================
    # Evaluate
    # =========================================================
    results = {}

    for domain_name in DIAGNOSTIC_DOMAINS:
        dataset = ChunkedNPZDataset(
            args.data_root
            / domain_name
        )

        evaluation = evaluate_model(
            model,
            dataset,
            normalization,
            device=device,
            batch_size=64,
        )

        target_stats = target_statistics(
            dataset
        )

        results[
            domain_name
        ] = {
            "evaluation": evaluation,
            "target_statistics": (
                target_stats
            ),
        }

        print_result(
            domain_name,
            evaluation,
            target_stats,
        )

    # =========================================================
    # Direct H-versus-I comparison
    # =========================================================
    h = results[
        "H_biochemical_source"
    ][
        "evaluation"
    ]

    i = results[
        "I_biochemical_target"
    ][
        "evaluation"
    ]

    h_rmse = metric_mean(
        h,
        "rmse",
    )

    i_rmse = metric_mean(
        i,
        "rmse",
    )

    h_angle = metric_mean(
        h,
        "spectral_angle",
    )

    i_angle = metric_mean(
        i,
        "spectral_angle",
    )

    comparison = {
        "I_minus_H_rmse": (
            i_rmse
            - h_rmse
        ),

        "I_over_H_rmse": (
            i_rmse
            / h_rmse
        ),

        "I_minus_H_spectral_angle": (
            i_angle
            - h_angle
        ),

        "I_over_H_spectral_angle": (
            i_angle
            / h_angle
        ),
    }

    results[
        "H_vs_I_comparison"
    ] = comparison

    print()
    print(
        "=" * 80
    )

    print(
        "H vs I CHEMISTRY-SHIFT COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        f"I - H RMSE: "
        f"{comparison['I_minus_H_rmse']:.6f}"
    )

    print(
        f"I / H RMSE: "
        f"{comparison['I_over_H_rmse']:.4f}"
    )

    print(
        f"I - H spectral angle: "
        f"{comparison['I_minus_H_spectral_angle']:.6f}"
    )

    print(
        f"I / H spectral angle: "
        f"{comparison['I_over_H_spectral_angle']:.4f}"
    )

    # =========================================================
    # Save
    # =========================================================
    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_root
        / "diagnostic.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print()
    print(
        f"Saved diagnostic: {output_path}"
    )


if __name__ == "__main__":
    main()