from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from CARSBench.ml import (
    CARSBaselineCNN,
    ChunkedNPZDataset,
    TrainingConfig,
    evaluate_model,
    fit_normalization,
    save_training_artifacts,
    seed_everything,
    select_device,
    train_with_early_stopping,
)


ROOT = Path(__file__).resolve().parents[1]

GENERATOR = (
    ROOT
    / "scripts"
    / "01_generate_full_dataset.py"
)

OOD_DOMAINS = [
    "B_high_res",
    "C_low_res",
    "D_high_noise",
    "E_calibration_shift",
    "F_window_shift",
    "G_system_response_shift",
    "J_compound_acquisition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a small end-to-end CARSBench baseline smoke experiment."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "carsbench_baseline_smoke",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "baseline_smoke",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--test-samples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        default=101,
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
        help=(
            "Reuse an already generated smoke dataset."
        ),
    )

    return parser.parse_args()


def run_generator(
    *,
    output_root: Path,
    domains: list[str],
    samples: int,
    start_index: int,
    chunk_size: int,
    dataset_seed: int,
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
        str(dataset_seed),
        "--overwrite",
        "--domains",
        *domains,
    ]

    print()
    print(
        "Running:"
    )
    print(
        " ".join(
            command
        )
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def generate_smoke_dataset(
    args: argparse.Namespace,
) -> None:
    acquisition_root = (
        args.data_root
        / "acquisition"
    )

    # ---------------------------------------------------------
    # A source train
    # indices [0, train_samples)
    # ---------------------------------------------------------
    run_generator(
        output_root=(
            acquisition_root
            / "train"
        ),
        domains=[
            "A_typical"
        ],
        samples=args.train_samples,
        start_index=0,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # ---------------------------------------------------------
    # A validation
    # kept far outside the training prefix
    # ---------------------------------------------------------
    run_generator(
        output_root=(
            acquisition_root
            / "validation"
        ),
        domains=[
            "A_typical"
        ],
        samples=args.validation_samples,
        start_index=100000,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # ---------------------------------------------------------
    # A ID test
    # ---------------------------------------------------------
    run_generator(
        output_root=(
            acquisition_root
            / "id_test"
        ),
        domains=[
            "A_typical"
        ],
        samples=args.test_samples,
        start_index=110000,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # ---------------------------------------------------------
    # Acquisition OOD tests
    # ---------------------------------------------------------
    run_generator(
        output_root=(
            acquisition_root
            / "ood"
        ),
        domains=OOD_DOMAINS,
        samples=args.test_samples,
        start_index=0,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )


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


def print_evaluation(
    name: str,
    evaluation: dict,
) -> None:
    rmse = metric_mean(
        evaluation,
        "rmse",
    )

    mae = metric_mean(
        evaluation,
        "mae",
    )

    angle = metric_mean(
        evaluation,
        "spectral_angle",
    )

    print(
        f"{name:28s} "
        f"RMSE={rmse:.6f}  "
        f"MAE={mae:.6f}  "
        f"angle={angle:.6f}"
    )

    if (
        "spectral_regions"
        in evaluation
    ):
        regions = evaluation[
            "spectral_regions"
        ]

        observed = regions[
            "observed"
        ]

        unobserved = regions[
            "unobserved"
        ]

        print(
            " " * 4
            + "observed   "
            + f"RMSE={observed['rmse']['mean']:.6f}  "
            + f"MAE={observed['mae']['mean']:.6f}  "
            + f"angle={observed['spectral_angle']['mean']:.6f}"
        )

        print(
            " " * 4
            + "unobserved "
            + f"RMSE={unobserved['rmse']['mean']:.6f}  "
            + f"MAE={unobserved['mae']['mean']:.6f}  "
            + f"angle={unobserved['spectral_angle']['mean']:.6f}"
        )


def main() -> None:
    args = parse_args()

    if args.train_samples <= 0:
        raise ValueError(
            "--train-samples must be positive."
        )

    if args.validation_samples <= 0:
        raise ValueError(
            "--validation-samples must be positive."
        )

    if args.test_samples <= 0:
        raise ValueError(
            "--test-samples must be positive."
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be positive."
        )

    if not args.skip_generation:
        generate_smoke_dataset(
            args
        )

    acquisition_root = (
        args.data_root
        / "acquisition"
    )

    # =========================================================
    # Load source datasets
    # =========================================================
    train_dataset = ChunkedNPZDataset(
        acquisition_root
        / "train"
        / "A_typical"
    )

    validation_dataset = ChunkedNPZDataset(
        acquisition_root
        / "validation"
        / "A_typical"
    )

    id_test_dataset = ChunkedNPZDataset(
        acquisition_root
        / "id_test"
        / "A_typical"
    )

    # =========================================================
    # Fit normalization ONLY on source training data
    # =========================================================
    print()
    print(
        "Fitting source-training normalization..."
    )

    normalization = fit_normalization(
        train_dataset,
        num_samples=args.train_samples,
    )

    print(
        json.dumps(
            normalization.to_dict(),
            indent=2,
        )
    )

    # =========================================================
    # Device + model
    # =========================================================
    device = select_device(
        args.device
    )

    print()
    print(
        f"Device: {device}"
    )

    seed_everything(
        args.model_seed
    )

    model = CARSBaselineCNN(
        in_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
    )

    # =========================================================
    # Smoke-training configuration
    #
    # IMPORTANT:
    # These reduced epoch/patience values are only for engineering
    # validation. They do NOT replace the full baseline protocol.
    # =========================================================
    training_config = TrainingConfig(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=64,
        maximum_epochs=12,
        gradient_clip_norm=1.0,
        early_stopping_patience=4,
        early_stopping_min_delta=1.0e-5,
    )

    run_root = (
        args.output_root
        / (
            f"A_train{args.train_samples}"
            f"_seed{args.model_seed}"
        )
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Train using A train and A validation only
    # =========================================================
    print()
    print(
        "Training baseline..."
    )

    training_result = train_with_early_stopping(
        model,
        train_dataset,
        validation_dataset,
        normalization,
        model_seed=args.model_seed,
        num_train_samples=args.train_samples,
        config=training_config,
        device=device,
        checkpoint_path=(
            run_root
            / "best_checkpoint.pt"
        ),
    )

    print()
    print(
        "Training result:"
    )

    print(
        json.dumps(
            training_result.to_dict(),
            indent=2,
        )
    )

    if (
        training_result.best_epoch
        < 0
    ):
        raise RuntimeError(
            "No valid best epoch was recorded."
        )

    if not np.isfinite(
        training_result.best_validation_mse
    ):
        raise RuntimeError(
            "Best validation MSE is non-finite."
        )

    # =========================================================
    # Save trained model + normalization + training metadata
    # =========================================================
    save_training_artifacts(
        run_root,
        model=model,
        normalization_stats=normalization,
        training_config=training_config,
        training_result=training_result,
        model_seed=args.model_seed,
    )

    # =========================================================
    # Evaluate source ID test
    # =========================================================
    evaluations: dict[str, dict] = {}

    evaluations[
        "A_typical_id_test"
    ] = evaluate_model(
        model,
        id_test_dataset,
        normalization,
        device=device,
        batch_size=64,
    )

    # =========================================================
    # Evaluate every acquisition OOD domain
    #
    # IMPORTANT:
    # The same A-training normalization is reused everywhere.
    # No OOD statistics are fitted.
    # =========================================================
    for domain_name in OOD_DOMAINS:
        dataset = ChunkedNPZDataset(
            acquisition_root
            / "ood"
            / domain_name
        )

        evaluations[
            domain_name
        ] = evaluate_model(
            model,
            dataset,
            normalization,
            device=device,
            batch_size=64,
        )

    # =========================================================
    # Save evaluation results
    # =========================================================
    evaluation_path = (
        run_root
        / "evaluation.json"
    )

    with evaluation_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            evaluations,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    # =========================================================
    # Human-readable summary
    # =========================================================
    print()
    print(
        "=" * 80
    )

    print(
        "SMOKE EVALUATION"
    )

    print(
        "=" * 80
    )

    print_evaluation(
        "A_typical ID",
        evaluations[
            "A_typical_id_test"
        ],
    )

    for domain_name in OOD_DOMAINS:
        print_evaluation(
            domain_name,
            evaluations[
                domain_name
            ],
        )

    print()
    print(
        f"Artifacts written to: {run_root}"
    )

    print(
        f"Evaluation written to: {evaluation_path}"
    )


if __name__ == "__main__":
    main()