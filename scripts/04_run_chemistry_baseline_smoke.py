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
    "I_biochemical_target",
    "K_joint_ood",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the CARSBench chemistry-track baseline smoke experiment."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            ROOT
            / "data"
            / "carsbench_chemistry_smoke"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            ROOT
            / "results"
            / "chemistry_smoke"
        ),
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
    print("Running:")
    print(" ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def generate_smoke_dataset(
    args: argparse.Namespace,
) -> None:
    chemistry_root = (
        args.data_root
        / "chemistry"
    )

    # H source training
    run_generator(
        output_root=(
            chemistry_root
            / "train"
        ),
        domains=[
            "H_biochemical_source"
        ],
        samples=args.train_samples,
        start_index=0,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # H validation
    run_generator(
        output_root=(
            chemistry_root
            / "validation"
        ),
        domains=[
            "H_biochemical_source"
        ],
        samples=args.validation_samples,
        start_index=100000,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # H ID test
    run_generator(
        output_root=(
            chemistry_root
            / "id_test"
        ),
        domains=[
            "H_biochemical_source"
        ],
        samples=args.test_samples,
        start_index=110000,
        chunk_size=args.chunk_size,
        dataset_seed=args.dataset_seed,
    )

    # I chemistry OOD + K joint OOD
    run_generator(
        output_root=(
            chemistry_root
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
    print(
        f"{name:28s} "
        f"RMSE={metric_mean(evaluation, 'rmse'):.6f}  "
        f"MAE={metric_mean(evaluation, 'mae'):.6f}  "
        f"angle={metric_mean(evaluation, 'spectral_angle'):.6f}"
    )


def main() -> None:
    args = parse_args()

    for name, value in [
        (
            "train_samples",
            args.train_samples,
        ),
        (
            "validation_samples",
            args.validation_samples,
        ),
        (
            "test_samples",
            args.test_samples,
        ),
        (
            "chunk_size",
            args.chunk_size,
        ),
    ]:
        if value <= 0:
            raise ValueError(
                f"{name} must be positive."
            )

    if not args.skip_generation:
        generate_smoke_dataset(
            args
        )

    chemistry_root = (
        args.data_root
        / "chemistry"
    )

    # =========================================================
    # Source H datasets
    # =========================================================
    train_dataset = ChunkedNPZDataset(
        chemistry_root
        / "train"
        / "H_biochemical_source"
    )

    validation_dataset = ChunkedNPZDataset(
        chemistry_root
        / "validation"
        / "H_biochemical_source"
    )

    id_test_dataset = ChunkedNPZDataset(
        chemistry_root
        / "id_test"
        / "H_biochemical_source"
    )

    # =========================================================
    # Fit normalization using H TRAIN ONLY
    # =========================================================
    print()
    print(
        "Fitting H source-training normalization..."
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
    # Device and model
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

    # Engineering smoke configuration only.
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
            f"H_train{args.train_samples}"
            f"_seed{args.model_seed}"
        )
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Train only on H.
    # Validation only on H.
    # =========================================================
    print()
    print(
        "Training chemistry baseline..."
    )

    training_result = train_with_early_stopping(
        model,
        train_dataset,
        validation_dataset,
        normalization,

        model_seed=args.model_seed,

        num_train_samples=(
            args.train_samples
        ),

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

    if training_result.best_epoch < 0:
        raise RuntimeError(
            "No valid best epoch was recorded."
        )

    if not np.isfinite(
        training_result.best_validation_mse
    ):
        raise RuntimeError(
            "Best validation MSE is non-finite."
        )

    save_training_artifacts(
        run_root,

        model=model,

        normalization_stats=(
            normalization
        ),

        training_config=(
            training_config
        ),

        training_result=(
            training_result
        ),

        model_seed=(
            args.model_seed
        ),
    )

    # =========================================================
    # ID evaluation
    # =========================================================
    evaluations: dict[str, dict] = {}

    evaluations[
        "H_biochemical_source_id_test"
    ] = evaluate_model(
        model,
        id_test_dataset,
        normalization,
        device=device,
        batch_size=64,
    )

    # =========================================================
    # Chemistry OOD / Joint OOD
    #
    # Same H normalization is reused.
    # No I or K statistics are fitted.
    # =========================================================
    for domain_name in OOD_DOMAINS:
        dataset = ChunkedNPZDataset(
            chemistry_root
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

    print()
    print(
        "=" * 80
    )

    print(
        "CHEMISTRY SMOKE EVALUATION"
    )

    print(
        "=" * 80
    )

    print_evaluation(
        "H source ID",
        evaluations[
            "H_biochemical_source_id_test"
        ],
    )

    print_evaluation(
        "I held-out glycogen",
        evaluations[
            "I_biochemical_target"
        ],
    )

    print_evaluation(
        "K joint OOD",
        evaluations[
            "K_joint_ood"
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