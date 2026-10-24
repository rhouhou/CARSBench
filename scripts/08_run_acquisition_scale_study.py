from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from CARSBench.ml import (
    CARSBaselineCNN,
    ChunkedNPZDataset,
    TrainingConfig,
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

SCALE_SPEC = (
    ROOT
    / "specs"
    / "carsbench_v1_scale_study.json"
)

BASELINE_PROTOCOL = (
    ROOT
    / "specs"
    / "carsbench_v1_baseline_protocol.json"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "carsbench_v1_scale_study"
)

RESULTS_ROOT = (
    ROOT
    / "results"
    / "scale_study"
    / "acquisition"
)


SOURCE_DOMAIN = "A_typical"

TRAIN_SIZES = [
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
]

MODEL_SEEDS = [
    101,
    202,
    303,
]

DATASET_SEED = 42

TRAIN_POOL_SIZE = 100_000

VALIDATION_START = 100_000
VALIDATION_SIZE = 5_000

CHUNK_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one CARSBench acquisition-track "
            "training-size SELECTION experiment.\n\n"
            "Only source training and source validation are used. "
            "ID/OOD test sets are deliberately excluded."
        )
    )

    parser.add_argument(
        "--train-size",
        type=int,
        choices=TRAIN_SIZES,
        default=5_000,
    )

    parser.add_argument(
        "--model-seed",
        type=int,
        choices=MODEL_SEEDS,
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
        "--force-regenerate-data",
        action="store_true",
    )

    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Train again even if this run already has "
            "a completed training result."
        ),
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Prepare the reusable training pool and "
            "validation set, then exit without training."
        ),
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def run_generator(
    *,
    output_root: Path,
    domain: str,
    samples: int,
    start_index: int,
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
        str(CHUNK_SIZE),

        "--seed",
        str(DATASET_SEED),

        "--overwrite",

        "--domains",
        domain,
    ]

    print()
    print(
        "Generating dataset:"
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


def dataset_length(
    path: Path,
) -> int | None:
    if not path.exists():
        return None

    try:
        dataset = ChunkedNPZDataset(
            path
        )

        return len(
            dataset
        )

    except Exception:
        return None


def ensure_train_pool(
    *,
    force: bool,
) -> Path:
    """
    Maintain one reusable 100k A_typical training pool.

    Every scale-study training set is a deterministic prefix:

        5k ⊂ 10k ⊂ 25k ⊂ 50k ⊂ 100k

    If an older smaller pool exists, it is replaced by the
    complete 100k deterministic pool.
    """
    train_root = (
        DATA_ROOT
        / "train"
    )

    train_path = (
        train_root
        / SOURCE_DOMAIN
    )

    current_length = dataset_length(
        train_path
    )

    if (
        not force
        and current_length == TRAIN_POOL_SIZE
    ):
        print()
        print(
            f"Reusing training pool: "
            f"{TRAIN_POOL_SIZE} samples"
        )

        return train_path

    print()
    print(
        "Preparing reusable 100k source-training pool."
    )

    if current_length is not None:
        print(
            f"Existing pool contains "
            f"{current_length} samples and will be "
            f"replaced by the deterministic 100k pool."
        )

    run_generator(
        output_root=train_root,
        domain=SOURCE_DOMAIN,
        samples=TRAIN_POOL_SIZE,
        start_index=0,
    )

    final_length = dataset_length(
        train_path
    )

    if final_length != TRAIN_POOL_SIZE:
        raise RuntimeError(
            "Training-pool generation failed: "
            f"expected {TRAIN_POOL_SIZE}, "
            f"found {final_length}."
        )

    return train_path


def ensure_validation(
    *,
    force: bool,
) -> Path:
    validation_root = (
        DATA_ROOT
        / "validation"
    )

    validation_path = (
        validation_root
        / SOURCE_DOMAIN
    )

    current_length = dataset_length(
        validation_path
    )

    if (
        not force
        and current_length == VALIDATION_SIZE
    ):
        print()
        print(
            f"Reusing validation set: "
            f"{VALIDATION_SIZE} samples"
        )

        return validation_path

    run_generator(
        output_root=validation_root,
        domain=SOURCE_DOMAIN,
        samples=VALIDATION_SIZE,
        start_index=VALIDATION_START,
    )

    final_length = dataset_length(
        validation_path
    )

    if final_length != VALIDATION_SIZE:
        raise RuntimeError(
            "Validation generation failed: "
            f"expected {VALIDATION_SIZE}, "
            f"found {final_length}."
        )

    return validation_path


def verify_dataset_ranges(
    *,
    train_path: Path,
    validation_path: Path,
    train_size: int,
) -> None:
    train_dataset = ChunkedNPZDataset(
        train_path
    )

    validation_dataset = ChunkedNPZDataset(
        validation_path
    )

    if len(train_dataset) < train_size:
        raise RuntimeError(
            "Training pool is smaller than the "
            "requested training size."
        )

    if len(validation_dataset) != VALIDATION_SIZE:
        raise RuntimeError(
            "Validation set has incorrect size."
        )

    train_first = int(
        train_dataset[0][
            "sample_index"
        ]
    )

    train_last = int(
        train_dataset[
            train_size - 1
        ][
            "sample_index"
        ]
    )

    validation_first = int(
        validation_dataset[0][
            "sample_index"
        ]
    )

    validation_last = int(
        validation_dataset[
            VALIDATION_SIZE - 1
        ][
            "sample_index"
        ]
    )

    expected = {
        "train_first": 0,
        "train_last": (
            train_size - 1
        ),
        "validation_first": (
            VALIDATION_START
        ),
        "validation_last": (
            VALIDATION_START
            + VALIDATION_SIZE
            - 1
        ),
    }

    actual = {
        "train_first": (
            train_first
        ),
        "train_last": (
            train_last
        ),
        "validation_first": (
            validation_first
        ),
        "validation_last": (
            validation_last
        ),
    }

    if actual != expected:
        raise RuntimeError(
            "Dataset ranges do not match the "
            "scale-study protocol.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    print()
    print(
        "Dataset ranges verified:"
    )

    print(
        json.dumps(
            actual,
            indent=2,
        )
    )


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=64,
        maximum_epochs=100,
        gradient_clip_norm=1.0,
        early_stopping_patience=10,
        early_stopping_min_delta=1.0e-5,
    )


def copy_protocol_files(
    run_root: Path,
) -> dict[str, str]:
    hashes = {}

    for path in [
        SCALE_SPEC,
        BASELINE_PROTOCOL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

        destination = (
            run_root
            / path.name
        )

        shutil.copy2(
            path,
            destination,
        )

        hashes[
            path.name
        ] = sha256_file(
            path
        )

    return hashes


def find_key_recursive(
    value: Any,
    key: str,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        if key in value:
            return value[
                key
            ]

        for child in value.values():
            result = find_key_recursive(
                child,
                key,
            )

            if result is not None:
                return result

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            result = find_key_recursive(
                child,
                key,
            )

            if result is not None:
                return result

    return None


def read_existing_training_result(
    run_root: Path,
) -> dict | None:
    """
    Allows the already completed 5k/seed-101 formal run
    to count toward the selection study without retraining.
    """
    selection_path = (
        run_root
        / "selection_result.json"
    )

    if selection_path.exists():
        with selection_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(
                handle
            )

    training_path = (
        run_root
        / "training.json"
    )

    model_path = (
        run_root
        / "model.pt"
    )

    normalization_path = (
        run_root
        / "normalization.json"
    )

    if not (
        training_path.exists()
        and model_path.exists()
        and normalization_path.exists()
    ):
        return None

    try:
        with training_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            training_json = json.load(
                handle
            )

        best_epoch = find_key_recursive(
            training_json,
            "best_epoch",
        )

        best_validation_mse = (
            find_key_recursive(
                training_json,
                "best_validation_mse",
            )
        )

        epochs_completed = (
            find_key_recursive(
                training_json,
                "epochs_completed",
            )
        )

        stopped_early = (
            find_key_recursive(
                training_json,
                "stopped_early",
            )
        )

        if (
            best_epoch is None
            or best_validation_mse is None
        ):
            return None

        result = {
            "source": (
                "existing_completed_training_run"
            ),
            "best_epoch": int(
                best_epoch
            ),
            "best_validation_mse": float(
                best_validation_mse
            ),
            "epochs_completed": (
                None
                if epochs_completed is None
                else int(
                    epochs_completed
                )
            ),
            "stopped_early": (
                None
                if stopped_early is None
                else bool(
                    stopped_early
                )
            ),
        }

        with selection_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
                sort_keys=True,
            )

            handle.write(
                "\n"
            )

        return result

    except Exception:
        return None


def main() -> None:
    args = parse_args()

    train_size = int(
        args.train_size
    )

    model_seed = int(
        args.model_seed
    )

    print(
        "=" * 80
    )

    print(
        "CARSBench acquisition scale-study SELECTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Training size: {train_size}"
    )

    print(
        f"Model seed:    {model_seed}"
    )

    print(
        f"Dataset seed:  {DATASET_SEED}"
    )

    print()
    print(
        "IMPORTANT: ID and OOD test domains are "
        "not used in this script."
    )

    # =========================================================
    # Reusable deterministic data
    # =========================================================
    train_path = ensure_train_pool(
        force=(
            args.force_regenerate_data
        )
    )

    validation_path = ensure_validation(
        force=(
            args.force_regenerate_data
        )
    )

    verify_dataset_ranges(
        train_path=train_path,
        validation_path=validation_path,
        train_size=train_size,
    )

    if args.prepare_only:
        print()
        print(
            "Data preparation complete."
        )

        print(
            "No model was trained."
        )

        return

    # =========================================================
    # Run location
    # =========================================================
    run_root = (
        RESULTS_ROOT
        / (
            f"A_train{train_size}"
            f"_seed{model_seed}"
        )
    )

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Reuse a completed run when possible.
    #
    # In particular, the existing 5k/seed-101 run is valid:
    # its model selection used only A validation.
    # OOD evaluation happened only afterward.
    # =========================================================
    if not args.force_rerun:
        existing = (
            read_existing_training_result(
                run_root
            )
        )

        if existing is not None:
            print()
            print(
                "=" * 80
            )

            print(
                "COMPLETED RUN REUSED"
            )

            print(
                "=" * 80
            )

            print(
                json.dumps(
                    existing,
                    indent=2,
                )
            )

            print()
            print(
                "No training was repeated."
            )

            return

    train_dataset = ChunkedNPZDataset(
        train_path
    )

    validation_dataset = (
        ChunkedNPZDataset(
            validation_path
        )
    )

    # =========================================================
    # Normalization fitted on the requested training prefix only
    # =========================================================
    print()
    print(
        "Fitting source-training normalization..."
    )

    normalization_start = (
        time.perf_counter()
    )

    normalization = fit_normalization(
        train_dataset,
        num_samples=train_size,
    )

    normalization_seconds = (
        time.perf_counter()
        - normalization_start
    )

    print(
        json.dumps(
            normalization.to_dict(),
            indent=2,
        )
    )

    print(
        f"Normalization time: "
        f"{normalization_seconds:.1f} s"
    )

    # =========================================================
    # Device and deterministic initialization
    # =========================================================
    device = select_device(
        args.device
    )

    print()
    print(
        f"Device: {device}"
    )

    seed_everything(
        model_seed
    )

    model = CARSBaselineCNN(
        in_channels=2,
        hidden_channels=64,
        num_blocks=6,
        kernel_size=7,
    )

    training_config = (
        build_training_config()
    )

    protocol_hashes = (
        copy_protocol_files(
            run_root
        )
    )

    # =========================================================
    # Training + SOURCE validation only
    # =========================================================
    print()
    print(
        "Training baseline..."
    )

    print(
        "Only A_typical validation controls "
        "early stopping."
    )

    training_start = (
        time.perf_counter()
    )

    training_result = (
        train_with_early_stopping(
            model,
            train_dataset,
            validation_dataset,
            normalization,
            model_seed=model_seed,
            num_train_samples=train_size,
            config=training_config,
            device=device,
            checkpoint_path=(
                run_root
                / "best_checkpoint.pt"
            ),
        )
    )

    training_seconds = (
        time.perf_counter()
        - training_start
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

    print()
    print(
        f"Training wall time: "
        f"{training_seconds / 60.0:.2f} min"
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

    # =========================================================
    # Save model/training artifacts.
    # No ID/OOD evaluation occurs here.
    # =========================================================
    save_training_artifacts(
        run_root,
        model=model,
        normalization_stats=normalization,
        training_config=training_config,
        training_result=training_result,
        model_seed=model_seed,
    )

    selection_result = {
        "track": (
            "acquisition"
        ),

        "source_domain": (
            SOURCE_DOMAIN
        ),

        "train_size": (
            train_size
        ),

        "model_seed": (
            model_seed
        ),

        "dataset_seed": (
            DATASET_SEED
        ),

        "best_epoch": int(
            training_result.best_epoch
        ),

        "best_validation_mse": float(
            training_result.best_validation_mse
        ),

        "epochs_completed": int(
            training_result.epochs_completed
        ),

        "stopped_early": bool(
            training_result.stopped_early
        ),

        "normalization_seconds": float(
            normalization_seconds
        ),

        "training_seconds": float(
            training_seconds
        ),

        "device": str(
            device
        ),

        "test_data_used": False,

        "ood_data_used": False,
    }

    selection_path = (
        run_root
        / "selection_result.json"
    )

    with selection_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            selection_result,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    run_manifest = {
        "purpose": (
            "training_size_selection"
        ),

        "track": (
            "acquisition"
        ),

        "source_domain": (
            SOURCE_DOMAIN
        ),

        "train_pool": {
            "start_index": 0,
            "num_samples": (
                TRAIN_POOL_SIZE
            ),
        },

        "selected_training_prefix": {
            "start_index": 0,
            "num_samples": (
                train_size
            ),
        },

        "validation": {
            "start_index": (
                VALIDATION_START
            ),
            "num_samples": (
                VALIDATION_SIZE
            ),
        },

        "model_seed": (
            model_seed
        ),

        "dataset_seed": (
            DATASET_SEED
        ),

        "device": str(
            device
        ),

        "id_test_used": False,

        "ood_used": False,

        "git_commit": (
            get_git_commit()
        ),

        "protocol_sha256": (
            protocol_hashes
        ),
    }

    manifest_path = (
        run_root
        / "selection_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            run_manifest,
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
        "SELECTION RUN COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        f"train_size:          {train_size}"
    )

    print(
        f"model_seed:          {model_seed}"
    )

    print(
        f"best_epoch:          "
        f"{training_result.best_epoch}"
    )

    print(
        f"best_validation_mse: "
        f"{training_result.best_validation_mse:.8f}"
    )

    print(
        f"epochs_completed:    "
        f"{training_result.epochs_completed}"
    )

    print(
        f"training_minutes:    "
        f"{training_seconds / 60.0:.2f}"
    )

    print()
    print(
        "ID test used: NO"
    )

    print(
        "OOD data used: NO"
    )

    print()
    print(
        f"Artifacts: {run_root}"
    )


if __name__ == "__main__":
    main()