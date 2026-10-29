
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

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

SPEC_PATH = (
    ROOT
    / "specs"
    / "carsbench_v1.json"
)

SCRIPT08_PATH = (
    ROOT
    / "scripts"
    / "08_run_acquisition_scale_study.py"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "fingerprint_source"
)

TRAIN_SIZE = 100_000
VALIDATION_SIZE = 5_000

MODEL_SEEDS = [
    101,
    202,
    303,
]


# ============================================================
# Load the already-established acquisition data protocol
# ============================================================

def load_acquisition_protocol():
    if not SCRIPT08_PATH.exists():
        raise FileNotFoundError(
            f"Missing source protocol script: {SCRIPT08_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "carsbench_acquisition_scale",
        SCRIPT08_PATH,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not import {SCRIPT08_PATH}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Fingerprint view
# ============================================================

class FingerprintDataset(ChunkedNPZDataset):
    """
    Deterministic 400–1800 cm^-1 view of an existing
    CARSBench ChunkedNPZDataset.

    The original generated source data are not modified.
    """

    def __init__(
        self,
        path: str | Path,
        fingerprint_mask: np.ndarray,
    ):
        super().__init__(path)

        self.fingerprint_mask = np.asarray(
            fingerprint_mask,
            dtype=bool,
        )

        if self.fingerprint_mask.ndim != 1:
            raise ValueError(
                "fingerprint_mask must be one-dimensional."
            )

        if int(
            self.fingerprint_mask.sum()
        ) != 512:
            raise ValueError(
                "Expected 512 fingerprint points, "
                f"found "
                f"{int(self.fingerprint_mask.sum())}."
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        item = super().__getitem__(
            index
        )

        fp = self.fingerprint_mask

        raw_spectrum = np.asarray(
            item["spectrum"],
            dtype=np.float32,
        )

        raw_mask = np.asarray(
            item["observation_mask"],
            dtype=np.float32,
        )

        raw_axis = np.asarray(
            item["axis"],
            dtype=np.float32,
        )

        raw_target = np.asarray(
            item["target"],
            dtype=np.float32,
        )

        if (
            raw_spectrum.shape[-1]
            != len(fp)
        ):
            raise ValueError(
                "Fingerprint mask length does not match "
                "source spectrum."
            )

        spectrum = raw_spectrum[
            fp
        ]

        observation_mask = raw_mask[
            fp
        ]

        axis = raw_axis[
            fp
        ]

        if raw_target.ndim == 2:
            target = raw_target[
                :,
                fp,
            ]

        elif raw_target.ndim == 1:
            target = raw_target[
                fp
            ][
                np.newaxis,
                :
            ]

        else:
            raise ValueError(
                "Unexpected target shape: "
                f"{raw_target.shape}"
            )

        model_input = np.stack(
            [
                spectrum,
                observation_mask,
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        result = dict(
            item
        )

        result[
            "spectrum"
        ] = spectrum

        result[
            "observation_mask"
        ] = observation_mask

        result[
            "target"
        ] = target

        result[
            "axis"
        ] = axis

        result[
            "input"
        ] = model_input

        return result


# ============================================================
# Fingerprint axis
# ============================================================

def load_fingerprint_mask():
    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        spec = json.load(
            handle
        )

    canonical = (
        spec["canonical_axis"]
    )

    axis = np.linspace(
        float(
            canonical[
                "nu_min_cm1"
            ]
        ),
        float(
            canonical[
                "nu_max_cm1"
            ]
        ),
        int(
            canonical[
                "num_points"
            ]
        ),
        dtype=np.float64,
    )

    window = (
        spec[
            "observation_windows"
        ][
            "fingerprint"
        ]
    )

    lower = float(
        window[
            "nu_min_cm1"
        ]
    )

    upper = float(
        window[
            "nu_max_cm1"
        ]
    )

    fingerprint_mask = (
        (axis >= lower)
        & (axis <= upper)
    )

    fingerprint_axis = (
        axis[
            fingerprint_mask
        ]
    )

    if (
        len(fingerprint_axis)
        != 512
    ):
        raise RuntimeError(
            "Expected 512 fingerprint points, "
            f"found {len(fingerprint_axis)}."
        )

    return (
        fingerprint_mask,
        fingerprint_axis,
    )


# ============================================================
# Training configuration
# ============================================================

def build_training_config():
    return TrainingConfig(
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=64,
        maximum_epochs=100,
        gradient_clip_norm=1.0,
        early_stopping_patience=10,
        early_stopping_min_delta=1.0e-5,
    )


# ============================================================
# One model seed
# ============================================================

def run_seed(
    *,
    seed: int,
    train_dataset,
    validation_dataset,
    device,
    fingerprint_axis,
):

    run_root = (
        OUTPUT_ROOT
        / (
            "A_fingerprint_"
            f"train100000_seed{seed}"
        )
    )

    model_path = (
        run_root
        / "model.pt"
    )

    normalization_path = (
        run_root
        / "normalization.json"
    )

    training_path = (
        run_root
        / "training.json"
    )

    protocol_path = (
        run_root
        / "cross_simulator_source_protocol.json"
    )

    # --------------------------------------------------------
    # Reuse completed run
    # --------------------------------------------------------

    if (
        model_path.exists()
        and normalization_path.exists()
        and training_path.exists()
        and protocol_path.exists()
    ):
        print()
        print(
            f"Seed {seed}: completed "
            "fingerprint model already exists."
        )

        print(
            "No training repeated."
        )

        return

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print(
        "FINGERPRINT SOURCE MODEL "
        f"— SEED {seed}"
    )
    print("=" * 72)

    print(
        "Training samples:  ",
        TRAIN_SIZE,
    )

    print(
        "Validation samples:",
        VALIDATION_SIZE,
    )

    print(
        "Spectral points:   ",
        len(fingerprint_axis),
    )

    print(
        "Spectral range:    ",
        f"{fingerprint_axis[0]:.6f}",
        "to",
        f"{fingerprint_axis[-1]:.6f}",
        "cm^-1",
    )

    # --------------------------------------------------------
    # Normalization
    # CARSBench source training only
    # --------------------------------------------------------

    print()
    print(
        "Fitting source-only fingerprint normalization..."
    )

    normalization_start = (
        time.perf_counter()
    )

    normalization = (
        fit_normalization(
            train_dataset,
            num_samples=TRAIN_SIZE,
        )
    )

    normalization_seconds = (
        time.perf_counter()
        - normalization_start
    )

    print()
    print(
        json.dumps(
            normalization.to_dict(),
            indent=2,
        )
    )

    print()
    print(
        "Normalization time:",
        f"{normalization_seconds:.1f} s",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    seed_everything(
        seed
    )

    model = (
        CARSBaselineCNN(
            in_channels=2,
            hidden_channels=64,
            num_blocks=6,
            kernel_size=7,
        )
    )

    config = (
        build_training_config()
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "Training fingerprint model..."
    )

    print(
        "CRIMSON samples used for training: 0"
    )

    print(
        "CRIMSON samples used for validation: 0"
    )

    print(
        "CRIMSON samples used for normalization: 0"
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
            model_seed=seed,
            num_train_samples=TRAIN_SIZE,
            config=config,
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
        "Training time:",
        f"{training_seconds / 60.0:.2f} min",
    )

    # --------------------------------------------------------
    # Save normal CARSBench artifacts
    # --------------------------------------------------------

    save_training_artifacts(
        run_root,
        model=model,
        normalization_stats=normalization,
        training_config=config,
        training_result=training_result,
        model_seed=seed,
    )

    protocol = {
        "protocol": (
            "CARSBench_cross_simulator_"
            "fingerprint_source_v1"
        ),

        "source_domain": (
            "A_typical"
        ),

        "source_training_samples": (
            TRAIN_SIZE
        ),

        "source_validation_samples": (
            VALIDATION_SIZE
        ),

        "model_seed": int(
            seed
        ),

        "spectral_window": (
            "fingerprint"
        ),

        "spectral_min_cm1": float(
            fingerprint_axis[0]
        ),

        "spectral_max_cm1": float(
            fingerprint_axis[-1]
        ),

        "spectral_points": int(
            len(
                fingerprint_axis
            )
        ),

        "crimson_training_samples": 0,

        "crimson_validation_samples": 0,

        "crimson_normalization_samples": 0,

        "crimson_used_for_model_selection": False,

        "notes": [
            (
                "Training data are the same deterministic "
                "A_typical 0-99999 source samples used by "
                "the acquisition scale study."
            ),
            (
                "Validation data are the same deterministic "
                "A_typical 100000-104999 source samples."
            ),
            (
                "Only the predefined 400-1800 cm^-1 "
                "fingerprint region is retained."
            ),
            (
                "CRIMSON is reserved exclusively for "
                "zero-shot external evaluation."
            ),
        ],
    }

    with protocol_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            protocol,
            handle,
            indent=2,
            sort_keys=True,
        )

    print()
    print(
        "Saved:",
        run_root,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

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

    parser.add_argument(
        "--seed",
        type=int,
        choices=MODEL_SEEDS,
        default=None,
        help=(
            "Run one model seed. "
            "If omitted, run all three."
        ),
    )

    args = parser.parse_args()

    print("=" * 72)
    print(
        "CARSBENCH MATCHED "
        "FINGERPRINT SOURCE TRAINING"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Use script 08 as the authoritative source-data protocol
    # --------------------------------------------------------

    acquisition_protocol = (
        load_acquisition_protocol()
    )

    print()
    print(
        "Using acquisition data protocol:"
    )
    print(
        SCRIPT08_PATH
    )

    # This will reuse the data if already available.
    # It regenerates only the deterministic source data
    # if they are missing.
    train_path = (
        acquisition_protocol.ensure_train_pool(
            force=False,
        )
    )

    validation_path = (
        acquisition_protocol.ensure_validation(
            force=False,
        )
    )

    print()
    print(
        "Source training path:"
    )
    print(
        train_path
    )

    print()
    print(
        "Source validation path:"
    )
    print(
        validation_path
    )

    # --------------------------------------------------------
    # Check exact sample ranges
    # --------------------------------------------------------

    acquisition_protocol.verify_dataset_ranges(
        train_path=train_path,
        validation_path=validation_path,
        train_size=TRAIN_SIZE,
    )

    # --------------------------------------------------------
    # Fingerprint grid
    # --------------------------------------------------------

    (
        fingerprint_mask,
        fingerprint_axis,
    ) = load_fingerprint_mask()

    print()
    print(
        "Fingerprint points:",
        len(fingerprint_axis),
    )

    print(
        "Fingerprint range:",
        f"{fingerprint_axis[0]:.6f}",
        "to",
        f"{fingerprint_axis[-1]:.6f}",
        "cm^-1",
    )

    # --------------------------------------------------------
    # Create deterministic views
    # --------------------------------------------------------

    train_dataset = (
        FingerprintDataset(
            train_path,
            fingerprint_mask,
        )
    )

    validation_dataset = (
        FingerprintDataset(
            validation_path,
            fingerprint_mask,
        )
    )

    if (
        len(train_dataset)
        != TRAIN_SIZE
    ):
        raise RuntimeError(
            "Incorrect source training size: "
            f"{len(train_dataset)}"
        )

    if (
        len(validation_dataset)
        != VALIDATION_SIZE
    ):
        raise RuntimeError(
            "Incorrect validation size: "
            f"{len(validation_dataset)}"
        )

    # Representative structural checks.
    train_first = (
        train_dataset[0]
    )

    train_last = (
        train_dataset[
            TRAIN_SIZE - 1
        ]
    )

    validation_first = (
        validation_dataset[0]
    )

    checks = [
        (
            "train first",
            train_first,
        ),
        (
            "train last",
            train_last,
        ),
        (
            "validation first",
            validation_first,
        ),
    ]

    for name, item in checks:

        if (
            item[
                "spectrum"
            ].shape
            != (512,)
        ):
            raise RuntimeError(
                f"{name}: invalid "
                "spectrum shape "
                f"{item['spectrum'].shape}"
            )

        if (
            item[
                "observation_mask"
            ].shape
            != (512,)
        ):
            raise RuntimeError(
                f"{name}: invalid "
                "mask shape."
            )

        if (
            item[
                "target"
            ].shape
            != (1, 512)
        ):
            raise RuntimeError(
                f"{name}: invalid "
                "target shape "
                f"{item['target'].shape}"
            )

        if (
            item[
                "axis"
            ].shape
            != (512,)
        ):
            raise RuntimeError(
                f"{name}: invalid "
                "axis shape."
            )

    print()
    print(
        "Dataset structural checks: PASS"
    )

    print(
        "Training sample indices:",
        int(
            train_first[
                "sample_index"
            ]
        ),
        "to",
        int(
            train_last[
                "sample_index"
            ]
        ),
    )

    print(
        "Validation begins at:",
        int(
            validation_first[
                "sample_index"
            ]
        ),
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        select_device(
            args.device
        )
    )

    print()
    print(
        "Device:",
        device,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Seeds
    # --------------------------------------------------------

    seeds = (
        [args.seed]
        if args.seed is not None
        else MODEL_SEEDS
    )

    for seed in seeds:
        run_seed(
            seed=seed,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            device=device,
            fingerprint_axis=fingerprint_axis,
        )

    print()
    print("=" * 72)
    print(
        "FINGERPRINT SOURCE "
        "TRAINING COMPLETE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
