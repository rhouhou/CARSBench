from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from CARSBench.ml import (
    CARSBaselineCNN,
    ChunkedNPZDataset,
    NormalizationStats,
    select_device,
)


ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    ROOT
    / "results"
    / "chemistry_smoke"
    / "H_train256_seed101"
)

DATA_ROOT = (
    ROOT
    / "data"
    / "carsbench_chemistry_diagnostic"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "chemistry_diagnostic"
)

DOMAINS = [
    "H_biochemical_source",
    "I_biochemical_target",
]

GLYCOGEN_PEAKS_CM1 = [
    484.0,
    856.0,
    940.0,
    1050.0,
    1084.0,
    1130.0,
]

WINDOW_HALF_WIDTH_CM1 = 20.0


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p05": float(np.percentile(values, 5.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def spectral_angle_rows(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    numerator = np.sum(
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

    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(
            numerator,
            dtype=np.float64,
        ),
        where=denominator > 1.0e-12,
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return np.arccos(
        cosine
    )


def calculate_region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
) -> dict:
    prediction_region = prediction[
        :,
        region_mask,
    ]

    target_region = target[
        :,
        region_mask,
    ]

    error = (
        prediction_region
        - target_region
    )

    rmse = np.sqrt(
        np.mean(
            error**2,
            axis=1,
        )
    )

    mae = np.mean(
        np.abs(error),
        axis=1,
    )

    angle = spectral_angle_rows(
        prediction_region,
        target_region,
    )

    target_rms = np.sqrt(
        np.mean(
            target_region**2,
            axis=1,
        )
    )

    target_mean_abs = np.mean(
        np.abs(
            target_region
        ),
        axis=1,
    )

    relative_rmse = np.divide(
        rmse,
        target_rms,
        out=np.full_like(
            rmse,
            np.nan,
            dtype=np.float64,
        ),
        where=target_rms > 1.0e-12,
    )

    relative_mae = np.divide(
        mae,
        target_mean_abs,
        out=np.full_like(
            mae,
            np.nan,
            dtype=np.float64,
        ),
        where=target_mean_abs > 1.0e-12,
    )

    return {
        "num_coordinates": int(
            np.sum(
                region_mask
            )
        ),

        "rmse": summarize(
            rmse
        ),

        "mae": summarize(
            mae
        ),

        "spectral_angle": summarize(
            angle
        ),

        "target_rms": summarize(
            target_rms
        ),

        "target_mean_abs": summarize(
            target_mean_abs
        ),

        "relative_rmse": summarize(
            relative_rmse[
                np.isfinite(
                    relative_rmse
                )
            ]
        ),

        "relative_mae": summarize(
            relative_mae[
                np.isfinite(
                    relative_mae
                )
            ]
        ),
    }


def predict_dataset(
    model: torch.nn.Module,
    dataset: ChunkedNPZDataset,
    normalization: NormalizationStats,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    predictions = []
    targets = []

    first_item = dataset[0]

    axis = np.asarray(
        first_item[
            "axis"
        ],
        dtype=np.float64,
    )

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(dataset),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(dataset),
            )

            batch_items = [
                dataset[index]
                for index in range(
                    start,
                    stop,
                )
            ]

            input_array = np.stack(
                [
                    np.asarray(
                        item[
                            "input"
                        ],
                        dtype=np.float32,
                    )
                    for item
                    in batch_items
                ],
                axis=0,
            )

            target_array = np.stack(
                [
                    np.asarray(
                        item[
                            "target"
                        ],
                        dtype=np.float32,
                    ).reshape(
                        -1
                    )
                    for item
                    in batch_items
                ],
                axis=0,
            )

            spectrum = input_array[
                :,
                0,
                :,
            ]

            mask = input_array[
                :,
                1,
                :,
            ]

            spectrum = (
                spectrum
                - normalization.spectrum_mean
            ) / normalization.spectrum_std

            spectrum[
                mask < 0.5
            ] = 0.0

            normalized_input = np.stack(
                [
                    spectrum,
                    mask,
                ],
                axis=1,
            )

            tensor_input = torch.from_numpy(
                normalized_input.astype(
                    np.float32
                )
            ).to(
                device
            )

            prediction_normalized = model(
                tensor_input
            )

            prediction_normalized = (
                prediction_normalized
                .detach()
                .cpu()
                .numpy()
                .reshape(
                    len(
                        batch_items
                    ),
                    -1,
                )
            )

            prediction = (
                prediction_normalized
                * normalization.target_std
                + normalization.target_mean
            )

            predictions.append(
                prediction
            )

            targets.append(
                target_array
            )

    return (
        axis,
        np.concatenate(
            predictions,
            axis=0,
        ),
        np.concatenate(
            targets,
            axis=0,
        ),
    )


def build_region_masks(
    axis: np.ndarray,
) -> dict[str, np.ndarray]:

    full = np.ones(
        axis.shape,
        dtype=bool,
    )

    glycogen_all = np.zeros(
        axis.shape,
        dtype=bool,
    )

    for center in GLYCOGEN_PEAKS_CM1:
        glycogen_all |= (
            np.abs(
                axis
                - center
            )
            <= WINDOW_HALF_WIDTH_CM1
        )

    glycogen_484 = (
        np.abs(
            axis
            - 484.0
        )
        <= WINDOW_HALF_WIDTH_CM1
    )

    non_glycogen = (
        ~glycogen_all
    )

    return {
        "full_spectrum": full,
        "all_glycogen_bands": glycogen_all,
        "glycogen_484_band": glycogen_484,
        "outside_glycogen_bands": non_glycogen,
    }


def print_domain(
    domain: str,
    results: dict,
) -> None:

    print()
    print(
        "=" * 80
    )

    print(
        domain
    )

    print(
        "=" * 80
    )

    for (
        region_name,
        metrics,
    ) in results.items():

        print()
        print(
            region_name
        )

        print(
            f"  coordinates     "
            f"{metrics['num_coordinates']}"
        )

        print(
            f"  RMSE            "
            f"{metrics['rmse']['mean']:.6f}"
        )

        print(
            f"  relative RMSE   "
            f"{metrics['relative_rmse']['mean']:.6f}"
        )

        print(
            f"  MAE             "
            f"{metrics['mae']['mean']:.6f}"
        )

        print(
            f"  relative MAE    "
            f"{metrics['relative_mae']['mean']:.6f}"
        )

        print(
            f"  spectral angle  "
            f"{metrics['spectral_angle']['mean']:.6f}"
        )


def main() -> None:

    normalization_path = (
        MODEL_ROOT
        / "normalization.json"
    )

    model_path = (
        MODEL_ROOT
        / "model.pt"
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

    device = select_device(
        "auto"
    )

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
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(
        device
    )

    model.eval()

    all_results = {}

    axis_reference = None

    for domain in DOMAINS:

        dataset = ChunkedNPZDataset(
            DATA_ROOT
            / domain
        )

        (
            axis,
            prediction,
            target,
        ) = predict_dataset(
            model,
            dataset,
            normalization,
            device,
        )

        if axis_reference is None:
            axis_reference = axis

        else:
            if not np.array_equal(
                axis,
                axis_reference,
            ):
                raise RuntimeError(
                    "Domain axes are not identical."
                )

        region_masks = build_region_masks(
            axis
        )

        domain_results = {}

        for (
            region_name,
            region_mask,
        ) in region_masks.items():

            domain_results[
                region_name
            ] = calculate_region_metrics(
                prediction,
                target,
                region_mask,
            )

        all_results[
            domain
        ] = domain_results

        print_domain(
            domain,
            domain_results,
        )

    # =========================================================
    # H versus I ratios
    # =========================================================

    h = all_results[
        "H_biochemical_source"
    ]

    i = all_results[
        "I_biochemical_target"
    ]

    comparison = {}

    print()
    print(
        "=" * 80
    )

    print(
        "I / H ERROR RATIOS"
    )

    print(
        "=" * 80
    )

    for region_name in h:

        comparison[
            region_name
        ] = {
            "relative_rmse_ratio": (
                i[
                    region_name
                ][
                    "relative_rmse"
                ][
                    "mean"
                ]
                /
                h[
                    region_name
                ][
                    "relative_rmse"
                ][
                    "mean"
                ]
            ),

            "spectral_angle_ratio": (
                i[
                    region_name
                ][
                    "spectral_angle"
                ][
                    "mean"
                ]
                /
                h[
                    region_name
                ][
                    "spectral_angle"
                ][
                    "mean"
                ]
            ),
        }

        print()
        print(
            region_name
        )

        print(
            "  relative RMSE ratio: "
            f"{comparison[region_name]['relative_rmse_ratio']:.4f}"
        )

        print(
            "  spectral angle ratio: "
            f"{comparison[region_name]['spectral_angle_ratio']:.4f}"
        )

    all_results[
        "I_over_H_comparison"
    ] = comparison

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "glycogen_band_diagnostic.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            all_results,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()