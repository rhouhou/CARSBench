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

GLYCOGEN_484_CM1 = 484.0

WINDOW_HALF_WIDTH_CM1 = 20.0


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }

    return {
        "mean": float(
            np.mean(finite)
        ),
        "median": float(
            np.median(finite)
        ),
        "std": float(
            np.std(finite)
        ),
        "p05": float(
            np.percentile(
                finite,
                5.0,
            )
        ),
        "p95": float(
            np.percentile(
                finite,
                95.0,
            )
        ),
    }


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if x.size < 2:
        return float("nan")

    if np.std(x) <= 1.0e-12:
        return float("nan")

    if np.std(y) <= 1.0e-12:
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


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
        first_item["axis"],
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
                        item["input"],
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
                        item["target"],
                        dtype=np.float32,
                    ).reshape(-1)
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

            normalized_spectrum = (
                spectrum
                - normalization.spectrum_mean
            ) / normalization.spectrum_std

            normalized_spectrum[
                mask < 0.5
            ] = 0.0

            normalized_input = np.stack(
                [
                    normalized_spectrum,
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
                    len(batch_items),
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
            - GLYCOGEN_484_CM1
        )
        <= WINDOW_HALF_WIDTH_CM1
    )

    outside = (
        ~glycogen_all
    )

    return {
        "full_spectrum": full,
        "all_glycogen_bands": glycogen_all,
        "glycogen_484_band": glycogen_484,
        "outside_glycogen_bands": outside,
    }


def absolute_region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    prediction_region = prediction[
        :,
        mask,
    ]

    target_region = target[
        :,
        mask,
    ]

    error = (
        prediction_region
        - target_region
    )

    sample_rmse = np.sqrt(
        np.mean(
            error**2,
            axis=1,
        )
    )

    sample_mae = np.mean(
        np.abs(
            error
        ),
        axis=1,
    )

    pooled_rmse = float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )

    pooled_mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    target_energy = float(
        np.sqrt(
            np.mean(
                target_region**2
            )
        )
    )

    if target_energy > 1.0e-12:
        pooled_nrmse = (
            pooled_rmse
            / target_energy
        )
    else:
        pooled_nrmse = float(
            "nan"
        )

    return {
        "num_coordinates": int(
            np.sum(mask)
        ),

        "sample_rmse": summarize(
            sample_rmse
        ),

        "sample_mae": summarize(
            sample_mae
        ),

        "pooled_rmse": pooled_rmse,

        "pooled_mae": pooled_mae,

        "pooled_target_rms": (
            target_energy
        ),

        "pooled_nrmse": float(
            pooled_nrmse
        ),
    }


def band_recovery_metrics(
    axis: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """
    Quantify recovery of a spectral band using quantities that remain
    interpretable when the source domain contains almost no signal.

    Peak metric:
        maximum target/predicted Raman intensity inside the band.

    Area metric:
        signed integrated Raman intensity inside the band.

    Relative errors are computed using DATASET-LEVEL mean target
    magnitude, not by dividing each sample by its own potentially
    near-zero target.
    """
    axis_region = axis[
        mask
    ]

    prediction_region = prediction[
        :,
        mask,
    ]

    target_region = target[
        :,
        mask,
    ]

    target_peak = np.max(
        target_region,
        axis=1,
    )

    prediction_peak = np.max(
        prediction_region,
        axis=1,
    )

    peak_error = (
        prediction_peak
        - target_peak
    )

    peak_abs_error = np.abs(
        peak_error
    )

    target_area = np.trapezoid(
        target_region,
        x=axis_region,
        axis=1,
    )

    prediction_area = np.trapezoid(
        prediction_region,
        x=axis_region,
        axis=1,
    )

    area_error = (
        prediction_area
        - target_area
    )

    area_abs_error = np.abs(
        area_error
    )

    mean_target_peak = float(
        np.mean(
            np.abs(
                target_peak
            )
        )
    )

    mean_target_area = float(
        np.mean(
            np.abs(
                target_area
            )
        )
    )

    mean_peak_abs_error = float(
        np.mean(
            peak_abs_error
        )
    )

    mean_area_abs_error = float(
        np.mean(
            area_abs_error
        )
    )

    if mean_target_peak > 1.0e-12:
        dataset_relative_peak_mae = (
            mean_peak_abs_error
            / mean_target_peak
        )
    else:
        dataset_relative_peak_mae = (
            float("nan")
        )

    if mean_target_area > 1.0e-12:
        dataset_relative_area_mae = (
            mean_area_abs_error
            / mean_target_area
        )
    else:
        dataset_relative_area_mae = (
            float("nan")
        )

    denominator = float(
        np.sum(
            target_peak**2
        )
    )

    if denominator > 1.0e-12:
        peak_slope_through_origin = float(
            np.sum(
                target_peak
                * prediction_peak
            )
            / denominator
        )
    else:
        peak_slope_through_origin = (
            float("nan")
        )

    if np.std(
        target_peak
    ) > 1.0e-12:
        slope, intercept = np.polyfit(
            target_peak,
            prediction_peak,
            deg=1,
        )

        peak_linear_slope = float(
            slope
        )

        peak_linear_intercept = float(
            intercept
        )

    else:
        peak_linear_slope = float(
            "nan"
        )

        peak_linear_intercept = float(
            "nan"
        )

    return {
        "num_coordinates": int(
            np.sum(mask)
        ),

        "target_peak": summarize(
            target_peak
        ),

        "prediction_peak": summarize(
            prediction_peak
        ),

        "peak_error": summarize(
            peak_error
        ),

        "peak_absolute_error": summarize(
            peak_abs_error
        ),

        "dataset_relative_peak_mae": float(
            dataset_relative_peak_mae
        ),

        "peak_correlation": safe_correlation(
            target_peak,
            prediction_peak,
        ),

        "peak_slope_through_origin": (
            peak_slope_through_origin
        ),

        "peak_linear_slope": (
            peak_linear_slope
        ),

        "peak_linear_intercept": (
            peak_linear_intercept
        ),

        "target_area": summarize(
            target_area
        ),

        "prediction_area": summarize(
            prediction_area
        ),

        "area_error": summarize(
            area_error
        ),

        "area_absolute_error": summarize(
            area_abs_error
        ),

        "dataset_relative_area_mae": float(
            dataset_relative_area_mae
        ),

        "area_correlation": safe_correlation(
            target_area,
            prediction_area,
        ),
    }


def print_absolute_metrics(
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

    for region_name, metrics in (
        results[
            "absolute_region_metrics"
        ].items()
    ):
        print()
        print(
            region_name
        )

        print(
            f"  coordinates       "
            f"{metrics['num_coordinates']}"
        )

        print(
            f"  RMSE mean         "
            f"{metrics['sample_rmse']['mean']:.6f}"
        )

        print(
            f"  MAE mean          "
            f"{metrics['sample_mae']['mean']:.6f}"
        )

        print(
            f"  pooled RMSE       "
            f"{metrics['pooled_rmse']:.6f}"
        )

        print(
            f"  target RMS        "
            f"{metrics['pooled_target_rms']:.6f}"
        )

        print(
            f"  pooled NRMSE      "
            f"{metrics['pooled_nrmse']:.6f}"
        )


def print_484_recovery(
    domain: str,
    recovery: dict,
) -> None:
    print()
    print(
        f"{domain} — 484 cm^-1 recovery"
    )

    print(
        "-" * 80
    )

    print(
        f"mean target peak       "
        f"{recovery['target_peak']['mean']:.6f}"
    )

    print(
        f"mean predicted peak    "
        f"{recovery['prediction_peak']['mean']:.6f}"
    )

    print(
        f"mean peak bias         "
        f"{recovery['peak_error']['mean']:.6f}"
    )

    print(
        f"mean peak abs error    "
        f"{recovery['peak_absolute_error']['mean']:.6f}"
    )

    print(
        f"dataset rel peak MAE   "
        f"{recovery['dataset_relative_peak_mae']:.6f}"
    )

    print(
        f"peak correlation       "
        f"{recovery['peak_correlation']:.6f}"
    )

    print(
        f"peak slope origin      "
        f"{recovery['peak_slope_through_origin']:.6f}"
    )

    print(
        f"linear slope           "
        f"{recovery['peak_linear_slope']:.6f}"
    )

    print(
        f"linear intercept       "
        f"{recovery['peak_linear_intercept']:.6f}"
    )

    print(
        f"mean target area       "
        f"{recovery['target_area']['mean']:.6f}"
    )

    print(
        f"mean predicted area    "
        f"{recovery['prediction_area']['mean']:.6f}"
    )

    print(
        f"mean area abs error    "
        f"{recovery['area_absolute_error']['mean']:.6f}"
    )

    print(
        f"dataset rel area MAE   "
        f"{recovery['dataset_relative_area_mae']:.6f}"
    )

    print(
        f"area correlation       "
        f"{recovery['area_correlation']:.6f}"
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

    if not normalization_path.exists():
        raise FileNotFoundError(
            normalization_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
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

        masks = build_region_masks(
            axis
        )

        absolute_metrics = {}

        for (
            region_name,
            region_mask,
        ) in masks.items():
            absolute_metrics[
                region_name
            ] = absolute_region_metrics(
                prediction,
                target,
                region_mask,
            )

        recovery_484 = (
            band_recovery_metrics(
                axis,
                prediction,
                target,
                masks[
                    "glycogen_484_band"
                ],
            )
        )

        all_results[
            domain
        ] = {
            "absolute_region_metrics": (
                absolute_metrics
            ),

            "glycogen_484_recovery": (
                recovery_484
            ),
        }

        print_absolute_metrics(
            domain,
            all_results[
                domain
            ],
        )

        print()

        print_484_recovery(
            domain,
            recovery_484,
        )

    # =========================================================
    # Direct H versus I absolute-error comparison
    #
    # These ratios do NOT divide by the H target magnitude.
    # They directly quantify how much prediction error increases
    # when the held-out biochemical component appears.
    # =========================================================

    h_metrics = all_results[
        "H_biochemical_source"
    ][
        "absolute_region_metrics"
    ]

    i_metrics = all_results[
        "I_biochemical_target"
    ][
        "absolute_region_metrics"
    ]

    comparison = {}

    print()
    print(
        "=" * 80
    )

    print(
        "I / H ABSOLUTE ERROR RATIOS"
    )

    print(
        "=" * 80
    )

    for region_name in h_metrics:
        h_rmse = h_metrics[
            region_name
        ][
            "sample_rmse"
        ][
            "mean"
        ]

        i_rmse = i_metrics[
            region_name
        ][
            "sample_rmse"
        ][
            "mean"
        ]

        h_mae = h_metrics[
            region_name
        ][
            "sample_mae"
        ][
            "mean"
        ]

        i_mae = i_metrics[
            region_name
        ][
            "sample_mae"
        ][
            "mean"
        ]

        rmse_ratio = (
            i_rmse
            / h_rmse
        )

        mae_ratio = (
            i_mae
            / h_mae
        )

        comparison[
            region_name
        ] = {
            "rmse_ratio": float(
                rmse_ratio
            ),

            "mae_ratio": float(
                mae_ratio
            ),
        }

        print()
        print(
            region_name
        )

        print(
            f"  RMSE ratio: "
            f"{rmse_ratio:.4f}"
        )

        print(
            f"  MAE ratio:  "
            f"{mae_ratio:.4f}"
        )

    all_results[
        "I_over_H_absolute_error_ratios"
    ] = comparison

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_ROOT
        / "heldout_component_recovery.json"
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