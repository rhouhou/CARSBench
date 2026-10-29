
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

EVAL_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "evaluation"
)

EVAL_JSON = (
    EVAL_ROOT
    / "cross_simulator_crimson_evaluation.json"
)

OUTPUT_JSON = (
    EVAL_ROOT
    / "cross_simulator_scale_diagnostic.json"
)

SEEDS = [
    101,
    202,
    303,
]


def sample_rmse(
    prediction: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:

    return np.sqrt(
        np.mean(
            (
                prediction
                - target
            ) ** 2,
            axis=1,
        )
    )


def summarize(
    values: np.ndarray,
) -> dict:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(values)
    ]

    return {
        "mean": float(
            np.mean(finite)
        ),
        "std": float(
            np.std(finite)
        ),
        "median": float(
            np.median(finite)
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


def global_scale_alignment(
    prediction,
    target,
):

    p = prediction.ravel().astype(
        np.float64
    )

    t = target.ravel().astype(
        np.float64
    )

    denominator = float(
        np.dot(p, p)
    )

    if denominator <= 0:
        raise RuntimeError(
            "Invalid prediction norm."
        )

    alpha = float(
        np.dot(p, t)
        / denominator
    )

    aligned = (
        alpha
        * prediction
    )

    return (
        alpha,
        aligned,
    )


def global_affine_alignment(
    prediction,
    target,
):

    p = prediction.ravel().astype(
        np.float64
    )

    t = target.ravel().astype(
        np.float64
    )

    p_mean = float(
        np.mean(p)
    )

    t_mean = float(
        np.mean(t)
    )

    p_centered = (
        p
        - p_mean
    )

    t_centered = (
        t
        - t_mean
    )

    denominator = float(
        np.dot(
            p_centered,
            p_centered,
        )
    )

    alpha = float(
        np.dot(
            p_centered,
            t_centered,
        )
        / denominator
    )

    beta = float(
        t_mean
        - alpha * p_mean
    )

    aligned = (
        alpha
        * prediction
        + beta
    )

    return (
        alpha,
        beta,
        aligned,
    )


def sample_scale_alignment(
    prediction,
    target,
):

    numerator = np.sum(
        prediction
        * target,
        axis=1,
    )

    denominator = np.sum(
        prediction ** 2,
        axis=1,
    )

    alpha = np.full(
        len(prediction),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        denominator
        > 1.0e-15
    )

    alpha[
        valid
    ] = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    aligned = (
        prediction
        * alpha[:, None]
    )

    return (
        alpha,
        aligned,
    )


def sample_affine_alignment(
    prediction,
    target,
):

    p_mean = np.mean(
        prediction,
        axis=1,
        keepdims=True,
    )

    t_mean = np.mean(
        target,
        axis=1,
        keepdims=True,
    )

    pc = (
        prediction
        - p_mean
    )

    tc = (
        target
        - t_mean
    )

    numerator = np.sum(
        pc * tc,
        axis=1,
    )

    denominator = np.sum(
        pc ** 2,
        axis=1,
    )

    alpha = np.full(
        len(prediction),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        denominator
        > 1.0e-15
    )

    alpha[
        valid
    ] = (
        numerator[
            valid
        ]
        / denominator[
            valid
        ]
    )

    beta = (
        t_mean[:, 0]
        - alpha
        * p_mean[:, 0]
    )

    aligned = (
        prediction
        * alpha[:, None]
        + beta[:, None]
    )

    return (
        alpha,
        beta,
        aligned,
    )


def diagnose_seed(
    seed: int,
):

    path = (
        EVAL_ROOT
        / f"seed{seed}_crimson.npz"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    data = np.load(
        path
    )

    prediction = np.asarray(
        data["prediction"],
        dtype=np.float64,
    )

    target = np.asarray(
        data["target"],
        dtype=np.float64,
    )

    native_rmse = (
        sample_rmse(
            prediction,
            target,
        )
    )

    # ========================================================
    # One global multiplicative factor
    # ========================================================

    (
        global_alpha,
        global_scale_prediction,
    ) = global_scale_alignment(
        prediction,
        target,
    )

    global_scale_rmse = (
        sample_rmse(
            global_scale_prediction,
            target,
        )
    )

    # ========================================================
    # One global affine transform
    # ========================================================

    (
        global_affine_alpha,
        global_affine_beta,
        global_affine_prediction,
    ) = global_affine_alignment(
        prediction,
        target,
    )

    global_affine_rmse = (
        sample_rmse(
            global_affine_prediction,
            target,
        )
    )

    # ========================================================
    # Per-spectrum oracle multiplicative alignment
    # ========================================================

    (
        sample_alpha,
        sample_scale_prediction,
    ) = sample_scale_alignment(
        prediction,
        target,
    )

    sample_scale_rmse = (
        sample_rmse(
            sample_scale_prediction,
            target,
        )
    )

    # ========================================================
    # Per-spectrum oracle affine alignment
    # ========================================================

    (
        sample_affine_alpha,
        sample_affine_beta,
        sample_affine_prediction,
    ) = sample_affine_alignment(
        prediction,
        target,
    )

    sample_affine_rmse = (
        sample_rmse(
            sample_affine_prediction,
            target,
        )
    )

    prediction_rms = np.sqrt(
        np.mean(
            prediction ** 2,
            axis=1,
        )
    )

    target_rms = np.sqrt(
        np.mean(
            target ** 2,
            axis=1,
        )
    )

    rms_ratio = (
        prediction_rms
        / target_rms
    )

    return {
        "native_rmse": (
            summarize(
                native_rmse
            )
        ),

        "global_scale_alignment": {
            "alpha_prediction_to_target": (
                global_alpha
            ),

            "rmse": (
                summarize(
                    global_scale_rmse
                )
            ),

            "fraction_rmse_remaining": float(
                np.mean(
                    global_scale_rmse
                )
                / np.mean(
                    native_rmse
                )
            ),
        },

        "global_affine_alignment": {
            "alpha_prediction_to_target": (
                global_affine_alpha
            ),

            "beta_prediction_to_target": (
                global_affine_beta
            ),

            "rmse": (
                summarize(
                    global_affine_rmse
                )
            ),

            "fraction_rmse_remaining": float(
                np.mean(
                    global_affine_rmse
                )
                / np.mean(
                    native_rmse
                )
            ),
        },

        "per_spectrum_scale_oracle": {
            "alpha": (
                summarize(
                    sample_alpha
                )
            ),

            "negative_alpha_fraction": float(
                np.mean(
                    sample_alpha < 0
                )
            ),

            "rmse": (
                summarize(
                    sample_scale_rmse
                )
            ),

            "fraction_rmse_remaining": float(
                np.mean(
                    sample_scale_rmse
                )
                / np.mean(
                    native_rmse
                )
            ),
        },

        "per_spectrum_affine_oracle": {
            "alpha": (
                summarize(
                    sample_affine_alpha
                )
            ),

            "beta": (
                summarize(
                    sample_affine_beta
                )
            ),

            "rmse": (
                summarize(
                    sample_affine_rmse
                )
            ),

            "fraction_rmse_remaining": float(
                np.mean(
                    sample_affine_rmse
                )
                / np.mean(
                    native_rmse
                )
            ),
        },

        "prediction_to_target_rms_ratio": (
            summarize(
                rms_ratio
            )
        ),
    }


def main():

    if not EVAL_JSON.exists():
        raise FileNotFoundError(
            EVAL_JSON
        )

    with EVAL_JSON.open(
        "r",
        encoding="utf-8",
    ) as handle:
        evaluation = json.load(
            handle
        )

    print("=" * 76)
    print(
        "CROSS-SIMULATOR SCALE / SHAPE DIAGNOSTIC"
    )
    print("=" * 76)

    print()
    print(
        "IMPORTANT: all target-fitted alignments below "
        "are diagnostic oracles only."
    )
    print(
        "They are NOT zero-shot benchmark results."
    )

    print()
    print(
        "SOURCE-NORMALIZED INPUT DIAGNOSTICS"
    )
    print("-" * 76)

    input_shift = (
        evaluation[
            "source_normalized_input_shift"
        ]
    )

    print()
    print("SOURCE ID")
    print(
        json.dumps(
            input_shift[
                "source_id"
            ],
            indent=2,
        )
    )

    print()
    print("CRIMSON")
    print(
        json.dumps(
            input_shift[
                "crimson"
            ],
            indent=2,
        )
    )

    result = {
        "warning": (
            "Target-fitted scale and affine alignments "
            "are diagnostic only and must not be reported "
            "as zero-shot benchmark performance."
        ),

        "source_normalized_input_shift": (
            input_shift
        ),

        "per_seed": {},
    }

    native_means = []
    global_scale_means = []
    global_affine_means = []
    sample_scale_means = []
    sample_affine_means = []

    for seed in SEEDS:

        diagnostic = (
            diagnose_seed(
                seed
            )
        )

        result[
            "per_seed"
        ][
            str(seed)
        ] = diagnostic

        native = (
            diagnostic[
                "native_rmse"
            ][
                "mean"
            ]
        )

        global_scale = (
            diagnostic[
                "global_scale_alignment"
            ][
                "rmse"
            ][
                "mean"
            ]
        )

        global_affine = (
            diagnostic[
                "global_affine_alignment"
            ][
                "rmse"
            ][
                "mean"
            ]
        )

        sample_scale = (
            diagnostic[
                "per_spectrum_scale_oracle"
            ][
                "rmse"
            ][
                "mean"
            ]
        )

        sample_affine = (
            diagnostic[
                "per_spectrum_affine_oracle"
            ][
                "rmse"
            ][
                "mean"
            ]
        )

        native_means.append(
            native
        )

        global_scale_means.append(
            global_scale
        )

        global_affine_means.append(
            global_affine
        )

        sample_scale_means.append(
            sample_scale
        )

        sample_affine_means.append(
            sample_affine
        )

        print()
        print(
            f"SEED {seed}"
        )
        print("-" * 76)

        print(
            "Native RMSE:",
            f"{native:.6f}",
        )

        print(
            "Global-scale alpha:",
            f"{diagnostic['global_scale_alignment']['alpha_prediction_to_target']:.6f}",
        )

        print(
            "Global-scale RMSE:",
            f"{global_scale:.6f}",
        )

        print(
            "Global-affine alpha:",
            f"{diagnostic['global_affine_alignment']['alpha_prediction_to_target']:.6f}",
        )

        print(
            "Global-affine beta:",
            f"{diagnostic['global_affine_alignment']['beta_prediction_to_target']:.6f}",
        )

        print(
            "Global-affine RMSE:",
            f"{global_affine:.6f}",
        )

        print(
            "Per-spectrum scale-oracle RMSE:",
            f"{sample_scale:.6f}",
        )

        print(
            "Per-spectrum affine-oracle RMSE:",
            f"{sample_affine:.6f}",
        )

    aggregate = {
        "native_rmse": {
            "mean_across_seeds": float(
                np.mean(
                    native_means
                )
            ),
        },

        "global_scale_aligned_rmse": {
            "mean_across_seeds": float(
                np.mean(
                    global_scale_means
                )
            ),
        },

        "global_affine_aligned_rmse": {
            "mean_across_seeds": float(
                np.mean(
                    global_affine_means
                )
            ),
        },

        "per_spectrum_scale_oracle_rmse": {
            "mean_across_seeds": float(
                np.mean(
                    sample_scale_means
                )
            ),
        },

        "per_spectrum_affine_oracle_rmse": {
            "mean_across_seeds": float(
                np.mean(
                    sample_affine_means
                )
            ),
        },
    }

    result[
        "aggregate"
    ] = aggregate

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

    print()
    print("=" * 76)
    print(
        "AGGREGATE DIAGNOSTIC"
    )
    print("=" * 76)

    print(
        "Native RMSE:",
        f"{np.mean(native_means):.6f}",
    )

    print(
        "Global scale-aligned RMSE:",
        f"{np.mean(global_scale_means):.6f}",
    )

    print(
        "Global affine-aligned RMSE:",
        f"{np.mean(global_affine_means):.6f}",
    )

    print(
        "Per-spectrum scale-oracle RMSE:",
        f"{np.mean(sample_scale_means):.6f}",
    )

    print(
        "Per-spectrum affine-oracle RMSE:",
        f"{np.mean(sample_affine_means):.6f}",
    )

    print()
    print(
        "Saved:",
        OUTPUT_JSON,
    )


if __name__ == "__main__":
    main()
