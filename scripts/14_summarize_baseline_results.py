from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/content/CARSBench")

RESULTS = ROOT / "results"

SCALE_ROOT = RESULTS / "scale_study"

ACQ_ROOT = SCALE_ROOT / "acquisition"
CHEM_ROOT = SCALE_ROOT / "chemistry"

FINAL_ACQ_PATH = (
    RESULTS
    / "final_evaluation"
    / "acquisition"
    / "final_acquisition_evaluation.json"
)

F_BREAKDOWN_PATH = (
    RESULTS
    / "final_evaluation"
    / "acquisition"
    / "F_window_mode_breakdown.json"
)

FINAL_CHEM_PATH = (
    RESULTS
    / "final_evaluation"
    / "chemistry"
    / "final_chemistry_evaluation.json"
)

OUTPUT_ROOT = RESULTS / "baseline_summary"
PLOT_ROOT = OUTPUT_ROOT / "plots"


TRAIN_SIZES = [
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
]

SEEDS = [
    101,
    202,
    303,
]


# ============================================================
# Basic helpers
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:

    if not rows:
        return

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def stats(
    values: list[float],
) -> dict:

    array = np.asarray(
        values,
        dtype=float,
    )

    mean = float(
        np.mean(array)
    )

    std = float(
        np.std(
            array,
            ddof=1,
        )
    )

    cv = float(
        100.0 * std / mean
    )

    return {
        "mean": mean,
        "std": std,
        "cv_percent": cv,
    }


# ============================================================
# Scale studies
# ============================================================

def read_scale_track(
    *,
    root: Path,
    prefix: str,
) -> tuple[
    list[dict],
    list[dict],
]:

    individual = []
    aggregate = []

    for size in TRAIN_SIZES:

        size_values = []

        for seed in SEEDS:

            path = (
                root
                / f"{prefix}_train{size}_seed{seed}"
                / "selection_result.json"
            )

            data = load_json(path)

            if int(
                data["train_size"]
            ) != size:
                raise RuntimeError(
                    f"Wrong train_size in {path}"
                )

            if int(
                data["model_seed"]
            ) != seed:
                raise RuntimeError(
                    f"Wrong model_seed in {path}"
                )

            mse = float(
                data[
                    "best_validation_mse"
                ]
            )

            size_values.append(mse)

            individual.append(
                {
                    "train_size": size,
                    "model_seed": seed,
                    "best_epoch": int(
                        data[
                            "best_epoch"
                        ]
                    ),
                    "best_validation_mse": mse,
                    "epochs_completed": int(
                        data[
                            "epochs_completed"
                        ]
                    ),
                    "training_minutes": float(
                        data[
                            "training_seconds"
                        ]
                    )
                    / 60.0,
                }
            )

        result = stats(
            size_values
        )

        aggregate.append(
            {
                "train_size": size,
                "n_seeds": 3,
                "validation_mse_mean": (
                    result["mean"]
                ),
                "validation_mse_std": (
                    result["std"]
                ),
                "validation_mse_cv_percent": (
                    result["cv_percent"]
                ),
            }
        )

    return (
        individual,
        aggregate,
    )


# ============================================================
# Generic final acquisition parser
# ============================================================

def is_metric_block(
    value,
) -> bool:

    return (
        isinstance(
            value,
            dict,
        )
        and "rmse" in value
        and "mae" in value
        and (
            "spectral_angle" in value
            or "angle" in value
        )
    )


def metric_pair(
    value,
) -> tuple[
    float,
    float,
]:

    if isinstance(
        value,
        dict,
    ):
        return (
            float(
                value["mean"]
            ),
            float(
                value.get(
                    "std",
                    0.0,
                )
            ),
        )

    return (
        float(value),
        0.0,
    )


def collect_metric_blocks(
    value,
    path=(),
):
    result = []

    if isinstance(
        value,
        dict,
    ):

        if is_metric_block(value):
            result.append(
                (
                    path,
                    value,
                )
            )

        for key, child in value.items():
            result.extend(
                collect_metric_blocks(
                    child,
                    path
                    + (
                        str(key),
                    ),
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):
            result.extend(
                collect_metric_blocks(
                    child,
                    path
                    + (
                        str(index),
                    ),
                )
            )

    return result


def parse_acquisition_final(
    data: dict,
) -> list[dict]:

    nodes = collect_metric_blocks(
        data
    )

    aggregate_nodes = [
        item
        for item in nodes
        if any(
            "aggregate"
            in token.lower()
            for token
            in item[0]
        )
    ]

    if aggregate_nodes:
        nodes = aggregate_nodes

    rows = {}

    ignored = {
        "aggregate",
        "metrics",
        "global",
        "results",
        "final",
        "evaluation",
        "domains",
    }

    for path, block in nodes:

        candidates = [
            token
            for token in path
            if (
                token.lower()
                not in ignored
                and not token.isdigit()
            )
        ]

        if not candidates:
            continue

        domain = candidates[-1]

        rmse_mean, rmse_std = metric_pair(
            block["rmse"]
        )

        mae_mean, mae_std = metric_pair(
            block["mae"]
        )

        angle_key = (
            "spectral_angle"
            if "spectral_angle"
            in block
            else "angle"
        )

        angle_mean, angle_std = metric_pair(
            block[angle_key]
        )

        rows[domain] = {
            "domain": domain,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "spectral_angle_mean": (
                angle_mean
            ),
            "spectral_angle_std": (
                angle_std
            ),
        }

    preferred_order = [
        "A_typical_id_test",
        "B_high_res",
        "C_low_res",
        "D_high_noise",
        "E_calibration_shift",
        "F_window_shift",
        "observed",
        "unobserved",
        "G_system_response_shift",
        "J_compound_acquisition",
    ]

    ordered = []

    for name in preferred_order:
        if name in rows:
            ordered.append(
                rows[name]
            )

    return ordered


# ============================================================
# Chemistry final
# ============================================================

def parse_chemistry_final(
    data: dict,
) -> list[dict]:

    aggregate = data[
        "aggregate_global"
    ]

    rows = []

    for domain, block in aggregate.items():

        rmse_mean, rmse_std = metric_pair(
            block["rmse"]
        )

        mae_mean, mae_std = metric_pair(
            block["mae"]
        )

        angle_mean, angle_std = metric_pair(
            block[
                "spectral_angle"
            ]
        )

        rows.append(
            {
                "domain": domain,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "mae_mean": mae_mean,
                "mae_std": mae_std,
                "spectral_angle_mean": (
                    angle_mean
                ),
                "spectral_angle_std": (
                    angle_std
                ),
            }
        )

    return rows


# ============================================================
# F window breakdown
# ============================================================

def parse_f_breakdown(
    data: dict,
) -> list[dict]:

    modes = [
        "fingerprint",
        "ch",
        "partial_fingerprint",
        "partial_ch",
    ]

    rows = []

    def find_mode(
        value,
        mode,
    ):
        if isinstance(
            value,
            dict,
        ):

            if (
                mode in value
                and isinstance(
                    value[mode],
                    dict,
                )
                and all(
                    key in value[mode]
                    for key in [
                        "full",
                        "observed",
                        "unobserved",
                    ]
                )
            ):
                return value[mode]

            for child in value.values():

                found = find_mode(
                    child,
                    mode,
                )

                if found is not None:
                    return found

        return None

    for mode in modes:

        block = find_mode(
            data,
            mode,
        )

        if block is None:
            raise RuntimeError(
                f"Cannot find F mode {mode}"
            )

        for region in [
            "full",
            "observed",
            "unobserved",
        ]:

            metrics = block[
                region
            ]

            rmse_mean, rmse_std = metric_pair(
                metrics["rmse"]
            )

            mae_mean, mae_std = metric_pair(
                metrics["mae"]
            )

            angle_key = (
                "spectral_angle"
                if "spectral_angle"
                in metrics
                else "angle"
            )

            angle_mean, angle_std = metric_pair(
                metrics[angle_key]
            )

            rows.append(
                {
                    "mode": mode,
                    "region": region,
                    "rmse_mean": rmse_mean,
                    "rmse_std": rmse_std,
                    "mae_mean": mae_mean,
                    "mae_std": mae_std,
                    "spectral_angle_mean": (
                        angle_mean
                    ),
                    "spectral_angle_std": (
                        angle_std
                    ),
                }
            )

    return rows


# ============================================================
# Glycogen
# ============================================================

def parse_glycogen(
    data: dict,
) -> list[dict]:

    aggregate = data[
        "aggregate_glycogen_484"
    ]

    wanted = [
        "target_peak_mean",
        "predicted_peak_mean",
        "peak_bias_mean",
        "peak_mae",
        "relative_peak_mae",
        "peak_correlation",
        "peak_slope_through_origin",
        "target_area_mean",
        "predicted_area_mean",
        "area_bias_mean",
        "area_mae",
        "relative_area_mae",
        "area_correlation",
    ]

    rows = []

    for domain, block in aggregate.items():

        row = {
            "domain": domain
        }

        for key in wanted:

            value = block[key]

            if isinstance(
                value,
                dict,
            ):

                row[
                    f"{key}_mean"
                ] = float(
                    value["mean"]
                )

                row[
                    f"{key}_std"
                ] = float(
                    value.get(
                        "std",
                        0.0,
                    )
                )

            else:

                row[
                    f"{key}_mean"
                ] = float(value)

                row[
                    f"{key}_std"
                ] = 0.0

        rows.append(row)

    return rows


# ============================================================
# Plots
# ============================================================

def plot_learning_curve(
    rows,
    title,
    path,
):

    x = np.array(
        [
            row["train_size"]
            for row in rows
        ]
    )

    y = np.array(
        [
            row[
                "validation_mse_mean"
            ]
            for row in rows
        ]
    )

    error = np.array(
        [
            row[
                "validation_mse_std"
            ]
            for row in rows
        ]
    )

    fig, ax = plt.subplots(
        figsize=(7, 4.5)
    )

    ax.errorbar(
        x,
        y,
        yerr=error,
        marker="o",
        capsize=4,
    )

    ax.set_xscale("log")

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{size // 1000}k"
            for size in x
        ]
    )

    ax.set_xlabel(
        "Training samples"
    )

    ax.set_ylabel(
        "Validation normalized MSE"
    )

    ax.set_title(title)

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_rmse(
    rows,
    title,
    path,
):

    labels = [
        row["domain"]
        for row in rows
    ]

    values = [
        row["rmse_mean"]
        for row in rows
    ]

    errors = [
        row["rmse_std"]
        for row in rows
    ]

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                7,
                len(labels) * 0.9,
            ),
            4.8,
        )
    )

    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel("RMSE")

    ax.set_title(title)

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# Printing
# ============================================================

def print_learning_curve(
    title,
    rows,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for row in rows:

        print(
            f"{row['train_size']:>6d}  "
            f"MSE "
            f"{row['validation_mse_mean']:.8f} "
            f"± "
            f"{row['validation_mse_std']:.8f}  "
            f"CV "
            f"{row['validation_mse_cv_percent']:.2f}%"
        )


def print_final(
    title,
    rows,
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    for row in rows:

        print()
        print(
            row["domain"]
        )

        print(
            "  RMSE   "
            f"{row['rmse_mean']:.6f} "
            f"± {row['rmse_std']:.6f}"
        )

        print(
            "  MAE    "
            f"{row['mae_mean']:.6f} "
            f"± {row['mae_std']:.6f}"
        )

        print(
            "  angle  "
            f"{row['spectral_angle_mean']:.6f} "
            f"± {row['spectral_angle_std']:.6f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Reading saved scale-study results..."
    )

    (
        acq_individual,
        acq_curve,
    ) = read_scale_track(
        root=ACQ_ROOT,
        prefix="A",
    )

    (
        chem_individual,
        chem_curve,
    ) = read_scale_track(
        root=CHEM_ROOT,
        prefix="H",
    )

    print(
        "Acquisition scale runs found: "
        f"{len(acq_individual)}/15"
    )

    print(
        "Chemistry scale runs found:   "
        f"{len(chem_individual)}/15"
    )

    acquisition_data = load_json(
        FINAL_ACQ_PATH
    )

    chemistry_data = load_json(
        FINAL_CHEM_PATH
    )

    f_data = load_json(
        F_BREAKDOWN_PATH
    )

    acquisition_final = (
        parse_acquisition_final(
            acquisition_data
        )
    )

    chemistry_final = (
        parse_chemistry_final(
            chemistry_data
        )
    )

    f_breakdown = (
        parse_f_breakdown(
            f_data
        )
    )

    glycogen = (
        parse_glycogen(
            chemistry_data
        )
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------
    write_csv(
        OUTPUT_ROOT
        / "acquisition_learning_curve.csv",
        acq_curve,
    )

    write_csv(
        OUTPUT_ROOT
        / "chemistry_learning_curve.csv",
        chem_curve,
    )

    write_csv(
        OUTPUT_ROOT
        / "acquisition_scale_runs.csv",
        acq_individual,
    )

    write_csv(
        OUTPUT_ROOT
        / "chemistry_scale_runs.csv",
        chem_individual,
    )

    write_csv(
        OUTPUT_ROOT
        / "final_acquisition.csv",
        acquisition_final,
    )

    write_csv(
        OUTPUT_ROOT
        / "final_chemistry.csv",
        chemistry_final,
    )

    write_csv(
        OUTPUT_ROOT
        / "F_window_modes.csv",
        f_breakdown,
    )

    write_csv(
        OUTPUT_ROOT
        / "glycogen_484_recovery.csv",
        glycogen,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_learning_curve(
        acq_curve,
        "Acquisition-track training-size study",
        PLOT_ROOT
        / "acquisition_learning_curve.png",
    )

    plot_learning_curve(
        chem_curve,
        "Chemistry-track training-size study",
        PLOT_ROOT
        / "chemistry_learning_curve.png",
    )

    plot_rmse(
        acquisition_final,
        "Final acquisition-domain generalization",
        PLOT_ROOT
        / "final_acquisition_rmse.png",
    )

    plot_rmse(
        chemistry_final,
        "Final chemistry-domain generalization",
        PLOT_ROOT
        / "final_chemistry_rmse.png",
    )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------
    summary = {
        "benchmark": "CARSBench",
        "baseline_model": "CARSBaselineCNN",

        "model_seeds": SEEDS,

        "selected_training_sizes": {
            "acquisition": 100_000,
            "chemistry": 100_000,
        },

        "selection_statement": (
            "100k was the best-performing training "
            "size within the predefined candidate range "
            "for both source tracks; saturation beyond "
            "100k was not established."
        ),

        "acquisition_learning_curve": (
            acq_curve
        ),

        "chemistry_learning_curve": (
            chem_curve
        ),

        "final_acquisition": (
            acquisition_final
        ),

        "final_chemistry": (
            chemistry_final
        ),

        "F_window_mode_breakdown": (
            f_breakdown
        ),

        "glycogen_484_recovery": (
            glycogen
        ),
    }

    summary_path = (
        OUTPUT_ROOT
        / "baseline_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    # --------------------------------------------------------
    # Terminal
    # --------------------------------------------------------
    print_learning_curve(
        "ACQUISITION LEARNING CURVE",
        acq_curve,
    )

    print_learning_curve(
        "CHEMISTRY LEARNING CURVE",
        chem_curve,
    )

    print_final(
        "FINAL ACQUISITION",
        acquisition_final,
    )

    print_final(
        "FINAL CHEMISTRY",
        chemistry_final,
    )

    print()
    print("=" * 80)
    print(
        "BASELINE CONSOLIDATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Summary JSON: {summary_path}"
    )

    print(
        f"Tables:       {OUTPUT_ROOT}"
    )

    print(
        f"Plots:        {PLOT_ROOT}"
    )


if __name__ == "__main__":
    main()
