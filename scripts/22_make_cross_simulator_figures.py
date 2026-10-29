
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULT_PATH = (
    ROOT
    / "results"
    / "cross_simulator"
    / "evaluation"
    / "cross_simulator_multiseed_evaluation.json"
)

OUTPUT_ROOT = (
    ROOT
    / "results"
    / "cross_simulator"
    / "figures"
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


def load_results():

    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            RESULT_PATH
        )

    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def save_figure(
    fig,
    name,
):

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        OUTPUT_ROOT
        / f"{name}.png"
    )

    pdf = (
        OUTPUT_ROOT
        / f"{name}.pdf"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        "Saved:",
        png,
    )

    print(
        "Saved:",
        pdf,
    )


def figure_external_seed_consistency(
    result,
):

    metrics = [
        (
            "rmse",
            "RMSE",
        ),
        (
            "spectral_angle",
            "Spectral angle (rad)",
        ),
        (
            "pearson",
            "Pearson correlation",
        ),
    ]

    for metric, ylabel in metrics:

        means = []

        model_stds = []

        for external_seed in EXTERNAL_SEEDS:

            summary = (
                result[
                    "external_seed_summary"
                ][
                    str(external_seed)
                ][
                    metric
                ]
            )

            means.append(
                summary["mean"]
            )

            model_stds.append(
                summary["std"]
            )

        fig, ax = plt.subplots(
            figsize=(6.5, 4.5)
        )

        x = np.arange(
            len(EXTERNAL_SEEDS)
        )

        ax.errorbar(
            x,
            means,
            yerr=model_stds,
            marker="o",
            capsize=4,
            linewidth=1.5,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                str(seed)
                for seed
                in EXTERNAL_SEEDS
            ]
        )

        ax.set_xlabel(
            "CRIMSON generation seed"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            "Cross-simulator performance "
            "across independent CRIMSON datasets"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"crimson_external_seed_{metric}",
        )


def figure_model_seed_sensitivity(
    result,
):

    metrics = [
        (
            "rmse",
            "RMSE",
        ),
        (
            "spectral_angle",
            "Spectral angle (rad)",
        ),
        (
            "pearson",
            "Pearson correlation",
        ),
    ]

    for metric, ylabel in metrics:

        means = []

        external_stds = []

        for model_seed in MODEL_SEEDS:

            summary = (
                result[
                    "model_seed_summary"
                ][
                    str(model_seed)
                ][
                    metric
                ]
            )

            means.append(
                summary["mean"]
            )

            external_stds.append(
                summary["std"]
            )

        fig, ax = plt.subplots(
            figsize=(6.5, 4.5)
        )

        x = np.arange(
            len(MODEL_SEEDS)
        )

        ax.errorbar(
            x,
            means,
            yerr=external_stds,
            marker="o",
            capsize=4,
            linewidth=1.5,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                str(seed)
                for seed
                in MODEL_SEEDS
            ]
        )

        ax.set_xlabel(
            "Model initialization seed"
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            "Cross-simulator sensitivity "
            "to model initialization"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"model_seed_sensitivity_{metric}",
        )


def figure_source_vs_external(
    result,
):

    source = (
        result[
            "source_id_reference"
        ]
    )

    external = (
        result[
            "primary_external_sampling_summary"
        ]
    )

    metrics = [
        (
            "rmse",
            "RMSE",
        ),
        (
            "mae",
            "MAE",
        ),
        (
            "spectral_angle",
            "Spectral angle",
        ),
    ]

    for metric, ylabel in metrics:

        source_value = (
            source[
                metric
            ][
                "mean_across_model_seeds"
            ]
        )

        external_value = (
            external[
                metric
            ][
                "mean"
            ]
        )

        external_std = (
            external[
                metric
            ][
                "std"
            ]
        )

        fig, ax = plt.subplots(
            figsize=(5.5, 4.5)
        )

        x = np.arange(
            2
        )

        ax.bar(
            x,
            [
                source_value,
                external_value,
            ],
        )

        ax.errorbar(
            [1],
            [external_value],
            yerr=[external_std],
            fmt="none",
            capsize=4,
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            [
                "CARSBench\nID",
                "CRIMSON\nzero-shot",
            ]
        )

        ax.set_ylabel(
            ylabel
        )

        ax.set_title(
            "Matched fingerprint-range transfer"
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        save_figure(
            fig,
            f"source_vs_crimson_{metric}",
        )


def write_summary(
    result,
):

    output = (
        OUTPUT_ROOT
        / "cross_simulator_publication_summary.txt"
    )

    primary = (
        result[
            "primary_external_sampling_summary"
        ]
    )

    degradation = (
        result[
            "cross_simulator_degradation"
        ]
    )

    lines = []

    lines.append(
        "CARSBench cross-simulator experiment"
    )

    lines.append(
        "===================================="
    )

    lines.append("")

    lines.append(
        "Protocol:"
    )

    lines.append(
        "3 source model seeds × "
        "3 independent CRIMSON generation seeds"
    )

    lines.append(
        "5000 CRIMSON spectra per external seed"
    )

    lines.append(
        "Matched physical fingerprint range: "
        "400–1798.63 cm^-1"
    )

    lines.append(
        "Strict zero-shot transfer; "
        "source normalization only."
    )

    lines.append("")

    lines.append(
        "Primary results:"
    )

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "pearson",
        "nrmse_target_rms",
        "prediction_to_target_rms_ratio",
    ]:

        summary = (
            primary[
                metric
            ]
        )

        lines.append(
            f"{metric}: "
            f"{summary['mean']:.6f} "
            f"± {summary['std']:.6f} "
            "(external-seed SD)"
        )

    lines.append("")

    lines.append(
        "Degradation vs matched source ID:"
    )

    for metric in [
        "rmse",
        "mae",
        "spectral_angle",
        "nrmse_target_rms",
    ]:

        lines.append(
            f"{metric}: "
            f"{degradation[metric]['ratio']:.3f}x"
        )

    lines.append(
        "Pearson absolute drop: "
        f"{degradation['pearson']['absolute_drop']:.6f}"
    )

    lines.append("")
    lines.append(
        "Interpretation:"
    )

    lines.append(
        "The independently generated CRIMSON "
        "datasets produced nearly identical "
        "cross-simulator degradation."
    )

    lines.append(
        "Variation across external generation "
        "seeds was much smaller than variation "
        "across model initialization seeds."
    )

    lines.append(
        "The failure therefore cannot be "
        "attributed to one anomalous external "
        "sample realization."
    )

    output.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "Saved:",
        output,
    )


def main():

    result = (
        load_results()
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure_external_seed_consistency(
        result
    )

    figure_model_seed_sensitivity(
        result
    )

    figure_source_vs_external(
        result
    )

    write_summary(
        result
    )

    print()
    print(
        "Cross-simulator publication "
        "figures complete."
    )


if __name__ == "__main__":
    main()
