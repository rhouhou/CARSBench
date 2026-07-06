from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res_noisy",
    "D_calibration_shift",
    "E_window_shift",
    "F_nrb_family_shift",
    "G_biochemical_source",
    "H_biochemical_target",
]

DEFAULT_SEEDS = [42, 123, 777]

DEFAULT_NUMERIC_PARAMETERS = [
    ("parameters.instrument_info.psf_fwhm", "PSF FWHM (cm^-1)", False),
    ("parameters.noise_info.shot_scale", "Shot scale", True),
    ("parameters.noise_info.read_sigma", "Read sigma", False),
    ("parameters.noise_info.spike_prob", "Spike probability", True),
    ("parameters.calibration_info.shift_cm1", "Calibration shift (cm^-1)", False),
    ("parameters.calibration_info.warp_cm1", "Calibration warp (cm^-1)", False),
    ("parameters.nrb_info.alpha", "NRB alpha", True),
    ("parameters.nrb_info.phase_total_change", "NRB phase total change", False),
    ("parameters.resolved_config.axis.nu_min", "Axis nu_min (cm^-1)", False),
    ("parameters.resolved_config.axis.nu_max", "Axis nu_max (cm^-1)", False),
    ("parameters.resolved_config.axis.num_points", "Number of points", False),
]


def get_nested(d: dict[str, Any], path: str) -> Any:
    """
    Access nested dict entries using dot notation.
    Example:
        get_nested(obj, "parameters.instrument_info.psf_fwhm")
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(".", "_")
    text = text.replace("/", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def load_parameter_values(
    data_root: Path,
    domains: list[str],
    seeds: list[int],
    parameter_path: str,
) -> dict[str, dict[int, list[float]]]:
    values: dict[str, dict[int, list[float]]] = {
        domain: {seed: [] for seed in seeds} for domain in domains
    }

    for seed in seeds:
        for domain in domains:
            metadata_path = (
                data_root / f"seed_{seed}" / domain / "metadata" / "metadata.jsonl"
            )

            if not metadata_path.exists():
                print(f"WARNING: metadata not found: {metadata_path}")
                continue

            with metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    value = get_nested(row, parameter_path)

                    if value is None:
                        continue

                    try:
                        values[domain][seed].append(float(value))
                    except (TypeError, ValueError):
                        continue

    return values


def plot_grouped_boxplots(
    values: dict[str, dict[int, list[float]]],
    domains: list[str],
    seeds: list[int],
    title: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
    showfliers: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))

    n_domains = len(domains)
    n_seeds = len(seeds)

    domain_centers = np.arange(n_domains) * 2.0
    offsets = np.linspace(-0.45, 0.45, n_seeds)

    positions = []
    box_data = []
    seed_for_box = []

    for d_idx, domain in enumerate(domains):
        for s_idx, seed in enumerate(seeds):
            vals = values[domain][seed]
            if len(vals) == 0:
                continue

            positions.append(domain_centers[d_idx] + offsets[s_idx])
            box_data.append(vals)
            seed_for_box.append(seed)

    if len(box_data) == 0:
        print(f"WARNING: no numeric data found for {title}")
        plt.close(fig)
        return

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=showfliers,
    )

    seed_to_color = {
        seeds[0]: "#4C72B0",
        seeds[1]: "#DD8452",
        seeds[2]: "#55A868",
    }

    for patch, seed in zip(bp["boxes"], seed_for_box):
        patch.set_facecolor(seed_to_color.get(seed, "#999999"))
        patch.set_alpha(0.8)

    for median in bp["medians"]:
        median.set_linewidth(1.5)

    ax.set_xticks(domain_centers)
    ax.set_xticklabels(domains, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if log_y:
        positive_vals = [
            v
            for domain in domains
            for seed in seeds
            for v in values[domain][seed]
            if v > 0
        ]
        if len(positive_vals) > 0:
            ax.set_yscale("log")
        else:
            print(
                f"WARNING: skipped log scale for {title} because no positive values were found."
            )

    handles = [
        plt.Line2D([0], [0], color=seed_to_color[s], lw=8, label=f"seed {s}")
        for s in seeds
    ]
    ax.legend(handles=handles, title="Seed", loc="best")

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.close(fig)


def generate_all_boxplots(
    data_root: Path,
    output_dir: Path,
    domains: list[str],
    seeds: list[int],
    parameter_specs: list[tuple[str, str, bool]],
    showfliers: bool = False,
) -> None:
    for parameter_path, ylabel, log_y in parameter_specs:
        print(f"Processing: {parameter_path}")

        values = load_parameter_values(
            data_root=data_root,
            domains=domains,
            seeds=seeds,
            parameter_path=parameter_path,
        )

        filename = f"{safe_name(parameter_path)}_boxplot.png"
        output_path = output_dir / filename
        title = f"{ylabel} across domains and seeds"

        plot_grouped_boxplots(
            values=values,
            domains=domains,
            seeds=seeds,
            title=title,
            ylabel=ylabel,
            output_path=output_path,
            log_y=log_y,
            showfliers=showfliers,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing seed_42, seed_123, seed_777, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/parameter_boxplots",
        help="Directory where all output figures will be saved.",
    )
    parser.add_argument(
        "--showfliers",
        action="store_true",
        help="Show outlier points in boxplots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    generate_all_boxplots(
        data_root=data_root,
        output_dir=output_dir,
        domains=DEFAULT_DOMAINS,
        seeds=DEFAULT_SEEDS,
        parameter_specs=DEFAULT_NUMERIC_PARAMETERS,
        showfliers=args.showfliers,
    )


if __name__ == "__main__":
    main()
