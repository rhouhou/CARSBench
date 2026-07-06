from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
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

# (parameter_path, y_label)
DEFAULT_CATEGORICAL_PARAMETERS = [
    ("parameters.instrument_info.envelope_family", "Envelope family"),
    ("parameters.nrb_info.family", "NRB family"),
    ("parameters.nrb_info.phase_model", "NRB phase model"),
    ("parameters.resolved_config.axis.window_mode", "Axis window mode"),
]


def get_nested(d: dict[str, Any], path: str) -> Any:
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


def load_categorical_counts(
    data_root: Path,
    domains: list[str],
    seeds: list[int],
    parameter_path: str,
) -> tuple[dict[str, dict[int, dict[str, int]]], list[str]]:
    """
    Returns:
        counts[domain][seed][category] = count
        categories = sorted list of all observed categories
    """
    counts: dict[str, dict[int, dict[str, int]]] = {
        domain: {seed: defaultdict(int) for seed in seeds} for domain in domains
    }

    all_categories: set[str] = set()

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

                    category = str(value)
                    counts[domain][seed][category] += 1
                    all_categories.add(category)

    categories = sorted(all_categories)
    return counts, categories


def convert_counts_to_fractions(
    counts: dict[str, dict[int, dict[str, int]]],
    domains: list[str],
    seeds: list[int],
    categories: list[str],
) -> dict[str, dict[int, dict[str, float]]]:
    fractions: dict[str, dict[int, dict[str, float]]] = {
        domain: {seed: {} for seed in seeds} for domain in domains
    }

    for domain in domains:
        for seed in seeds:
            total = sum(counts[domain][seed].get(cat, 0) for cat in categories)

            for cat in categories:
                c = counts[domain][seed].get(cat, 0)
                fractions[domain][seed][cat] = (c / total) if total > 0 else 0.0

    return fractions


def plot_grouped_categorical_bars(
    values: dict[str, dict[int, dict[str, float]]],
    categories: list[str],
    domains: list[str],
    seeds: list[int],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    For each domain, show 3 seed groups.
    Inside each seed group, show one bar per category.
    """
    fig, ax = plt.subplots(figsize=(18, 7))

    n_domains = len(domains)
    n_seeds = len(seeds)
    n_categories = len(categories)

    domain_centers = np.arange(n_domains) * 3.0
    seed_offsets = np.linspace(-0.7, 0.7, n_seeds)

    # Width of each category bar inside a seed-group
    total_seed_group_width = 0.55
    bar_width = total_seed_group_width / max(n_categories, 1)
    category_offsets = np.linspace(
        -total_seed_group_width / 2 + bar_width / 2,
        total_seed_group_width / 2 - bar_width / 2,
        n_categories,
    )

    cmap = plt.get_cmap("tab10")
    category_to_color = {cat: cmap(i % 10) for i, cat in enumerate(categories)}

    for d_idx, domain in enumerate(domains):
        for s_idx, seed in enumerate(seeds):
            seed_center = domain_centers[d_idx] + seed_offsets[s_idx]

            for c_idx, category in enumerate(categories):
                x = seed_center + category_offsets[c_idx]
                y = values[domain][seed].get(category, 0.0)

                ax.bar(
                    x,
                    y,
                    width=bar_width,
                    color=category_to_color[category],
                    alpha=0.85,
                )

    ax.set_xticks(domain_centers)
    ax.set_xticklabels(domains, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    # Category legend
    category_handles = [
        plt.Rectangle((0, 0), 1, 1, color=category_to_color[cat], alpha=0.85, label=cat)
        for cat in categories
    ]
    legend1 = ax.legend(
        handles=category_handles,
        title="Category",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(legend1)

    # Seed guide legend
    seed_handles = [
        plt.Line2D(
            [0], [0], color="black", lw=0, marker="|", markersize=18, label=f"seed {s}"
        )
        for s in seeds
    ]
    ax.legend(
        handles=seed_handles,
        title="Seed groups",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.close(fig)


def generate_all_categorical_plots(
    data_root: Path,
    output_dir: Path,
    domains: list[str],
    seeds: list[int],
    parameter_specs: list[tuple[str, str]],
) -> None:
    for parameter_path, ylabel in parameter_specs:
        print(f"Processing: {parameter_path}")

        counts, categories = load_categorical_counts(
            data_root=data_root,
            domains=domains,
            seeds=seeds,
            parameter_path=parameter_path,
        )

        if len(categories) == 0:
            print(f"WARNING: no categorical data found for {parameter_path}")
            continue

        fractions = convert_counts_to_fractions(
            counts=counts,
            domains=domains,
            seeds=seeds,
            categories=categories,
        )

        filename = f"{safe_name(parameter_path)}_categorical_barplot.png"
        output_path = output_dir / filename
        title = f"{ylabel} distribution across domains and seeds"

        plot_grouped_categorical_bars(
            values=fractions,
            categories=categories,
            domains=domains,
            seeds=seeds,
            title=title,
            ylabel="Fraction",
            output_path=output_path,
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
        default="figures/categorical_parameter_barplots",
        help="Directory where all output figures will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    generate_all_categorical_plots(
        data_root=data_root,
        output_dir=output_dir,
        domains=DEFAULT_DOMAINS,
        seeds=DEFAULT_SEEDS,
        parameter_specs=DEFAULT_CATEGORICAL_PARAMETERS,
    )


if __name__ == "__main__":
    main()
