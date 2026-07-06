from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_SEEDS = [42, 123, 777]
G_DOMAIN = "G_biochemical_source"
H_DOMAIN = "H_biochemical_target"


def canonical_axis() -> np.ndarray:
    return np.linspace(400.0, 3200.0, 1024, dtype=np.float64)


def get_nested(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def interpolate_rows(
    axis_2d: np.ndarray, y_2d: np.ndarray, target_axis: np.ndarray
) -> np.ndarray:
    out = np.empty((y_2d.shape[0], target_axis.size), dtype=np.float32)

    for i in range(y_2d.shape[0]):
        axis = np.asarray(axis_2d[i], dtype=np.float64)
        y = np.asarray(y_2d[i], dtype=np.float64)

        order = np.argsort(axis)
        axis = axis[order]
        y = y[order]

        out[i] = np.interp(target_axis, axis, y).astype(np.float32)

    return out


def load_domain_mean_std(
    data_root: Path,
    seed: int,
    domain: str,
    value_key: str = "raman_target",
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    batch_dir = data_root / f"seed_{seed}" / domain / "batches"
    if not batch_dir.exists():
        print(f"Missing: {batch_dir}")
        return None

    batch_files = sorted(batch_dir.glob("*.npz"))
    if max_batches is not None:
        batch_files = batch_files[:max_batches]

    target_axis = canonical_axis()
    n_total = 0
    sum_y = np.zeros(target_axis.size, dtype=np.float64)
    sum_y2 = np.zeros(target_axis.size, dtype=np.float64)

    for bf in batch_files:
        with np.load(bf, allow_pickle=False) as data:
            if "axis" not in data or value_key not in data:
                continue

            axis_2d = data["axis"]
            y_2d = data[value_key]
            Y = interpolate_rows(axis_2d, y_2d, target_axis)

            sum_y += Y.sum(axis=0, dtype=np.float64)
            sum_y2 += np.square(Y, dtype=np.float32).sum(axis=0, dtype=np.float64)
            n_total += Y.shape[0]

    if n_total == 0:
        return None

    mean_y = sum_y / n_total
    var_y = np.maximum(sum_y2 / n_total - mean_y**2, 0.0)
    std_y = np.sqrt(var_y)

    return target_axis, mean_y, std_y, n_total


def load_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def aggregate_component_counts(rows: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()

    for row in rows:
        comps = get_nested(row, "parameters.resonant_info.selected_components")
        if comps is None:
            comps = get_nested(
                row, "parameters.resolved_config.resonant.allowed_components"
            )

        if comps is None:
            continue

        if isinstance(comps, list):
            for c in comps:
                counter[str(c)] += 1
        else:
            counter[str(comps)] += 1

    return counter


def aggregate_peak_centers(rows: list[dict[str, Any]]) -> list[float]:
    centers: list[float] = []

    for row in rows:
        vals = get_nested(row, "parameters.resonant_info.peak_centers")
        if vals is None:
            continue
        if isinstance(vals, list):
            for v in vals:
                try:
                    centers.append(float(v))
                except (TypeError, ValueError):
                    pass

    return centers


def aggregate_peak_sources(rows: list[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()

    for row in rows:
        vals = get_nested(row, "parameters.resonant_info.peak_sources")
        if vals is None:
            continue
        if isinstance(vals, list):
            for v in vals:
                counter[str(v)] += 1

    return counter


def plot_raman_overlay(
    data_root: Path,
    output_path: Path,
    seed: int,
    max_batches: int | None = None,
    std_scale: float = 1.0,
) -> None:
    g = load_domain_mean_std(data_root, seed, G_DOMAIN, "raman_target", max_batches)
    h = load_domain_mean_std(data_root, seed, H_DOMAIN, "raman_target", max_batches)

    if g is None or h is None:
        print(f"Skipping overlay for seed {seed}")
        return

    axis_g, mean_g, std_g, n_g = g
    axis_h, mean_h, std_h, n_h = h

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(axis_g, mean_g, linewidth=2.0, label=f"{G_DOMAIN} (N={n_g})")
    ax.fill_between(
        axis_g, mean_g - std_scale * std_g, mean_g + std_scale * std_g, alpha=0.18
    )

    ax.plot(axis_h, mean_h, linewidth=2.0, label=f"{H_DOMAIN} (N={n_h})")
    ax.fill_between(
        axis_h, mean_h - std_scale * std_h, mean_h + std_scale * std_h, alpha=0.18
    )

    ax.set_title(f"G vs H | seed {seed} | raman_target mean ± {std_scale:.0f} std")
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Raman target")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_component_counts(
    data_root: Path,
    output_path: Path,
    seed: int,
) -> None:
    g_meta = data_root / f"seed_{seed}" / G_DOMAIN / "metadata" / "metadata.jsonl"
    h_meta = data_root / f"seed_{seed}" / H_DOMAIN / "metadata" / "metadata.jsonl"

    if not g_meta.exists() or not h_meta.exists():
        print(f"Missing metadata for seed {seed}")
        return

    g_rows = load_metadata_jsonl(g_meta)
    h_rows = load_metadata_jsonl(h_meta)

    g_counts = aggregate_component_counts(g_rows)
    h_counts = aggregate_component_counts(h_rows)

    categories = sorted(set(g_counts.keys()) | set(h_counts.keys()))
    if not categories:
        print(f"No component counts found for seed {seed}")
        return

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2, [g_counts.get(c, 0) for c in categories], width, label=G_DOMAIN
    )
    ax.bar(
        x + width / 2, [h_counts.get(c, 0) for c in categories], width, label=H_DOMAIN
    )

    ax.set_title(f"G vs H | seed {seed} | selected component counts")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    print(f"Seed {seed} | G component counts: {dict(g_counts)}")
    print(f"Seed {seed} | H component counts: {dict(h_counts)}")


def plot_peak_center_hist(
    data_root: Path,
    output_path: Path,
    seed: int,
    bins: int = 80,
) -> None:
    g_meta = data_root / f"seed_{seed}" / G_DOMAIN / "metadata" / "metadata.jsonl"
    h_meta = data_root / f"seed_{seed}" / H_DOMAIN / "metadata" / "metadata.jsonl"

    if not g_meta.exists() or not h_meta.exists():
        print(f"Missing metadata for seed {seed}")
        return

    g_rows = load_metadata_jsonl(g_meta)
    h_rows = load_metadata_jsonl(h_meta)

    g_centers = aggregate_peak_centers(g_rows)
    h_centers = aggregate_peak_centers(h_rows)

    if len(g_centers) == 0 or len(h_centers) == 0:
        print(f"No peak centers found for seed {seed}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(g_centers, bins=bins, alpha=0.45, density=True, label=G_DOMAIN)
    ax.hist(h_centers, bins=bins, alpha=0.45, density=True, label=H_DOMAIN)

    ax.set_title(f"G vs H | seed {seed} | peak center distribution")
    ax.set_xlabel("Peak center (cm$^{-1}$)")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")

    print(
        f"Seed {seed} | peak centers | "
        f"G mean={np.mean(g_centers):.2f}, H mean={np.mean(h_centers):.2f}, "
        f"G n={len(g_centers)}, H n={len(h_centers)}"
    )


def plot_peak_source_counts(
    data_root: Path,
    output_path: Path,
    seed: int,
) -> None:
    g_meta = data_root / f"seed_{seed}" / G_DOMAIN / "metadata" / "metadata.jsonl"
    h_meta = data_root / f"seed_{seed}" / H_DOMAIN / "metadata" / "metadata.jsonl"

    if not g_meta.exists() or not h_meta.exists():
        print(f"Missing metadata for seed {seed}")
        return

    g_rows = load_metadata_jsonl(g_meta)
    h_rows = load_metadata_jsonl(h_meta)

    g_counts = aggregate_peak_sources(g_rows)
    h_counts = aggregate_peak_sources(h_rows)

    categories = sorted(set(g_counts.keys()) | set(h_counts.keys()))
    if not categories:
        print(f"No peak-source counts found for seed {seed}")
        return

    x = np.arange(len(categories))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2, [g_counts.get(c, 0) for c in categories], width, label=G_DOMAIN
    )
    ax.bar(
        x + width / 2, [h_counts.get(c, 0) for c in categories], width, label=H_DOMAIN
    )

    ax.set_title(f"G vs H | seed {seed} | peak source counts")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, default="figures/chemistry_validation"
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--std-scale", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"

        plot_raman_overlay(
            data_root=data_root,
            output_path=seed_dir / "G_vs_H_raman_target_overlay.png",
            seed=seed,
            max_batches=args.max_batches,
            std_scale=args.std_scale,
        )

        plot_component_counts(
            data_root=data_root,
            output_path=seed_dir / "G_vs_H_selected_component_counts.png",
            seed=seed,
        )

        plot_peak_center_hist(
            data_root=data_root,
            output_path=seed_dir / "G_vs_H_peak_center_hist.png",
            seed=seed,
            bins=args.bins,
        )

        plot_peak_source_counts(
            data_root=data_root,
            output_path=seed_dir / "G_vs_H_peak_source_counts.png",
            seed=seed,
        )


if __name__ == "__main__":
    main()
