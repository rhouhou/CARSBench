from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CARSBench.datasets.reader import DatasetReader


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


def canonical_axis() -> np.ndarray:
    return np.linspace(400.0, 3200.0, 1024, dtype=np.float64)


def interpolate_samples(samples, value_key: str, target_axis: np.ndarray) -> np.ndarray:
    ys = []

    for s in samples:
        axis = np.asarray(s.axis, dtype=np.float64)
        y = np.asarray(getattr(s, value_key), dtype=np.float64)

        order = np.argsort(axis)
        axis = axis[order]
        y = y[order]

        y_interp = np.interp(target_axis, axis, y)
        ys.append(y_interp)

    return np.stack(ys, axis=0)


def region_stats(axis: np.ndarray, mean_y: np.ndarray, std_y: np.ndarray, lo: float, hi: float) -> dict:
    mask = (axis >= lo) & (axis <= hi)
    if not np.any(mask):
        return {
            "region_mean": np.nan,
            "region_std_mean": np.nan,
            "region_std_max": np.nan,
        }

    return {
        "region_mean": float(np.mean(mean_y[mask])),
        "region_std_mean": float(np.mean(std_y[mask])),
        "region_std_max": float(np.max(std_y[mask])),
    }


def plot_random_spectra(samples, output_path: Path, title: str, n_show: int, rng_seed: int) -> None:
    rng = np.random.default_rng(rng_seed)
    n_show = min(n_show, len(samples))
    idx = rng.choice(len(samples), size=n_show, replace=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i in idx:
        s = samples[int(i)]
        ax.plot(s.axis, s.spectrum, linewidth=1.0, alpha=0.7)
        #ax.plot(s.axis, s.raman_target, linewidth=1.0, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mean_std(samples, output_path: Path, title: str, value_key: str = "spectrum") -> dict:
    target_axis = canonical_axis()
    Y = interpolate_samples(samples, value_key=value_key, target_axis=target_axis)

    mean_y = np.mean(Y, axis=0)
    std_y = np.std(Y, axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(target_axis, mean_y, linewidth=1.8, label="Mean")
    ax.fill_between(target_axis, mean_y - 2 * std_y, mean_y + 2 * std_y, alpha=0.25, label="Mean ± 2 std")

    ax.set_title(title)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fp_stats = region_stats(target_axis, mean_y, std_y, 700.0, 1800.0)
    ch_stats = region_stats(target_axis, mean_y, std_y, 2800.0, 3050.0)

    return {
        "global_mean_of_mean": float(np.mean(mean_y)),
        "global_mean_of_std": float(np.mean(std_y)),
        "global_max_std": float(np.max(std_y)),
        "fingerprint": fp_stats,
        "ch_stretch": ch_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures/spectra_validation")
    parser.add_argument("--n-show", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    for seed in DEFAULT_SEEDS:
        for domain in DEFAULT_DOMAINS:
            domain_root = data_root / f"seed_{seed}" / domain
            if not domain_root.exists():
                print(f"Skipping missing: {domain_root}")
                continue

            print(f"\nProcessing {domain} | seed {seed}")
            reader = DatasetReader(domain_root)
            batch = reader.read_all_batches()
            samples = batch.samples

            if len(samples) == 0:
                print("  No samples found.")
                continue

            plot_random_spectra(
                samples=samples,
                output_path=output_dir / f"seed_{seed}" / f"{domain}_random_spectra.png",
                title=f"{domain} | seed {seed} | random spectra",
                n_show=args.n_show,
                rng_seed=seed,
            )

            stats = plot_mean_std(
                samples=samples,
                output_path=output_dir / f"seed_{seed}" / f"{domain}_mean_std.png",
                title=f"{domain} | seed {seed} | mean ± 2 std",
                value_key="spectrum",
            )

            print(
                f"  global: mean(mean)={stats['global_mean_of_mean']:.4g}, "
                f"mean(std)={stats['global_mean_of_std']:.4g}, "
                f"max(std)={stats['global_max_std']:.4g}"
            )
            print(
                f"  fingerprint: mean={stats['fingerprint']['region_mean']:.4g}, "
                f"mean(std)={stats['fingerprint']['region_std_mean']:.4g}, "
                f"max(std)={stats['fingerprint']['region_std_max']:.4g}"
            )
            print(
                f"  CH stretch: mean={stats['ch_stretch']['region_mean']:.4g}, "
                f"mean(std)={stats['ch_stretch']['region_std_mean']:.4g}, "
                f"max(std)={stats['ch_stretch']['region_std_max']:.4g}"
            )


if __name__ == "__main__":
    main()