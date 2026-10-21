from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CARSBench.api import generate_dataset, list_domains


OUTPUT_DIR = Path("qc/step1_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 100
SEED = 42


def summarize_domain(domain_name: str):
    batch = generate_dataset(
        num_samples=NUM_SAMPLES,
        domain_name=domain_name,
        seed=SEED,
    )

    spectra = np.stack(
        [sample.spectrum for sample in batch.samples]
    )

    targets = np.stack(
        [sample.raman_target for sample in batch.samples]
    )

    ratios = np.array(
        [
            sample.metadata.parameters[
                "resonant_info"
            ]["realized_rnr_ratio"]
            for sample in batch.samples
        ],
        dtype=float,
    )

    peak_counts = np.array(
        [
            sample.metadata.parameters[
                "resonant_info"
            ].get("num_peaks", np.nan)
            for sample in batch.samples
        ],
        dtype=float,
    )

    print()
    print("=" * 70)
    print(domain_name)
    print("=" * 70)

    print("Samples:", len(batch.samples))

    print(
        "Spectrum finite:",
        np.all(np.isfinite(spectra)),
    )

    print(
        "Target finite:",
        np.all(np.isfinite(targets)),
    )

    print(
        "Spectrum min/max:",
        float(np.min(spectra)),
        float(np.max(spectra)),
    )

    print(
        "Target min/max:",
        float(np.min(targets)),
        float(np.max(targets)),
    )

    print(
        "R/NRB ratio:",
        {
            "min": float(np.min(ratios)),
            "median": float(np.median(ratios)),
            "mean": float(np.mean(ratios)),
            "max": float(np.max(ratios)),
        },
    )

    if not np.all(np.isnan(peak_counts)):
        print(
            "Peak count:",
            {
                "min": float(np.nanmin(peak_counts)),
                "median": float(np.nanmedian(peak_counts)),
                "mean": float(np.nanmean(peak_counts)),
                "max": float(np.nanmax(peak_counts)),
            },
        )

    # ---------------------------------------------------------
    # Plot a few samples
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 9),
        sharex=True,
    )

    for sample in batch.samples[:3]:
        axes[0].plot(
            sample.axis,
            sample.raman_target,
            alpha=0.8,
        )

        axes[1].plot(
            sample.axis,
            sample.clean_intensity,
            alpha=0.8,
        )

        axes[2].plot(
            sample.axis,
            sample.spectrum,
            alpha=0.8,
        )

    axes[0].set_ylabel("Raman target")
    axes[1].set_ylabel("Clean CARS")
    axes[2].set_ylabel("Measured CARS")
    axes[2].set_xlabel("Raman shift (cm$^{-1}$)")

    fig.suptitle(domain_name)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / f"{domain_name}.png",
        dpi=150,
    )

    plt.close(fig)


def main():
    domains = list_domains()

    print("Domains:")
    for domain in domains:
        print(" -", domain)

    for domain in domains:
        summarize_domain(domain)


if __name__ == "__main__":
    main()