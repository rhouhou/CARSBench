from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from CARSBench.datasets.schema import SampleBatch


def plot_example_spectra(
    batch: SampleBatch,
    num_samples: int = 5,
) -> None:
    """
    Plot example measured spectra from a batch.
    """
    plt.figure(figsize=(8, 4))

    for sample in batch.samples[:num_samples]:
        plt.plot(sample.axis, sample.spectrum, alpha=0.7)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("bCARS intensity")
    plt.title("Example simulated spectra")
    plt.tight_layout()
    plt.show()


def plot_raman_targets(
    batch: SampleBatch,
    num_samples: int = 5,
) -> None:
    """
    Plot example Raman targets.
    """
    plt.figure(figsize=(8, 4))

    for sample in batch.samples[:num_samples]:
        plt.plot(sample.axis, sample.raman_target, alpha=0.7)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Raman target")
    plt.title("Ground-truth Raman spectra")
    plt.tight_layout()
    plt.show()


def plot_nrb_examples(
    batch: SampleBatch,
    num_samples: int = 10,
) -> None:
    """
    Plot example NRB magnitude curves when latent NRB is available.
    """
    plt.figure(figsize=(8, 4))

    for sample in batch.samples[:num_samples]:
        if sample.chi_nr_real is None or sample.chi_nr_imag is None:
            continue

        chi_nr = sample.chi_nr_real + 1j * sample.chi_nr_imag
        mag = np.abs(chi_nr)

        plt.plot(sample.axis, mag, alpha=0.6)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("|chi_NR|")
    plt.title("Example NRB shapes")
    plt.tight_layout()
    plt.show()


def plot_domain_histogram(batch: SampleBatch) -> None:
    """
    Plot the number of samples per domain.
    """
    counts: dict[str, int] = {}

    for sample in batch.samples:
        counts[sample.domain_name] = counts.get(sample.domain_name, 0) + 1

    plt.figure(figsize=(6, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.ylabel("Number of samples")
    plt.title("Domain distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def plot_resolution_effect(sample_low, sample_high) -> None:
    """
    Compare two spectra with different resolution conditions.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(sample_low.axis, sample_low.spectrum, label="low resolution")
    plt.plot(sample_high.axis, sample_high.spectrum, label="high resolution")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Resolution effect on spectra")
    plt.tight_layout()
    plt.show()


def plot_noise_effect(sample_clean, sample_noisy) -> None:
    """
    Compare cleaner vs noisier spectra.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(sample_clean.axis, sample_clean.spectrum, label="clean")
    plt.plot(sample_noisy.axis, sample_noisy.spectrum, label="noisy")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.title("Noise model effect")
    plt.tight_layout()
    plt.show()