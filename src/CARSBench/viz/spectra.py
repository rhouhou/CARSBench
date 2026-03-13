from __future__ import annotations

import matplotlib.pyplot as plt

from CARSBench.datasets.schema import SpectrumSample


def plot_sample(sample: SpectrumSample) -> None:
    """
    Plot one simulated sample with measured spectrum and Raman target.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(sample.axis, sample.spectrum, label="bCARS")
    plt.plot(sample.axis, sample.raman_target, label="Raman target")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sample_with_latents(sample: SpectrumSample) -> None:
    """
    Plot sample plus latent resonant / NRB terms when available.
    """
    plt.figure(figsize=(9, 5))
    plt.plot(sample.axis, sample.spectrum, label="Measured bCARS")
    plt.plot(sample.axis, sample.raman_target, label="Raman target")

    if sample.chi_r_imag is not None:
        plt.plot(sample.axis, sample.chi_r_imag, label="Im{chi_R}", linestyle="--")

    if sample.chi_nr_real is not None and sample.chi_nr_imag is not None:
        plt.plot(
            sample.axis,
            (sample.chi_nr_real**2 + sample.chi_nr_imag**2) ** 0.5,
            label="|chi_NR|",
            linestyle=":",
        )

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Signal")
    plt.legend()
    plt.tight_layout()
    plt.show()