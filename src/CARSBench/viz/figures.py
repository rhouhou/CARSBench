from __future__ import annotations

from CARSBench.datasets.schema import SampleBatch

from .domains import plot_domain_pca, plot_mean_spectrum_per_domain
from .spectra import plot_sample
from .validation import (
    plot_domain_histogram,
    plot_example_spectra,
    plot_nrb_examples,
    plot_raman_targets,
)


def make_validation_figure_set(batch: SampleBatch) -> None:
    """
    Generate a basic validation figure sequence for a batch.
    """
    plot_example_spectra(batch)
    plot_raman_targets(batch)
    plot_nrb_examples(batch)
    plot_domain_histogram(batch)
    plot_domain_pca(batch)
    plot_mean_spectrum_per_domain(batch)


def make_single_sample_figure(sample) -> None:
    """
    Convenience wrapper for a single example sample.
    """
    plot_sample(sample)