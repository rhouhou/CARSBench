from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from CARSBench.datasets.schema import SampleBatch


def _make_reference_axis(
    batch: SampleBatch,
    num_points: int = 1024,
) -> np.ndarray:
    """
    Build a common reference axis covering the overlap region
    shared across all samples.
    """
    mins = [float(sample.axis.min()) for sample in batch.samples]
    maxs = [float(sample.axis.max()) for sample in batch.samples]

    ref_min = max(mins)
    ref_max = min(maxs)

    if ref_max <= ref_min:
        raise ValueError(
            "No overlapping axis region across samples. "
            "Cannot build a common reference axis."
        )

    return np.linspace(ref_min, ref_max, num_points, dtype=np.float64)


def _resample_batch_to_reference_axis(
    batch: SampleBatch,
    num_points: int = 1024,
):
    """
    Interpolate all spectra onto a common reference axis.
    """
    ref_axis = _make_reference_axis(batch, num_points=num_points)

    spectra = []
    domains = []

    for sample in batch.samples:
        interp_spec = np.interp(ref_axis, sample.axis, sample.spectrum)
        spectra.append(interp_spec)
        domains.append(sample.domain_name)

    spectra = np.stack(spectra, axis=0)
    return ref_axis, spectra, domains


def pca_projection(
    X: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """
    Simple PCA projection without sklearn dependency.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)

    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order[:n_components]]

    return X @ eigvecs


def plot_domain_pca(
    batch: SampleBatch,
    num_points: int = 1024,
) -> None:
    """
    Visualize domain shift using PCA on spectra interpolated
    to a common reference axis.
    """
    _, X, domains = _resample_batch_to_reference_axis(
        batch,
        num_points=num_points,
    )

    proj = pca_projection(X, n_components=2)

    groups = defaultdict(list)
    for point, domain in zip(proj, domains):
        groups[domain].append(point)

    plt.figure(figsize=(6, 6))

    for domain, points in groups.items():
        points = np.asarray(points)
        plt.scatter(points[:, 0], points[:, 1], label=domain, alpha=0.6)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Domain shift visualization (PCA)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mean_spectrum_per_domain(
    batch: SampleBatch,
    num_points: int = 1024,
) -> None:
    """
    Plot mean measured spectrum per domain after interpolation
    to a common reference axis.
    """
    ref_axis, spectra, domains = _resample_batch_to_reference_axis(
        batch,
        num_points=num_points,
    )

    groups = defaultdict(list)
    for spectrum, domain in zip(spectra, domains):
        groups[domain].append(spectrum)

    plt.figure(figsize=(8, 4))

    for domain, domain_spectra in groups.items():
        mean_spec = np.stack(domain_spectra, axis=0).mean(axis=0)
        plt.plot(ref_axis, mean_spec, label=domain)

    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean intensity")
    plt.title("Mean spectrum per domain")
    plt.legend()
    plt.tight_layout()
    plt.show()