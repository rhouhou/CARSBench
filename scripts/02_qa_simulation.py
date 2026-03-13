from __future__ import annotations

import os
from collections import Counter
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from CARSBench.api import generate_multi_domain_dataset


OUTPUT_DIR = "qa_plots"
DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res_noisy",
    "D_calibration_shift",
    "E_window_shift",
    "F_nrb_family_shift",
    "G_biochemical_source",
    "H_biochemical_target",
]
PCA_DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res_noisy",
    "D_calibration_shift",
    "F_nrb_family_shift",
    "G_biochemical_source",
    "H_biochemical_target",
]
SAMPLES_PER_DOMAIN = 100


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_peaks(signal: np.ndarray) -> int:
    peaks = 0
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            peaks += 1
    return peaks


def get_common_axis(samples, num_points: int = 1024) -> np.ndarray:
    mins = [float(s.axis.min()) for s in samples]
    maxs = [float(s.axis.max()) for s in samples]
    ref_min = max(mins)
    ref_max = min(maxs)
    if ref_max <= ref_min:
        raise ValueError("No overlapping axis region across selected samples.")
    return np.linspace(ref_min, ref_max, num_points, dtype=np.float64)


def plot_example_spectra(domain_batches: dict[str, list]) -> None:
    domains = list(domain_batches.keys())
    n_cols = 3

    fig, axes = plt.subplots(
        len(domains),
        n_cols,
        figsize=(4 * n_cols, 2.5 * len(domains)),
        sharey=False,
    )

    for r, domain in enumerate(domains):
        for c in range(n_cols):
            sample = domain_batches[domain][c]
            ax = axes[r, c]
            ax.plot(sample.axis, sample.spectrum, linewidth=1.3)
            if c == 0:
                ax.set_ylabel(domain)
            if r == 0:
                ax.set_title(f"Example {c+1}")

    axes[-1, 0].set_xlabel("Wavenumber (cm⁻¹)")
    plt.suptitle("Example BCARS spectra per domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "example_spectra_per_domain.png"), dpi=300)
    plt.close()


def plot_mean_std_spectrum_per_domain(domain_batches: dict[str, list]) -> None:
    all_samples = [s for samples in domain_batches.values() for s in samples]
    ref_axis = get_common_axis(all_samples)

    plt.figure(figsize=(10, 6))

    domain_means = {}
    global_max = 0.0
    for domain, samples in domain_batches.items():
        spectra = np.array([np.interp(ref_axis, s.axis, s.spectrum) for s in samples])
        mean_spec = spectra.mean(axis=0)
        domain_means[domain] = (mean_spec, spectra.std(axis=0))
        global_max = max(global_max, float(mean_spec.max()))

    offset_step = 0.8 * global_max if global_max > 0 else 1.0
    offset = 0.0

    for domain, (mean_spec, std_spec) in domain_means.items():
        mean_offset = mean_spec + offset
        plt.plot(ref_axis, mean_offset, label=domain, linewidth=2)
        plt.fill_between(
            ref_axis,
            mean_offset - std_spec,
            mean_offset + std_spec,
            alpha=0.22,
        )
        offset += offset_step

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Mean intensity (offset)")
    plt.title("Mean ± std spectrum per domain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_std_spectrum_per_domain.png"), dpi=300)
    plt.close()


def plot_raman_vs_bcars(domain_batches: dict[str, list]) -> None:
    fig, axes = plt.subplots(len(domain_batches), 1, figsize=(8, 2.4 * len(domain_batches)))

    for i, (domain, samples) in enumerate(domain_batches.items()):
        sample = samples[0]
        axes[i].plot(sample.axis, sample.raman_target, label="Raman target", linewidth=1.5)
        axes[i].plot(sample.axis, sample.spectrum, label="BCARS", linewidth=1.5)
        axes[i].set_ylabel(domain)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    plt.suptitle("Raman vs BCARS per domain")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "raman_vs_bcars_per_domain.png"), dpi=300)
    plt.close()


def plot_pca(batch) -> None:
    pca_samples = [s for s in batch.samples if s.domain_name in PCA_DOMAINS]
    ref_axis = get_common_axis(pca_samples)

    X = np.array([np.interp(ref_axis, s.axis, s.spectrum) for s in pca_samples], dtype=np.float64)
    labels = [s.domain_name for s in pca_samples]

    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    print("PCA explained variance:", pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 6))
    for d in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == d]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=12, alpha=0.6, label=d)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Spectral diversity PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_diversity.png"), dpi=300)
    plt.close()


def plot_domain_difference_heatmap(domain_batches: dict[str, list], exclude_domains=None, filename="domain_difference_heatmap.png") -> None:
    if exclude_domains is None:
        exclude_domains = []

    filtered = {d: s for d, s in domain_batches.items() if d not in exclude_domains}
    all_samples = [x for samples in filtered.values() for x in samples]
    ref_axis = get_common_axis(all_samples)

    domain_means = {}
    for domain, samples in filtered.items():
        spectra = np.array([np.interp(ref_axis, s.axis, s.spectrum) for s in samples])
        domain_means[domain] = spectra.mean(axis=0)

    domains = list(domain_means.keys())
    diff_matrix = []
    labels = []

    for d1, d2 in itertools.combinations(domains, 2):
        diff_matrix.append(np.abs(domain_means[d1] - domain_means[d2]))
        labels.append(f"{d1} vs {d2}")

    diff_matrix = np.array(diff_matrix)

    plt.figure(figsize=(12, 6))
    plt.imshow(
        diff_matrix,
        aspect="auto",
        extent=[ref_axis[0], ref_axis[-1], 0, len(labels)],
        cmap="magma",
    )
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.title("Domain difference heatmap")
    plt.colorbar(label="Absolute spectral difference")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def main() -> None:
    ensure_output_dir()

    batch = generate_multi_domain_dataset(
        domain_names=DOMAINS,
        samples_per_domain=SAMPLES_PER_DOMAIN,
        seed=42,
    )

    domain_batches = {d: [] for d in DOMAINS}
    for sample in batch.samples:
        domain_batches[sample.domain_name].append(sample)

    lengths = [len(s.spectrum) for s in batch.samples]
    proto_counts = [
        len(s.metadata.parameters["resonant_info"]["selected_components"])
        for s in batch.samples
    ]
    peak_counts = [count_peaks(s.raman_target) for s in batch.samples]

    corrs = []
    for s in batch.samples:
        c = np.corrcoef(s.spectrum, s.raman_target)[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    print("Total samples:", len(batch.samples))
    print("Domain distribution:", Counter([s.domain_name for s in batch.samples]))
    print("Spectrum lengths:", Counter(lengths))
    print("Prototype counts:", Counter(proto_counts))
    print("Peak count mean/min/max:", float(np.mean(peak_counts)), int(np.min(peak_counts)), int(np.max(peak_counts)))
    print("Raman vs BCARS corr mean/min/max:", float(np.mean(corrs)), float(np.min(corrs)), float(np.max(corrs)))

    plot_example_spectra(domain_batches)
    plot_mean_std_spectrum_per_domain(domain_batches)
    plot_raman_vs_bcars(domain_batches)
    plot_pca(batch)
    plot_domain_difference_heatmap(domain_batches, exclude_domains=None, filename="domain_difference_heatmap_with_E.png")
    plot_domain_difference_heatmap(domain_batches, exclude_domains=["E_window_shift"], filename="domain_difference_heatmap_without_E.png")

    print(f"Saved QA plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()