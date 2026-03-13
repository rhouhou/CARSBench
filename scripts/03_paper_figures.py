from __future__ import annotations

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from CARSBench.api import generate_dataset, generate_multi_domain_dataset


OUTPUT_DIR = "paper_figures"


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=300)
    plt.close()


def get_common_axis(samples, num_points: int = 1024) -> np.ndarray:
    mins = [float(s.axis.min()) for s in samples]
    maxs = [float(s.axis.max()) for s in samples]
    ref_min = max(mins)
    ref_max = min(maxs)
    return np.linspace(ref_min, ref_max, num_points, dtype=np.float64)


def figure_forward_model_examples() -> None:
    batch = generate_dataset(num_samples=4, domain_name="A_typical", seed=10)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for i, sample in enumerate(batch.samples):
        axes[i].plot(sample.axis, sample.raman_target, label="Raman target", linewidth=2)
        axes[i].plot(sample.axis, sample.spectrum, label="BCARS measurement", linewidth=2)
        if i == 0:
            axes[i].legend()

    axes[-1].set_xlabel("Wavenumber (cm⁻¹)")
    plt.suptitle("Forward BCARS simulation")
    savefig("figure_forward_model_examples.png")


def figure_pca_diversity() -> None:
    domains = [
        "A_typical",
        "B_high_res",
        "C_low_res_noisy",
        "D_calibration_shift",
        "F_nrb_family_shift",
        "G_biochemical_source",
        "H_biochemical_target",
    ]
    batch = generate_multi_domain_dataset(domain_names=domains, samples_per_domain=200, seed=42)

    ref_axis = get_common_axis(batch.samples)
    X = np.array([np.interp(ref_axis, s.axis, s.spectrum) for s in batch.samples], dtype=np.float64)
    labels = [s.domain_name for s in batch.samples]

    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for domain in domains:
        idx = [i for i, lab in enumerate(labels) if lab == domain]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.6, label=domain)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Spectral diversity PCA")
    plt.legend()
    savefig("figure_pca_diversity.png")


def figure_mean_std_domains() -> None:
    domains = [
        "A_typical",
        "B_high_res",
        "C_low_res_noisy",
        "D_calibration_shift",
        "F_nrb_family_shift",
        "G_biochemical_source",
        "H_biochemical_target",
    ]
    domain_batches = {
        d: generate_dataset(num_samples=100, domain_name=d, seed=42).samples
        for d in domains
    }

    all_samples = [x for samples in domain_batches.values() for x in samples]
    ref_axis = get_common_axis(all_samples)

    plt.figure(figsize=(10, 6))

    all_means = {}
    global_max = 0.0
    for domain, samples in domain_batches.items():
        spectra = np.array([np.interp(ref_axis, s.axis, s.spectrum) for s in samples])
        mean_spec = spectra.mean(axis=0)
        std_spec = spectra.std(axis=0)
        all_means[domain] = (mean_spec, std_spec)
        global_max = max(global_max, float(mean_spec.max()))

    offset_step = 0.8 * global_max if global_max > 0 else 1.0
    offset = 0.0

    for domain, (mean_spec, std_spec) in all_means.items():
        y = mean_spec + offset
        plt.plot(ref_axis, y, linewidth=2, label=domain)
        plt.fill_between(ref_axis, y - std_spec, y + std_spec, alpha=0.22)
        offset += offset_step

    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Mean intensity (offset)")
    plt.title("Mean ± std spectrum per domain")
    plt.legend()
    savefig("figure_mean_std_domains.png")


def main() -> None:
    ensure_output_dir()
    figure_forward_model_examples()
    figure_pca_diversity()
    figure_mean_std_domains()
    print(f"Saved paper figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()