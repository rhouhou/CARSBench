import csv
from pathlib import Path

import numpy as np

import CARSBench as cb


def rmse(prediction, target):
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def mae(prediction, target):
    return float(np.mean(np.abs(prediction - target)))


def spectral_angle(prediction, target, eps=1e-12):
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    numerator = np.sum(prediction * target)
    denominator = np.linalg.norm(prediction) * np.linalg.norm(target) + eps

    cosine = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.arccos(cosine))


def normalize(signal, eps=1e-12):
    signal = np.asarray(signal)
    return (signal - signal.min()) / (signal.max() - signal.min() + eps)


def smooth_signal(signal, window_size=15):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode="same")


def zero_baseline(spectrum):
    return np.zeros_like(spectrum)


def normalized_input_baseline(spectrum):
    return normalize(spectrum)


def smoothed_input_baseline(spectrum):
    return normalize(smooth_signal(spectrum))


BASELINES = {
    "zero": zero_baseline,
    "normalized_input": normalized_input_baseline,
    "smoothed_input": smoothed_input_baseline,
}


def evaluate_domain(domain_name, num_samples=100, seed=42):
    batch = cb.generate_dataset(
        num_samples=num_samples,
        domain_name=domain_name,
        seed=seed,
    )

    rows = []

    for baseline_name, baseline_fn in BASELINES.items():
        rmses = []
        maes = []
        angles = []

        for sample in batch.samples:
            target = normalize(sample.raman_target)
            prediction = baseline_fn(sample.spectrum)

            rmses.append(rmse(prediction, target))
            maes.append(mae(prediction, target))
            angles.append(spectral_angle(prediction, target))

        rows.append(
            {
                "domain": domain_name,
                "baseline": baseline_name,
                "num_samples": num_samples,
                "seed": seed,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "spectral_angle_mean": float(np.mean(angles)),
                "spectral_angle_std": float(np.std(angles)),
            }
        )

    return rows


def main():
    output_dir = Path("results/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    domains = [
        "A_typical",
        "B_high_res",
        "C_low_res_noisy",
        "D_calibration_shift",
        "E_window_shift",
        "F_nrb_family_shift",
        "G_biochemical_source",
        "H_biochemical_target",
    ]

    all_rows = []

    for domain in domains:
        print(f"Evaluating {domain}...")
        all_rows.extend(evaluate_domain(domain_name=domain, num_samples=100, seed=42))

    output_path = output_dir / "baseline_results.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved baseline results to {output_path}")


if __name__ == "__main__":
    main()
