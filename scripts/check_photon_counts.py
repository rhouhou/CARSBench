from __future__ import annotations

import numpy as np

from CARSBench.api import generate_dataset


DOMAINS = [
    "A_typical",
    "D_high_noise",
]

NUM_SAMPLES = 500
SEED = 42


for domain_name in DOMAINS:
    batch = generate_dataset(
        num_samples=NUM_SAMPLES,
        domain_name=domain_name,
        seed=SEED,
    )

    medians = []
    means = []
    p95s = []
    maxima = []
    scales = []
    read_noises = []

    for sample in batch.samples:
        info = sample.metadata.parameters[
            "noise_info"
        ]

        medians.append(
            info[
                "expected_photoelectrons_median"
            ]
        )

        means.append(
            info[
                "expected_photoelectrons_mean"
            ]
        )

        p95s.append(
            info[
                "expected_photoelectrons_p95"
            ]
        )

        maxima.append(
            info[
                "expected_photoelectrons_max"
            ]
        )

        scales.append(
            info[
                "photoelectrons_per_intensity_unit"
            ]
        )

        read_noises.append(
            info[
                "read_noise_electrons"
            ]
        )

    medians = np.asarray(medians)
    means = np.asarray(means)
    p95s = np.asarray(p95s)
    maxima = np.asarray(maxima)
    scales = np.asarray(scales)
    read_noises = np.asarray(read_noises)

    print()
    print("=" * 70)
    print(domain_name)
    print("=" * 70)

    print(
        "photoelectron scale:",
        {
            "min": float(scales.min()),
            "median": float(np.median(scales)),
            "max": float(scales.max()),
        },
    )

    print(
        "read noise [e-]:",
        {
            "min": float(read_noises.min()),
            "median": float(np.median(read_noises)),
            "max": float(read_noises.max()),
        },
    )

    print(
        "expected count median per spectrum:",
        {
            "min": float(medians.min()),
            "median": float(np.median(medians)),
            "max": float(medians.max()),
        },
    )

    print(
        "expected count mean per spectrum:",
        {
            "min": float(means.min()),
            "median": float(np.median(means)),
            "max": float(means.max()),
        },
    )

    print(
        "expected count p95 per spectrum:",
        {
            "min": float(p95s.min()),
            "median": float(np.median(p95s)),
            "max": float(p95s.max()),
        },
    )

    print(
        "expected count maximum per spectrum:",
        {
            "min": float(maxima.min()),
            "median": float(np.median(maxima)),
            "max": float(maxima.max()),
        },
    )