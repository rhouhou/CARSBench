from __future__ import annotations

import numpy as np

from CARSBench.api import generate_dataset


DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res",
    "D_high_noise",
    "E_calibration_shift",
    "F_window_shift",
    "G_system_response_shift",
    "H_biochemical_source",
    "I_biochemical_target",
]

NUM_SAMPLES = 1000
SEED = 42


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main() -> None:
    for domain_name in DOMAINS:
        batch = generate_dataset(
            num_samples=NUM_SAMPLES,
            domain_name=domain_name,
            seed=SEED,
        )

        ratios = []

        targets = []

        for sample in batch.samples:
            info = sample.metadata.parameters[
                "resonant_info"
            ]

            ratios.append(
                float(
                    info[
                        "realized_rnr_ratio"
                    ]
                )
            )

            targets.append(
                info.get(
                    "target_rnr_ratio"
                )
            )

        ratios = np.asarray(
            ratios,
            dtype=np.float64,
        )

        print()
        print("=" * 72)
        print(domain_name)
        print("=" * 72)

        print(
            "R/NRB ratio:",
            summarize(ratios),
        )

        target_values = np.asarray(
            targets,
            dtype=np.float64,
        )

        print(
            "Target R/NRB:",
            summarize(
                target_values
            ),
        )


if __name__ == "__main__":
    main()