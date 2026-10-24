from __future__ import annotations

import numpy as np

from CARSBench.api import generate_dataset


DOMAIN = "A_typical"

SEEDS = [
    42,
    123,
    777,
    2026,
    9876,
]

NUM_SAMPLES = 5000


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "min": float(
            np.min(values)
        ),
        "p01": float(
            np.percentile(
                values,
                1.0,
            )
        ),
        "p05": float(
            np.percentile(
                values,
                5.0,
            )
        ),
        "p25": float(
            np.percentile(
                values,
                25.0,
            )
        ),
        "median": float(
            np.median(values)
        ),
        "mean": float(
            np.mean(values)
        ),
        "p75": float(
            np.percentile(
                values,
                75.0,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95.0,
            )
        ),
        "p99": float(
            np.percentile(
                values,
                99.0,
            )
        ),
        "max": float(
            np.max(values)
        ),
    }


def collect_ratios(
    seed: int,
) -> np.ndarray:
    batch = generate_dataset(
        num_samples=NUM_SAMPLES,
        domain_name=DOMAIN,
        seed=seed,
    )

    values = [
        float(
            sample.metadata.parameters[
                "resonant_info"
            ][
                "realized_rnr_ratio"
            ]
        )
        for sample in batch.samples
    ]

    return np.asarray(
        values,
        dtype=np.float64,
    )


def log_distribution_summary(
    values: np.ndarray,
) -> dict[str, float]:
    log_values = np.log(
        values
    )

    mu = float(
        np.mean(log_values)
    )

    sigma = float(
        np.std(
            log_values,
            ddof=1,
        )
    )

    return {
        "log_mean_mu": mu,
        "log_std_sigma": sigma,
        "implied_median": float(
            np.exp(mu)
        ),
        "implied_mean_if_lognormal": float(
            np.exp(
                mu
                + 0.5 * sigma**2
            )
        ),
    }


def main() -> None:
    all_values = []

    print(
        f"Domain: {DOMAIN}"
    )

    print(
        f"Samples per seed: {NUM_SAMPLES}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    for seed in SEEDS:
        ratios = collect_ratios(
            seed
        )

        all_values.append(
            ratios
        )

        print()
        print(
            "=" * 72
        )

        print(
            f"seed = {seed}"
        )

        print(
            "=" * 72
        )

        print(
            "R/NRB:",
            summarize(
                ratios
            ),
        )

        print(
            "log-space:",
            log_distribution_summary(
                ratios
            ),
        )

    pooled = np.concatenate(
        all_values
    )

    print()
    print(
        "=" * 72
    )

    print(
        "POOLED"
    )

    print(
        "=" * 72
    )

    print(
        f"Total samples: {len(pooled)}"
    )

    print(
        "R/NRB:",
        summarize(
            pooled
        ),
    )

    print(
        "log-space:",
        log_distribution_summary(
            pooled
        ),
    )


if __name__ == "__main__":
    main()