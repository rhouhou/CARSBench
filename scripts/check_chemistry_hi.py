from __future__ import annotations

from collections import Counter

import numpy as np

from CARSBench.api import generate_dataset


H_DOMAIN = "H_biochemical_source"
I_DOMAIN = "I_biochemical_target"

NUM_SAMPLES = 5000
SEED = 42

NUM_RANDOM_PAIRS = 10000
PAIR_SEED = 12345


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def spectral_angle(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    na = float(
        np.linalg.norm(a)
    )

    nb = float(
        np.linalg.norm(b)
    )

    if (
        na <= 1e-12
        or nb <= 1e-12
    ):
        return np.nan

    cosine = float(
        np.dot(
            a,
            b,
        )
        / (na * nb)
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.arccos(
            cosine
        )
    )


def collect_domain(
    domain_name: str,
):
    batch = generate_dataset(
        num_samples=NUM_SAMPLES,
        domain_name=domain_name,
        seed=SEED,
    )

    targets = np.stack(
        [
            sample.raman_target
            for sample in batch.samples
        ],
        axis=0,
    )

    metadata = [
        sample.metadata.parameters[
            "resonant_info"
        ]
        for sample in batch.samples
    ]

    return (
        batch,
        targets,
        metadata,
    )


def print_component_summary(
    domain_name: str,
    metadata: list[dict],
) -> None:
    occurrence = Counter()
    count_distribution = Counter()

    peak_counts = []
    minor_component_fractions = []

    glycogen_weights = []

    for info in metadata:
        components = list(
            info.get(
                "selected_components",
                [],
            )
        )

        weights = np.asarray(
            info.get(
                "mixture_weights",
                [],
            ),
            dtype=np.float64,
        )

        n_components = int(
            info.get(
                "n_components",
                len(components),
            )
        )

        count_distribution[
            n_components
        ] += 1

        peak_counts.append(
            int(
                info.get(
                    "num_peaks",
                    0,
                )
            )
        )

        for component in components:
            occurrence[
                component
            ] += 1

        if len(weights) >= 2:
            minor_component_fractions.append(
                float(
                    np.min(weights)
                )
            )

        if "glycogen" in components:
            idx = components.index(
                "glycogen"
            )

            glycogen_weights.append(
                float(
                    weights[idx]
                )
            )

    print()
    print("=" * 72)
    print(domain_name)
    print("=" * 72)

    print(
        "Component occurrence counts:",
        dict(
            sorted(
                occurrence.items()
            )
        ),
    )

    print(
        "Number of components per sample:",
        dict(
            sorted(
                count_distribution.items()
            )
        ),
    )

    print(
        "Peak count:",
        summarize(
            np.asarray(
                peak_counts,
                dtype=np.float64,
            )
        ),
    )

    if minor_component_fractions:
        print(
            "Smallest mixture weight per sample:",
            summarize(
                np.asarray(
                    minor_component_fractions,
                    dtype=np.float64,
                )
            ),
        )

    if glycogen_weights:
        print(
            "Glycogen mixture weight:",
            summarize(
                np.asarray(
                    glycogen_weights,
                    dtype=np.float64,
                )
            ),
        )


def random_within_domain_angles(
    targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    angles = []

    n = len(
        targets
    )

    while len(
        angles
    ) < NUM_RANDOM_PAIRS:
        i = int(
            rng.integers(
                0,
                n,
            )
        )

        j = int(
            rng.integers(
                0,
                n,
            )
        )

        if i == j:
            continue

        angle = spectral_angle(
            targets[i],
            targets[j],
        )

        if np.isfinite(angle):
            angles.append(
                angle
            )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


def random_cross_domain_angles(
    h_targets: np.ndarray,
    i_targets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    angles = []

    for _ in range(
        NUM_RANDOM_PAIRS
    ):
        h_idx = int(
            rng.integers(
                0,
                len(h_targets),
            )
        )

        i_idx = int(
            rng.integers(
                0,
                len(i_targets),
            )
        )

        angle = spectral_angle(
            h_targets[h_idx],
            i_targets[i_idx],
        )

        if np.isfinite(angle):
            angles.append(
                angle
            )

    return np.asarray(
        angles,
        dtype=np.float64,
    )


def main() -> None:
    (
        h_batch,
        h_targets,
        h_metadata,
    ) = collect_domain(
        H_DOMAIN
    )

    (
        i_batch,
        i_targets,
        i_metadata,
    ) = collect_domain(
        I_DOMAIN
    )

    print_component_summary(
        H_DOMAIN,
        h_metadata,
    )

    print_component_summary(
        I_DOMAIN,
        i_metadata,
    )

    rng = np.random.default_rng(
        PAIR_SEED
    )

    h_within = (
        random_within_domain_angles(
            h_targets,
            rng,
        )
    )

    i_within = (
        random_within_domain_angles(
            i_targets,
            rng,
        )
    )

    cross = (
        random_cross_domain_angles(
            h_targets,
            i_targets,
            rng,
        )
    )

    print()
    print("=" * 72)
    print("SHAPE-ONLY SPECTRAL ANGLES")
    print("=" * 72)

    print(
        "H vs H:",
        summarize(
            h_within
        ),
    )

    print(
        "I vs I:",
        summarize(
            i_within
        ),
    )

    print(
        "H vs I:",
        summarize(
            cross
        ),
    )

    # ---------------------------------------------------------
    # Membership invariants
    # ---------------------------------------------------------

    h_allowed = {
        "lipid",
        "protein",
        "nucleic_acid",
    }

    for info in h_metadata:
        components = set(
            info.get(
                "selected_components",
                [],
            )
        )

        assert components.issubset(
            h_allowed
        )

        assert (
            "glycogen"
            not in components
        )

        assert (
            2
            <= len(components)
            <= 3
        )

    for info in i_metadata:
        components = set(
            info.get(
                "selected_components",
                [],
            )
        )

        assert (
            "glycogen"
            in components
        )

        assert (
            2
            <= len(components)
            <= 3
        )

    print()
    print(
        "Chemistry-domain membership checks passed."
    )


if __name__ == "__main__":
    main()