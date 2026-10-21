import numpy as np

from CARSBench.api import generate_dataset


for domain_name in [
    "H_biochemical_source",
    "I_biochemical_target",
]:
    batch = generate_dataset(
        num_samples=100,
        domain_name=domain_name,
        seed=42,
    )

    counts = []

    for sample in batch.samples:
        info = sample.metadata.parameters[
            "resonant_info"
        ]

        if "num_peaks" in info:
            counts.append(
                info["num_peaks"]
            )

    counts = np.asarray(
        counts,
        dtype=float,
    )

    print()
    print(domain_name)

    if len(counts) == 0:
        print("No peak-count metadata found.")
    else:
        print("Minimum:", counts.min())
        print("Median:", np.median(counts))
        print("Mean:", counts.mean())
        print("Maximum:", counts.max())