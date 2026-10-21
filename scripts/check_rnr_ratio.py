import numpy as np

from CARSBench.api import generate_dataset


batch = generate_dataset(
    num_samples=100,
    domain_name="A_typical",
    seed=42,
)

ratios = np.array(
    [
        sample.metadata.parameters[
            "resonant_info"
        ]["realized_rnr_ratio"]
        for sample in batch.samples
    ]
)

print("N:", len(ratios))
print("Min:", ratios.min())
print("Max:", ratios.max())
print("Mean:", ratios.mean())
print("Median:", np.median(ratios))
print("Std:", ratios.std())