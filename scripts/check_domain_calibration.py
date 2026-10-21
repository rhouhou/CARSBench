from CARSBench.api import generate_dataset


batch = generate_dataset(
    num_samples=10,
    domain_name="E_calibration_shift",
    seed=42,
)

for sample in batch.samples:
    info = sample.metadata.parameters[
        "calibration_info"
    ]

    print(
        sample.sample_id,
        "shift:",
        info["shift_cm1"],
        "warp:",
        info["warp_cm1"],
    )