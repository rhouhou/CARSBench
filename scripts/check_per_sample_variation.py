from CARSBench.api import generate_dataset


batch = generate_dataset(
    num_samples=5,
    domain_name="A_typical",
    seed=42,
)

for sample in batch.samples:
    cfg = sample.metadata.parameters["resolved_config"]

    print(
        sample.sample_id,
        "PSF:",
        cfg["instrument"]["psf_fwhm"],
        "NRB alpha:",
        cfg["nrb"]["alpha"],
        "shot:",
        cfg["noise"]["shot_scale"],
    )