from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path("data/invariance_check")

RUN_A = ROOT / "chunk_4"
RUN_B = ROOT / "chunk_7"
RUN_C = ROOT / "domain_subset"


def run_generation(
    output_root: Path,
    chunk_size: int,
    domains: list[str],
) -> None:
    cmd = [
        "python",
        "scripts/01_generate_full_dataset.py",
        "--samples-per-domain",
        "10",
        "--chunk-size",
        str(chunk_size),
        "--output-root",
        str(output_root),
        "--domains",
        *domains,
    ]

    print()
    print("Running:")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        check=True,
    )


def load_domain_samples(
    root: Path,
    domain_name: str,
) -> dict[int, dict[str, np.ndarray]]:
    """
    Load all NPZ chunks for one domain and reconstruct samples
    using their global sequential index.

    This deliberately does not depend on sample_ids being stored
    inside the NPZ files.
    """
    domain_root = (
        root
        / domain_name
        / "batches"
    )

    files = sorted(
        domain_root.glob(
            "batch_*.npz"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No batch files found in {domain_root}"
        )

    samples = {}

    global_index = 0

    for file in files:
        with np.load(
            file,
            allow_pickle=True,
        ) as data:

            axis = data["axis"]
            spectrum = data["spectrum"]
            raman_target = data[
                "raman_target"
            ]

            n_samples = spectrum.shape[0]

            for i in range(n_samples):
                samples[global_index] = {
                    "axis": axis[i],
                    "spectrum": spectrum[i],
                    "raman_target": (
                        raman_target[i]
                    ),
                }

                global_index += 1

    return samples


def compare_runs(
    root_1: Path,
    root_2: Path,
    domain_name: str,
) -> None:
    samples_1 = load_domain_samples(
        root_1,
        domain_name,
    )

    samples_2 = load_domain_samples(
        root_2,
        domain_name,
    )

    ids_1 = set(samples_1)
    ids_2 = set(samples_2)

    assert ids_1 == ids_2

    for sample_id in sorted(ids_1):
        for key in [
            "axis",
            "spectrum",
            "raman_target",
        ]:
            np.testing.assert_array_equal(
                samples_1[sample_id][key],
                samples_2[sample_id][key],
            )

    print(
        f"PASS: {domain_name} identical between "
        f"{root_1} and {root_2}"
    )


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)

    # ---------------------------------------------------------
    # Same domains, different chunk sizes
    # ---------------------------------------------------------

    run_generation(
        output_root=RUN_A,
        chunk_size=4,
        domains=[
            "A_typical",
            "D_high_noise",
        ],
    )

    run_generation(
        output_root=RUN_B,
        chunk_size=7,
        domains=[
            "A_typical",
            "D_high_noise",
        ],
    )

    compare_runs(
        RUN_A,
        RUN_B,
        "A_typical",
    )

    compare_runs(
        RUN_A,
        RUN_B,
        "D_high_noise",
    )

    # ---------------------------------------------------------
    # Generate only A_typical
    # ---------------------------------------------------------

    run_generation(
        output_root=RUN_C,
        chunk_size=5,
        domains=[
            "A_typical",
        ],
    )

    compare_runs(
        RUN_A,
        RUN_C,
        "A_typical",
    )

    print()
    print(
        "All reproducibility invariance checks passed."
    )


if __name__ == "__main__":
    main()