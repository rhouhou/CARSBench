from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SEEDS = [42, 123, 777]
OUTPUT_ROOT = "/Volumes/mohamadhjeij1982/Biophotonics-AI/Project-BCARSBench/carsbench_data"

SAMPLES_PER_DOMAIN = 5000
CHUNK_SIZE = 500


def main() -> None:
    python_exe = sys.executable

    scripts_dir = Path(__file__).resolve().parent
    generator_script = scripts_dir / "01_generate_full_dataset.py"

    if not generator_script.exists():
        raise FileNotFoundError(
            f"Generator script not found: {generator_script}"
        )

    for seed in SEEDS:
        print(f"\n=== Generating dataset for seed {seed} ===")

        subprocess.run(
            [
                python_exe,
                str(generator_script),
                "--output-root",
                f"{OUTPUT_ROOT}/seed_{seed}",
                "--samples-per-domain",
                str(SAMPLES_PER_DOMAIN),
                "--chunk-size",
                str(CHUNK_SIZE),
                "--seed",
                str(seed),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()