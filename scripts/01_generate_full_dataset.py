from __future__ import annotations

import argparse
from pathlib import Path

from CARSBench.configs.defaults import get_base_defaults
from CARSBench.domains import build_default_registry, DomainSampler
from CARSBench.datasets.batch import BatchSimulator
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.datasets.writer import DatasetWriter


DEFAULT_DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res_noisy",
    "D_calibration_shift",
    "E_window_shift",
    "F_nrb_family_shift",
    "G_biochemical_source",
    "H_biochemical_target",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, default="data/carsbench_v1")
    parser.add_argument("--samples-per-domain", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-latents",
        action="store_true",
        help="Save latent arrays such as chi_r, chi_nr, clean_intensity.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=DEFAULT_DOMAINS,
        help="Domains to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    registry = build_default_registry()
    sampler = DomainSampler(
        base_defaults=get_base_defaults(),
        seed=args.seed,
    )
    simulator = SampleSimulator(seed=args.seed)
    batch_sim = BatchSimulator(simulator)

    for domain_name in args.domains:
        if domain_name not in registry.names():
            raise ValueError(f"Unknown domain: {domain_name!r}")

        print(f"\n=== Generating {domain_name} ===")

        domain_cfg = registry.get(domain_name)
        domain_spec = sampler.resolve(domain_cfg, seed=args.seed)

        domain_root = output_root / domain_name
        writer = DatasetWriter(domain_root)

        n_total = args.samples_per_domain
        chunk_size = args.chunk_size
        n_chunks = (n_total + chunk_size - 1) // chunk_size

        all_samples = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_total)
            n_this = end - start

            print(f"  chunk {chunk_idx+1}/{n_chunks} | samples {start}..{end-1}")

            batch = batch_sim.simulate_from_domain(
                domain_spec=domain_spec,
                num_samples=n_this,
                id_prefix=domain_name,
                start_index=start,
                include_latents=args.include_latents,
                generator="frequency",
            )

            writer.write_batch_npz(
                batch=batch,
                filename=f"batch_{chunk_idx:03d}.npz",
                relative_dir="batches",
                compress=True,
            )

            all_samples.extend(batch.samples)

        writer.write_metadata_jsonl(
            samples=all_samples,
            filename="metadata.jsonl",
            relative_dir="metadata",
        )

        writer.write_manifest(
            samples=all_samples,
            filename="manifest.json",
            relative_dir=".",
            extra={
                "domain_name": domain_name,
                "samples_per_domain": n_total,
                "chunk_size": chunk_size,
                "include_latents": bool(args.include_latents),
                "master_seed": args.seed,
            },
        )

        print(f"Saved: {domain_root}")

# Example Usage
# python scripts/generate_full_dataset.py --samples-per-domain 5000 --chunk-size 500

if __name__ == "__main__":
    main()