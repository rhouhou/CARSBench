from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from CARSBench.configs.defaults import get_base_defaults
from CARSBench.datasets.schema import SampleBatch
from CARSBench.datasets.simulate import SampleSimulator
from CARSBench.datasets.writer import DatasetWriter
from CARSBench.domains import DomainSampler, build_default_registry


DEFAULT_DOMAINS = [
    "A_typical",
    "B_high_res",
    "C_low_res",
    "D_high_noise",
    "E_calibration_shift",
    "F_window_shift",
    "G_nrb_family_shift",
    "H_biochemical_source",
    "I_biochemical_target",
]


def derive_seed(
    master_seed: int,
    *parts: object,
) -> int:
    """
    Derive a deterministic 32-bit child seed from a master seed
    and an arbitrary sequence of identifying values.

    This makes dataset generation independent of:
    - chunk size
    - domain generation order
    - whether domains are generated together or separately
    """
    payload = "|".join(
        [
            str(master_seed),
            *(str(part) for part in parts),
        ]
    )

    digest = hashlib.blake2b(
        payload.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="little",
        signed=False,
    ) % (2**32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the CARSBench synthetic benchmark dataset."
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/carsbench_v1",
        help="Root directory for the generated dataset.",
    )

    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=5000,
        help="Number of spectra generated for each selected domain.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Number of samples written per NPZ batch.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )

    parser.add_argument(
        "--include-latents",
        action="store_true",
        help=(
            "Save latent arrays such as chi_r, chi_nr, "
            "clean_intensity, envelope, and baseline."
        ),
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

    if args.samples_per_domain <= 0:
        raise ValueError(
            "--samples-per-domain must be greater than zero."
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "--chunk-size must be greater than zero."
        )

    output_root = Path(
        args.output_root
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = build_default_registry()

    available_domains = registry.names()

    for domain_name in args.domains:
        if domain_name not in available_domains:
            raise ValueError(
                f"Unknown domain {domain_name!r}. "
                f"Available domains: {available_domains}"
            )

    # =========================================================
    # Generate each requested domain
    # =========================================================

    for domain_name in args.domains:
        print()
        print(
            f"=== Generating {domain_name} ==="
        )

        domain_cfg = registry.get(
            domain_name
        )

        domain_root = (
            output_root
            / domain_name
        )

        writer = DatasetWriter(
            domain_root
        )

        # -----------------------------------------------------
        # Domain-local simulator and sampler
        # -----------------------------------------------------
        #
        # Their internal random state is not used as the primary
        # sample identity. Explicit deterministic seeds are supplied
        # below for every configuration and simulation realization.
        # -----------------------------------------------------

        domain_seed = derive_seed(
            args.seed,
            domain_name,
            "domain",
        )

        sampler = DomainSampler(
            base_defaults=get_base_defaults(),
            seed=domain_seed,
        )

        simulator = SampleSimulator(
            seed=domain_seed
        )

        # -----------------------------------------------------
        # Resolve the domain-level axis constraints once
        # -----------------------------------------------------

        template_seed = derive_seed(
            args.seed,
            domain_name,
            "axis_template",
        )

        template_spec = sampler.resolve(
            domain=domain_cfg,
            seed=template_seed,
        )

        template_axis = (
            template_spec.resolved.get(
                "axis",
                {},
            )
        )

        if domain_name == "F_window_shift":
            # F deliberately varies spectral coverage between
            # samples, but all samples retain a common array length.
            section_overrides = {
                "axis": {
                    "num_points": template_axis.get(
                        "num_points"
                    ),
                }
            }

            axis_policy = (
                "spectral_window_varies_per_sample;"
                "num_points_fixed_within_domain"
            )

        else:
            # All other diagnostic domains use one common physical
            # spectral grid. This prevents accidental spectral-axis
            # variation from confounding the intended domain shift.
            section_overrides = {
                "axis": template_axis,
            }

            axis_policy = (
                "spectral_axis_fixed_within_domain"
            )

        n_total = (
            args.samples_per_domain
        )

        chunk_size = (
            args.chunk_size
        )

        n_chunks = (
            n_total
            + chunk_size
            - 1
        ) // chunk_size

        all_samples = []

        # =====================================================
        # Generate chunks
        # =====================================================

        for chunk_idx in range(
            n_chunks
        ):
            start = (
                chunk_idx
                * chunk_size
            )

            end = min(
                (chunk_idx + 1)
                * chunk_size,
                n_total,
            )

            print(
                f"  chunk "
                f"{chunk_idx + 1}/{n_chunks}"
                f" | samples "
                f"{start}..{end - 1}"
            )

            chunk_samples = []

            for sample_index in range(
                start,
                end,
            ):
                # ---------------------------------------------
                # Independent deterministic seed streams
                # ---------------------------------------------

                config_seed = derive_seed(
                    args.seed,
                    domain_name,
                    sample_index,
                    "config",
                )

                simulation_seed = derive_seed(
                    args.seed,
                    domain_name,
                    sample_index,
                    "simulation",
                )

                # ---------------------------------------------
                # Resolve stochastic domain parameters
                # ---------------------------------------------

                sample_domain_spec = (
                    sampler.resolve_with_overrides(
                        domain=domain_cfg,
                        section_overrides=section_overrides,
                        seed=config_seed,
                    )
                )

                sample_id = (
                    f"{domain_name}_"
                    f"{sample_index:06d}"
                )

                # ---------------------------------------------
                # Generate spectrum
                # ---------------------------------------------

                sample = (
                    simulator.simulate_sample(
                        domain_spec=sample_domain_spec,
                        sample_id=sample_id,
                        seed=simulation_seed,
                        generator="frequency",
                        include_latents=(
                            args.include_latents
                        ),
                    )
                )

                chunk_samples.append(
                    sample
                )

            batch = SampleBatch(
                chunk_samples
            )

            # ---------------------------------------------
            # Save current chunk
            # ---------------------------------------------

            writer.write_batch_npz(
                batch=batch,
                filename=(
                    f"batch_"
                    f"{chunk_idx:03d}.npz"
                ),
                relative_dir="batches",
                compress=True,
            )

            all_samples.extend(
                chunk_samples
            )

        # =====================================================
        # Save metadata
        # =====================================================

        writer.write_metadata_jsonl(
            samples=all_samples,
            filename="metadata.jsonl",
            relative_dir="metadata",
        )

        # =====================================================
        # Save manifest
        # =====================================================

        writer.write_manifest(
            samples=all_samples,
            filename="manifest.json",
            relative_dir=".",
            extra={
                "domain_name": domain_name,
                "samples_per_domain": n_total,
                "chunk_size": chunk_size,
                "include_latents": bool(
                    args.include_latents
                ),
                "master_seed": args.seed,
                "domain_seed": domain_seed,
                "seed_policy": (
                    "deterministic_seed_derived_from_"
                    "master_seed_domain_name_sample_index_"
                    "and_random_stream"
                ),
                "chunk_invariant_generation": True,
                "domain_order_invariant_generation": True,
                "axis_policy": axis_policy,
                "generator": "frequency",
            },
        )

        print(
            f"Saved: {domain_root}"
        )


# Example:
#
# python scripts/01_generate_full_dataset.py \
#     --samples-per-domain 5000 \
#     --chunk-size 500
#
# Generate only selected domains:
#
# python scripts/01_generate_full_dataset.py \
#     --domains A_typical D_high_noise \
#     --samples-per-domain 1000 \
#     --chunk-size 250


if __name__ == "__main__":
    main()