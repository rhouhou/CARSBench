from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

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
    "G_system_response_shift",
    "H_biochemical_source",
    "I_biochemical_target",
    "J_compound_acquisition",
    "K_joint_ood",
]


def derive_seed(
    master_seed: int,
    *parts: object,
) -> int:
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


def json_default(
    value: Any,
) -> Any:
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    raise TypeError(
        f"Object of type {type(value).__name__} "
        f"is not JSON serializable."
    )


def metadata_to_dict(
    sample,
) -> dict[str, Any]:
    return asdict(
        sample.metadata
    )


def append_metadata_jsonl(
    path: Path,
    samples,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        for sample in samples:
            row = metadata_to_dict(
                sample
            )

            handle.write(
                json.dumps(
                    row,
                    default=json_default,
                    sort_keys=True,
                )
            )

            handle.write("\n")


def write_generation_manifest(
    path: Path,
    *,
    domain_name: str,
    start_index: int,
    samples_per_domain: int,
    chunk_size: int,
    n_chunks: int,
    include_latents: bool,
    master_seed: int,
    domain_seed: int,
    physical_axis_policy: str,
    observation_policy: str,
) -> None:
    end_index_exclusive = (
        start_index
        + samples_per_domain
    )

    manifest = {
        "format": "CARSBench",
        "benchmark_version": "1.0.0",
        "domain_name": domain_name,
        "generator": "frequency",

        "num_samples": samples_per_domain,

        "start_index": start_index,
        "end_index_exclusive": end_index_exclusive,

        "chunk_size": chunk_size,
        "n_chunks": n_chunks,

        "include_latents": include_latents,

        "master_seed": master_seed,
        "domain_seed": domain_seed,

        "seed_policy": (
            "deterministic_seed_derived_from_"
            "master_seed_domain_name_sample_index_"
            "and_random_stream"
        ),

        "chunk_invariant_generation": True,
        "domain_order_invariant_generation": True,

        "physical_axis_policy": (
            physical_axis_policy
        ),

        "observation_policy": (
            observation_policy
        ),

        "observation_representation": (
            "canonical_grid_plus_binary_mask"
        ),

        "metadata_format": "jsonl",
        "metadata_streamed": True,

        "memory_policy": (
            "memory_scales_with_chunk_size_not_domain_size"
        ),
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CARSBench synthetic spectra."
        )
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="data/carsbench_v1",
    )

    parser.add_argument(
        "--samples-per-domain",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Global benchmark sample index assigned to the first "
            "generated sample."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--include-latents",
        action="store_true",
    )

    parser.add_argument(
        "--domains",
        nargs="*",
        default=DEFAULT_DOMAINS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete existing selected domain output before "
            "regeneration."
        ),
    )

    return parser.parse_args()


def prepare_domain_directory(
    domain_root: Path,
    *,
    overwrite: bool,
) -> None:
    if domain_root.exists():
        has_contents = any(
            domain_root.iterdir()
        )

        if has_contents:
            if not overwrite:
                raise FileExistsError(
                    f"Output domain already exists and is non-empty: "
                    f"{domain_root}\n"
                    f"Use --overwrite to regenerate this domain."
                )

            shutil.rmtree(
                domain_root
            )

    domain_root.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    args = parse_args()

    if args.samples_per_domain <= 0:
        raise ValueError(
            "--samples-per-domain must be greater than zero."
        )

    if args.start_index < 0:
        raise ValueError(
            "--start-index must be greater than or equal to zero."
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

    for domain_name in args.domains:
        print()
        print(
            f"=== Generating {domain_name} ==="
        )

        print(
            f"index range: "
            f"[{args.start_index}, "
            f"{args.start_index + args.samples_per_domain})"
        )

        domain_cfg = registry.get(
            domain_name
        )

        domain_root = (
            output_root
            / domain_name
        )

        prepare_domain_directory(
            domain_root,
            overwrite=args.overwrite,
        )

        writer = DatasetWriter(
            domain_root
        )

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

        # =====================================================
        # Fixed canonical physical axis
        # =====================================================

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

        required_axis_keys = {
            "nu_min",
            "nu_max",
            "num_points",
        }

        missing_axis_keys = (
            required_axis_keys
            - set(
                template_axis.keys()
            )
        )

        if missing_axis_keys:
            raise ValueError(
                "Resolved axis configuration is missing "
                f"required keys: "
                f"{sorted(missing_axis_keys)}"
            )

        section_overrides = {
            "axis": template_axis,
        }

        physical_axis_policy = (
            "physical_spectral_axis_fixed_within_domain"
        )

        if domain_name == "F_window_shift":
            observation_policy = (
                "observation_window_varies_per_"
                "sample_on_fixed_canonical_axis"
            )
        else:
            observation_policy = (
                "full_observation_on_fixed_canonical_axis"
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

        metadata_path = (
            domain_root
            / "metadata"
            / "metadata.jsonl"
        )

        # =====================================================
        # Generate chunks
        # =====================================================

        for chunk_idx in range(
            n_chunks
        ):
            local_start = (
                chunk_idx
                * chunk_size
            )

            local_end = min(
                (
                    chunk_idx
                    + 1
                )
                * chunk_size,
                n_total,
            )

            global_start = (
                args.start_index
                + local_start
            )

            global_end = (
                args.start_index
                + local_end
            )

            print(
                f"  chunk "
                f"{chunk_idx + 1}/{n_chunks}"
                f" | global samples "
                f"{global_start}..{global_end - 1}"
            )

            chunk_samples = []

            for local_index in range(
                local_start,
                local_end,
            ):
                sample_index = (
                    args.start_index
                    + local_index
                )

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

                sample_domain_spec = (
                    sampler.resolve_with_overrides(
                        domain=domain_cfg,
                        section_overrides=(
                            section_overrides
                        ),
                        seed=config_seed,
                    )
                )

                sample_id = (
                    f"{domain_name}_"
                    f"{sample_index:06d}"
                )

                sample = (
                    simulator.simulate_sample(
                        domain_spec=(
                            sample_domain_spec
                        ),
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

            writer.write_batch_npz(
                batch=batch,
                filename=(
                    f"batch_"
                    f"{chunk_idx:03d}.npz"
                ),
                relative_dir="batches",
                compress=True,
            )

            append_metadata_jsonl(
                metadata_path,
                chunk_samples,
            )

            del batch
            del chunk_samples

        # =====================================================
        # Manifest
        # =====================================================

        write_generation_manifest(
            domain_root
            / "manifest.json",

            domain_name=domain_name,

            start_index=(
                args.start_index
            ),

            samples_per_domain=(
                n_total
            ),

            chunk_size=(
                chunk_size
            ),

            n_chunks=(
                n_chunks
            ),

            include_latents=bool(
                args.include_latents
            ),

            master_seed=(
                args.seed
            ),

            domain_seed=(
                domain_seed
            ),

            physical_axis_policy=(
                physical_axis_policy
            ),

            observation_policy=(
                observation_policy
            ),
        )

        print(
            f"Saved: {domain_root}"
        )


if __name__ == "__main__":
    main()