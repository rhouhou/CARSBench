
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
from pathlib import Path

import numpy as np


FINGERPRINT_MIN_CM1 = 400.0
FINGERPRINT_MAX_CM1 = 1800.0


def load_external_generator(repo: Path):
    generator_path = repo / "synthetic-data-generator.py"

    if not generator_path.exists():
        raise FileNotFoundError(
            f"CRIMSON generator not found: {generator_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "crimson_external_generator",
        generator_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not import external generator: {generator_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module, generator_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def generate_valid_samples(
    generator,
    num_samples: int,
    n_points: int,
    seed: int,
    case_a: int,
    case_b: str,
):
    np.random.seed(seed)
    random.seed(seed)

    params = generator.key_parameters(case_a, case_b)

    if params is None:
        raise ValueError(
            f"Undefined CRIMSON parameter case: {case_a}{case_b}"
        )

    (
        min_features,
        max_features,
        min_width,
        max_width,
    ) = params

    step = (
        generator.MAX_WAVENUMBER
        - generator.MIN_WAVENUMBER
    ) / n_points

    wavenumber_axis = np.arange(
        generator.MIN_WAVENUMBER,
        generator.MAX_WAVENUMBER,
        step,
    )

    if len(wavenumber_axis) != n_points:
        raise RuntimeError(
            "Unexpected CRIMSON axis length: "
            f"{len(wavenumber_axis)} != {n_points}"
        )

    nu = np.linspace(0.0, 1.0, n_points)

    bcars = []
    raman = []
    stokes_profiles = []

    attempted = 0
    rejected = 0

    max_attempts = num_samples * 100

    while len(bcars) < num_samples:
        attempted += 1

        if attempted > max_attempts:
            raise RuntimeError(
                "Too many rejected CRIMSON samples. "
                f"Accepted {len(bcars)} of {num_samples} "
                f"after {attempted} attempts."
            )

        c, r, s = generator.generate_bCARS(
            min_features,
            max_features,
            min_width,
            max_width,
            wavenumber_axis,
            nu,
            n_points,
        )

        if c is None or r is None or s is None:
            rejected += 1
            continue

        if not (
            np.all(np.isfinite(c))
            and np.all(np.isfinite(r))
            and np.all(np.isfinite(s))
        ):
            rejected += 1
            continue

        bcars.append(c)
        raman.append(r)
        stokes_profiles.append(s)

    bcars = np.stack(bcars, axis=0)
    raman = np.stack(raman, axis=0)
    stokes_profiles = np.stack(stokes_profiles, axis=0)

    fingerprint_mask = (
        (wavenumber_axis >= FINGERPRINT_MIN_CM1)
        & (wavenumber_axis <= FINGERPRINT_MAX_CM1)
    )

    metadata = {
        "protocol": "CRIMSON_XSIM_v1",
        "external_simulator": "CRIMSON NRB_removal",
        "generator_case": f"{case_a}{case_b}",
        "generator_parameters": {
            "min_features": int(min_features),
            "max_features": int(max_features),
            "min_width_cm1": float(min_width),
            "max_width_cm1": float(max_width),
        },
        "random_seed_numpy": int(seed),
        "random_seed_python": int(seed),
        "requested_valid_samples": int(num_samples),
        "accepted_samples": int(len(bcars)),
        "attempted_samples": int(attempted),
        "rejected_samples": int(rejected),
        "acceptance_rate": float(len(bcars) / attempted),
        "n_points": int(n_points),
        "axis_min_cm1": float(wavenumber_axis[0]),
        "axis_max_cm1": float(wavenumber_axis[-1]),
        "axis_step_cm1": float(step),
        "fingerprint_min_cm1": FINGERPRINT_MIN_CM1,
        "fingerprint_max_cm1": FINGERPRINT_MAX_CM1,
        "fingerprint_points": int(fingerprint_mask.sum()),
        "notes": [
            "Samples are generated with CRIMSON generate_bCARS().",
            "CRIMSON generate_batch() is intentionally not used because "
            "rejected samples can leave uninitialized np.empty rows.",
            "The third output of generate_bCARS() is I_S, the Stokes "
            "intensity profile; it is stored here as stokes_profile.",
            "No CRIMSON data are used for CARSBench training, validation, "
            "normalization fitting, or model selection.",
        ],
    }

    return (
        wavenumber_axis,
        fingerprint_mask,
        bcars,
        raman,
        stokes_profiles,
        metadata,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--external-repo",
        type=Path,
        default=Path("/content/CRIMSON_NRB_external"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/cross_simulator/crimson"
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--n-points",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--case-a",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--case-b",
        type=str,
        default="b",
    )

    args = parser.parse_args()

    repo = args.external_repo.resolve()

    generator, generator_path = load_external_generator(repo)

    print("=" * 70)
    print("CRIMSON CROSS-SIMULATOR DATA GENERATION")
    print("=" * 70)

    print("External repository:", repo)
    print("Generator:", generator_path)
    print("External commit:", git_commit(repo))
    print("Generator SHA256:", sha256_file(generator_path))

    (
        axis,
        fingerprint_mask,
        bcars,
        raman,
        stokes,
        metadata,
    ) = generate_valid_samples(
        generator=generator,
        num_samples=args.num_samples,
        n_points=args.n_points,
        seed=args.seed,
        case_a=args.case_a,
        case_b=args.case_b,
    )

    metadata["external_git_commit"] = git_commit(repo)
    metadata["generator_sha256"] = sha256_file(generator_path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = (
        f"crimson_case{args.case_a}{args.case_b}"
        f"_n{args.num_samples}"
        f"_seed{args.seed}"
    )

    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}_metadata.json"

    np.savez_compressed(
        npz_path,
        wavenumber_cm1=axis,
        fingerprint_mask=fingerprint_mask,
        bcars=bcars,
        raman=raman,
        stokes_profile=stokes,
    )

    with json_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    fp = fingerprint_mask

    print()
    print("Generation complete")
    print("-------------------")
    print("Accepted:", metadata["accepted_samples"])
    print("Rejected:", metadata["rejected_samples"])
    print("Attempts:", metadata["attempted_samples"])
    print(
        "Acceptance rate:",
        f"{100 * metadata['acceptance_rate']:.2f}%",
    )

    print()
    print("Full axis:")
    print(
        f"  {axis[0]:.4f} – {axis[-1]:.4f} cm^-1"
    )
    print("  points:", len(axis))
    print("  spacing:", axis[1] - axis[0])

    print()
    print("Matched fingerprint region:")
    print(
        f"  {axis[fp][0]:.4f} – "
        f"{axis[fp][-1]:.4f} cm^-1"
    )
    print("  points:", int(fp.sum()))

    print()
    print("Arrays:")
    print("  BCARS:", bcars.shape, bcars.dtype)
    print("  Raman:", raman.shape, raman.dtype)
    print("  Stokes:", stokes.shape, stokes.dtype)

    print()
    print("Finite checks:")
    print("  BCARS:", np.isfinite(bcars).all())
    print("  Raman:", np.isfinite(raman).all())
    print("  Stokes:", np.isfinite(stokes).all())

    print()
    print("Ranges:")
    print(
        "  BCARS:",
        float(np.min(bcars)),
        float(np.max(bcars)),
    )
    print(
        "  Raman:",
        float(np.min(raman)),
        float(np.max(raman)),
    )

    print()
    print("Saved:")
    print(" ", npz_path)
    print(" ", json_path)


if __name__ == "__main__":
    main()
