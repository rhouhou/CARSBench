from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

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

DEFAULT_SEEDS = [42, 123, 777]


def canonical_axis() -> np.ndarray:
    return np.linspace(400.0, 3200.0, 1024, dtype=np.float64)


def interpolate_rows(
    axis_2d: np.ndarray, y_2d: np.ndarray, target_axis: np.ndarray
) -> np.ndarray:
    out = np.empty((y_2d.shape[0], target_axis.size), dtype=np.float32)

    for i in range(y_2d.shape[0]):
        axis = np.asarray(axis_2d[i], dtype=np.float64)
        y = np.asarray(y_2d[i], dtype=np.float64)

        order = np.argsort(axis)
        axis = axis[order]
        y = y[order]

        out[i] = np.interp(target_axis, axis, y).astype(np.float32)

    return out


def load_domain_matrix(
    data_root: Path,
    seed: int,
    domain: str,
    value_key: str,
    max_batches: int | None = None,
) -> np.ndarray:
    batch_dir = data_root / f"seed_{seed}" / domain / "batches"
    batch_files = sorted(batch_dir.glob("*.npz"))
    if max_batches is not None:
        batch_files = batch_files[:max_batches]

    target_axis = canonical_axis()
    chunks = []

    for bf in batch_files:
        with np.load(bf, allow_pickle=False) as data:
            if "axis" not in data or value_key not in data:
                continue

            Y = interpolate_rows(data["axis"], data[value_key], target_axis)
            chunks.append(Y)

    if not chunks:
        return np.empty((0, target_axis.size), dtype=np.float32)

    return np.concatenate(chunks, axis=0)


def region_mean(Y: np.ndarray, axis: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (axis >= lo) & (axis <= hi)
    if not np.any(mask):
        return np.full(Y.shape[0], np.nan, dtype=np.float64)
    return Y[:, mask].mean(axis=1)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def pooled_effect_size(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan

    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1)
    if pooled <= 0:
        return np.nan
    return float((mx - my) / np.sqrt(pooled))


def sharpness_metric(Y: np.ndarray) -> float:
    if len(Y) == 0:
        return np.nan
    d = np.abs(np.diff(Y, axis=1))
    return float(np.mean(d))


def summarize_domain_vs_reference(
    Y_ref: np.ndarray,
    Y_dom: np.ndarray,
    axis: np.ndarray,
) -> dict:
    if len(Y_ref) == 0 or len(Y_dom) == 0:
        return {}

    mean_ref = Y_ref.mean(axis=0)
    mean_dom = Y_dom.mean(axis=0)

    std_ref = Y_ref.std(axis=0)
    std_dom = Y_dom.std(axis=0)

    fp_ref = region_mean(Y_ref, axis, 700.0, 1800.0)
    fp_dom = region_mean(Y_dom, axis, 700.0, 1800.0)

    ch_ref = region_mean(Y_ref, axis, 2800.0, 3050.0)
    ch_dom = region_mean(Y_dom, axis, 2800.0, 3050.0)

    return {
        "n_ref": int(len(Y_ref)),
        "n_dom": int(len(Y_dom)),
        "mean_curve_l2": float(np.linalg.norm(mean_ref - mean_dom)),
        "mean_curve_cosine": cosine_similarity(mean_ref, mean_dom),
        "mean_curve_corr": float(np.corrcoef(mean_ref, mean_dom)[0, 1]),
        "fp_effect_size": pooled_effect_size(fp_ref, fp_dom),
        "ch_effect_size": pooled_effect_size(ch_ref, ch_dom),
        "ref_mean_std": float(np.mean(std_ref)),
        "dom_mean_std": float(np.mean(std_dom)),
        "ref_max_std": float(np.max(std_ref)),
        "dom_max_std": float(np.max(std_dom)),
        "ref_sharpness": sharpness_metric(Y_ref),
        "dom_sharpness": sharpness_metric(Y_dom),
        "sharpness_ratio_dom_over_ref": (
            float(sharpness_metric(Y_dom) / sharpness_metric(Y_ref))
            if sharpness_metric(Y_ref) not in [0, np.nan]
            else np.nan
        ),
        "fp_mean_ref": float(np.nanmean(fp_ref)),
        "fp_mean_dom": float(np.nanmean(fp_dom)),
        "ch_mean_ref": float(np.nanmean(ch_ref)),
        "ch_mean_dom": float(np.nanmean(ch_dom)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="qc_general.csv")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    parser.add_argument("--reference-domain", type=str, default="A_typical")
    parser.add_argument(
        "--value-key",
        type=str,
        default="spectrum",
        choices=["spectrum", "clean_intensity", "raman_target"],
    )
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_csv = Path(args.output_csv)
    axis = canonical_axis()

    rows = []

    for seed in args.seeds:
        Y_ref = load_domain_matrix(
            data_root=data_root,
            seed=seed,
            domain=args.reference_domain,
            value_key=args.value_key,
            max_batches=args.max_batches,
        )

        for domain in args.domains:
            Y_dom = load_domain_matrix(
                data_root=data_root,
                seed=seed,
                domain=domain,
                value_key=args.value_key,
                max_batches=args.max_batches,
            )

            if len(Y_ref) == 0 or len(Y_dom) == 0:
                print(f"Skipping seed={seed}, domain={domain}")
                continue

            stats = summarize_domain_vs_reference(Y_ref, Y_dom, axis)
            row = {
                "seed": seed,
                "reference_domain": args.reference_domain,
                "domain": domain,
                "value_key": args.value_key,
            }
            row.update(stats)
            rows.append(row)

    if not rows:
        print("No rows to save.")
        return

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
