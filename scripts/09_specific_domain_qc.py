from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SEEDS = [42, 123, 777]


def get_nested(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def load_metadata_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return np.nan, np.nan
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def summarize_D(rows: list[dict[str, Any]]) -> dict:
    shift = []
    warp = []
    for row in rows:
        v1 = get_nested(row, "parameters.calibration_info.shift_cm1")
        v2 = get_nested(row, "parameters.calibration_info.warp_cm1")
        if v1 is not None:
            shift.append(float(v1))
        if v2 is not None:
            warp.append(float(v2))

    shift_mean, shift_std = mean_std(shift)
    warp_mean, warp_std = mean_std(warp)

    return {
        "shift_mean": shift_mean,
        "shift_std": shift_std,
        "warp_mean": warp_mean,
        "warp_std": warp_std,
    }


def summarize_E(rows: list[dict[str, Any]]) -> dict:
    nu_min = []
    nu_max = []
    window_mode = Counter()

    for row in rows:
        vmin = get_nested(row, "parameters.resolved_config.axis.nu_min")
        vmax = get_nested(row, "parameters.resolved_config.axis.nu_max")
        w = get_nested(row, "parameters.resolved_config.axis.window_mode")

        if vmin is not None:
            nu_min.append(float(vmin))
        if vmax is not None:
            nu_max.append(float(vmax))
        if w is not None:
            window_mode[str(w)] += 1

    nu_min_mean, nu_min_std = mean_std(nu_min)
    nu_max_mean, nu_max_std = mean_std(nu_max)

    out = {
        "nu_min_mean": nu_min_mean,
        "nu_min_std": nu_min_std,
        "nu_max_mean": nu_max_mean,
        "nu_max_std": nu_max_std,
    }

    for k, v in sorted(window_mode.items()):
        out[f"window_mode_count_{k}"] = int(v)

    return out


def summarize_F(rows: list[dict[str, Any]]) -> dict:
    alpha = []
    phase_total_change = []
    family = Counter()
    phase_model = Counter()

    for row in rows:
        a = get_nested(row, "parameters.nrb_info.alpha")
        p = get_nested(row, "parameters.nrb_info.phase_total_change")
        f = get_nested(row, "parameters.nrb_info.family")
        pm = get_nested(row, "parameters.nrb_info.phase_model")

        if a is not None:
            alpha.append(float(a))
        if p is not None:
            phase_total_change.append(float(p))
        if f is not None:
            family[str(f)] += 1
        if pm is not None:
            phase_model[str(pm)] += 1

    alpha_mean, alpha_std = mean_std(alpha)
    phase_mean, phase_std = mean_std(phase_total_change)

    out = {
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "phase_total_change_mean": phase_mean,
        "phase_total_change_std": phase_std,
    }

    for k, v in sorted(family.items()):
        out[f"family_count_{k}"] = int(v)
    for k, v in sorted(phase_model.items()):
        out[f"phase_model_count_{k}"] = int(v)

    return out


def summarize_GH(rows: list[dict[str, Any]]) -> dict:
    selected_components = Counter()
    peak_sources = Counter()
    peak_centers = []
    num_peaks = []

    for row in rows:
        comps = get_nested(row, "parameters.resonant_info.selected_components")
        if comps is not None:
            for c in comps:
                selected_components[str(c)] += 1

        srcs = get_nested(row, "parameters.resonant_info.peak_sources")
        if srcs is not None:
            for s in srcs:
                peak_sources[str(s)] += 1

        centers = get_nested(row, "parameters.resonant_info.peak_centers")
        if centers is not None:
            for c in centers:
                try:
                    peak_centers.append(float(c))
                except (TypeError, ValueError):
                    pass

        npk = get_nested(row, "parameters.resonant_info.num_peaks")
        if npk is not None:
            num_peaks.append(int(npk))

    centers_mean, centers_std = mean_std(peak_centers)
    num_peaks_mean, num_peaks_std = mean_std(num_peaks)

    out = {
        "peak_centers_mean": centers_mean,
        "peak_centers_std": centers_std,
        "num_peaks_mean": num_peaks_mean,
        "num_peaks_std": num_peaks_std,
    }

    for k, v in sorted(selected_components.items()):
        out[f"selected_component_count_{k}"] = int(v)
    for k, v in sorted(peak_sources.items()):
        out[f"peak_source_count_{k}"] = int(v)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="qc_specific.csv")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    rows_out = []

    domains = {
        "D_calibration_shift": summarize_D,
        "E_window_shift": summarize_E,
        "F_nrb_family_shift": summarize_F,
        "G_biochemical_source": summarize_GH,
        "H_biochemical_target": summarize_GH,
    }

    for seed in args.seeds:
        for domain, fn in domains.items():
            meta_path = (
                data_root / f"seed_{seed}" / domain / "metadata" / "metadata.jsonl"
            )
            if not meta_path.exists():
                print(f"Missing: {meta_path}")
                continue

            rows = load_metadata_jsonl(meta_path)
            stats = fn(rows)

            row = {
                "seed": seed,
                "domain": domain,
            }
            row.update(stats)
            rows_out.append(row)

    if not rows_out:
        print("No rows to save.")
        return

    all_keys = set()
    for r in rows_out:
        all_keys.update(r.keys())
    fieldnames = ["seed", "domain"] + sorted(
        k for k in all_keys if k not in {"seed", "domain"}
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
