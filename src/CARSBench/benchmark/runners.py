from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List
import numpy as np

from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.models.targets import TargetKind, make_target
from CARSBench.benchmark.metrics import rmse, mae, sam
from CARSBench.benchmark.protocol import DomainSplit, lodo_splits
from CARSBench.core.batch import simulate_batch, simulate_image
from CARSBench.config.spatial import SpatialConfig, BatchConfig
from CARSBench.config.enums import CubeOrder

Reconstructor = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]
# signature: reconstructor(I_meas, context) -> y_pred target array


@dataclass
class BenchmarkConfig:
    domains: List[str]  # e.g. ["typical","noisy","high_res","calibration_shifted"]
    n_train_per_domain: int = 50
    n_test_per_domain: int = 50
    target: TargetKind = TargetKind.IM_CHI_R
    seed: int = 0
    mode: str = "spectra"  # "spectra" | "batch" | "image"
    batch_size: int = 32
    height: int = 32
    width: int = 32


def _simulate_dataset(domain: str, n: int, base_seed: int, mode: str, batch_size: int, height: int, width: int):
    outs = []
    for i in range(n):
        cfg = SimulationConfig(seed=base_seed + i, domain_preset=domain)
        if mode == "spectra":
            outs.append(simulate(cfg))
        elif mode == "batch":
            cfg.batch = BatchConfig(batch_size=batch_size)
            outs.append(simulate_batch(cfg))
        elif mode == "image":
            cfg.spatial = SpatialConfig(height=height, width=width, order=CubeOrder.H_W_N)
            cfg.batch = BatchConfig(batch_size=height * width)
            outs.append(simulate_image(cfg))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return outs


def run_benchmark(
    reconstructor: Reconstructor,
    bench_cfg: BenchmarkConfig,
    reconstructor_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Runs LODO benchmark on simulated domains.

    reconstructor: function that takes I_meas and context and returns y_pred (same shape as target).
    For classical baselines, reconstructor could ignore context.
    """
    params = reconstructor_params or {}
    splits = lodo_splits(bench_cfg.domains)

    results = {
        "target": bench_cfg.target.value,
        "splits": [],
    }

    # Pre-generate datasets per domain for reproducibility + speed
    datasets = {}
    for d in bench_cfg.domains:
        datasets[d] = {
            "train": _simulate_dataset(d, bench_cfg.n_train_per_domain, base_seed=1000 + hash((bench_cfg.seed, d)) % 100000),
            "test": _simulate_dataset(d, bench_cfg.n_test_per_domain, base_seed=2000 + hash((bench_cfg.seed, d)) % 100000),
        }

    for split in splits:
        # In this skeleton, reconstructor is not trained.
        # In the future you can add training here using split.train_domains datasets.
        # For now: evaluate directly on test domains.
        split_res = {
            "train_domains": split.train_domains,
            "test_domains": split.test_domains,
            "per_domain": {},
        }

        for d in split.test_domains:
            outs = datasets[d]["test"]
            y_true_list = []
            y_pred_list = []

            for out in outs:
                y_true = make_target(out, bench_cfg.target)
                context = {"nu_cm1": out.nu_cm1, "meta": out.meta, "resolved": out.meta.get("resolved", {})}
                y_pred = reconstructor(out.I_meas, {**context, **params})

                y_true_list.append(y_true)
                y_pred_list.append(y_pred)

            # stack and compute metrics
            y_true_all = np.stack(y_true_list, axis=0)
            y_pred_all = np.stack(y_pred_list, axis=0)

            # compute metrics on mean spectrum if 1D targets; if (2,N), flatten safely
            split_res["per_domain"][d] = {
                "rmse": rmse(y_true_all, y_pred_all),
                "mae": mae(y_true_all, y_pred_all),
                "sam": sam(y_true_all, y_pred_all),
            }

        results["splits"].append(split_res)

    return results