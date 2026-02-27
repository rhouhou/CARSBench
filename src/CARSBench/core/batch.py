from __future__ import annotations
import numpy as np

from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.output import BatchSimulationOutput, ImageSimulationOutput
from CARSBench.config.enums import CubeOrder

from CARSBench.config.domains import apply_domain_preset
from CARSBench.config.dists import Dist
from CARSBench.core.utils import make_rng

def _freeze_common_params(cfg: SimulationConfig) -> SimulationConfig:
    """
    Freeze axis and instrument distributions so all samples share identical shapes
    in a batch/image.
    """
    rng = make_rng(cfg.seed)

    # Apply preset first (so we freeze preset-specific distributions)
    cfg = apply_domain_preset(cfg)
    cfg.domain_preset = None

    # ---- Freeze AXIS ----
    nu_min = float(cfg.axis.nu_min.sample(rng))
    nu_max = float(cfg.axis.nu_max.sample(rng))
    n_points = int(cfg.axis.n_points.sample(rng))
    shift = float(cfg.axis.shift_cm1.sample(rng))
    warp = float(cfg.axis.warp_cm1.sample(rng))

    cfg.axis.nu_min = Dist("fixed", {"value": nu_min})
    cfg.axis.nu_max = Dist("fixed", {"value": nu_max})
    cfg.axis.n_points = Dist("fixed", {"value": n_points})
    cfg.axis.shift_cm1 = Dist("fixed", {"value": shift})
    cfg.axis.warp_cm1 = Dist("fixed", {"value": warp})

    # ---- Freeze INSTRUMENT (recommended) ----
    fwhm = float(cfg.instrument.fwhm_res_cm1.sample(rng))
    env = float(cfg.instrument.envelope_strength.sample(rng))
    cfg.instrument.fwhm_res_cm1 = Dist("fixed", {"value": fwhm})
    cfg.instrument.envelope_strength = Dist("fixed", {"value": env})

    return cfg

def simulate_batch(cfg: SimulationConfig) -> BatchSimulationOutput:
    """
    Simulate a batch of B independent spectra.
    Uses cfg.batch.batch_size.
    Ensures identical axis shape across batch by freezing axis + instrument once.
    """
    B = int(cfg.batch.batch_size)
    if B < 1:
        raise ValueError("batch_size must be >= 1")

    cfg_frozen = _freeze_common_params(cfg)

    outs = []
    for i in range(B):
        ci = SimulationConfig(**{**cfg_frozen.__dict__})
        ci.seed = int(cfg_frozen.seed) + i
        ci.spatial = None  # batch mode ignores spatial shaping
        ci.domain_preset = None
        outs.append(simulate(ci))

    nu = outs[0].nu_cm1

    chi_r = np.stack([o.chi_r for o in outs], axis=0)
    chi_nrb = np.stack([o.chi_nrb for o in outs], axis=0)
    chi_total = np.stack([o.chi_total for o in outs], axis=0)
    I_true = np.stack([o.I_true for o in outs], axis=0)
    I_instr = np.stack([o.I_instr for o in outs], axis=0)
    I_meas = np.stack([o.I_meas for o in outs], axis=0)

    meta = {
        "batch_size": B,
        "base_seed": cfg_frozen.seed,
        "domain_preset": cfg_frozen.domain_preset,
        "frozen_axis": {
            "nu_min": float(nu.min()),
            "nu_max": float(nu.max()),
            "n_points": int(nu.size),
        },
        "resolved_first": outs[0].meta.get("resolved", {}),
    }

    return BatchSimulationOutput(
        nu_cm1=nu,
        chi_r=chi_r,
        chi_nrb=chi_nrb,
        chi_total=chi_total,
        I_true=I_true,
        I_instr=I_instr,
        I_meas=I_meas,
        meta=meta,
    )

def simulate_image(cfg: SimulationConfig) -> ImageSimulationOutput:
    """
    Simulate an HxW hyperspectral cube by simulating B=H*W spectra then reshaping.
    """
    if cfg.spatial is None:
        raise ValueError("cfg.spatial must be set for simulate_image().")

    H = int(cfg.spatial.height)
    W = int(cfg.spatial.width)
    if H < 1 or W < 1:
        raise ValueError("height/width must be >= 1")

    # Use batch simulation with B=H*W
    ci = SimulationConfig(**{**cfg.__dict__})
    ci.batch.batch_size = H * W
    batch_out = simulate_batch(ci)

    N = batch_out.nu_cm1.size
    order = cfg.spatial.order

    if order == CubeOrder.H_W_N:
        reshape = (H, W, N)
    else:  # N_H_W
        reshape = (N, H, W)

    def reshape_cube(x):
        # x is (B, N)
        if order == CubeOrder.H_W_N:
            return x.reshape(H, W, N)
        else:
            return x.reshape(H, W, N).transpose(2, 0, 1)

    out = ImageSimulationOutput(
        nu_cm1=batch_out.nu_cm1,
        chi_r=reshape_cube(batch_out.chi_r),
        chi_nrb=reshape_cube(batch_out.chi_nrb),
        chi_total=reshape_cube(batch_out.chi_total),
        I_true=reshape_cube(batch_out.I_true),
        I_instr=reshape_cube(batch_out.I_instr),
        I_meas=reshape_cube(batch_out.I_meas),
        height=H,
        width=W,
        order=order.value,
        meta={**batch_out.meta, "height": H, "width": W, "order": order.value},
    )
    return out