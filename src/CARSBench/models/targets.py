from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

from CARSBench.output import SimulationOutput


class TargetKind(str, Enum):
    """
    Defines common supervised targets for NRB-removal / Raman retrieval.
    """
    IM_CHI_R = "im_chi_r"              # Raman-like
    RE_CHI_R = "re_chi_r"
    CHI_R_COMPLEX = "chi_r_complex"    # (2, N): [Re, Im]
    CHI_TOTAL_COMPLEX = "chi_total_complex"
    I_TRUE = "i_true"
    I_MEAS = "i_meas"


def make_target(out: SimulationOutput, kind: TargetKind) -> np.ndarray:
    """
    Returns a numpy array target for ML training.

    Notes:
    - For complex targets we return stacked real/imag with shape (2, N)
      so it’s easy to feed into standard models.
    """
    k = TargetKind(kind)

    if k == TargetKind.IM_CHI_R:
        return out.chi_r.imag.astype(np.float32)

    if k == TargetKind.RE_CHI_R:
        return out.chi_r.real.astype(np.float32)

    if k == TargetKind.CHI_R_COMPLEX:
        return np.stack([out.chi_r.real, out.chi_r.imag], axis=0).astype(np.float32)

    if k == TargetKind.CHI_TOTAL_COMPLEX:
        return np.stack([out.chi_total.real, out.chi_total.imag], axis=0).astype(np.float32)

    if k == TargetKind.I_TRUE:
        return out.I_true.astype(np.float32)

    if k == TargetKind.I_MEAS:
        return out.I_meas.astype(np.float32)

    raise ValueError(f"Unknown target kind: {kind}")


# Optional improvement note:
# If you want more realistic resonant variability, modify generate_chi_r()
# to sample per-peak amplitudes and widths, not a single value for all peaks.