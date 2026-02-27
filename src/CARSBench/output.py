from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np

@dataclass
class SimulationOutput:
    nu_cm1: np.ndarray
    chi_r: np.ndarray
    chi_nrb: np.ndarray
    chi_total: np.ndarray
    I_true: np.ndarray
    I_instr: np.ndarray
    I_meas: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class BatchSimulationOutput:
    """
    Batch of spectra. Shapes:
      nu_cm1: (N,)
      chi_r: (B, N) complex
      I_meas: (B, N)
    """
    nu_cm1: np.ndarray
    chi_r: np.ndarray
    chi_nrb: np.ndarray
    chi_total: np.ndarray
    I_true: np.ndarray
    I_instr: np.ndarray
    I_meas: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ImageSimulationOutput:
    """
    Hyperspectral cube. Stored as float/complex arrays in chosen order.

    If order == H_W_N:
      I_meas: (H, W, N)
    If order == N_H_W:
      I_meas: (N, H, W)
    """
    nu_cm1: np.ndarray
    chi_r: np.ndarray
    chi_nrb: np.ndarray
    chi_total: np.ndarray
    I_true: np.ndarray
    I_instr: np.ndarray
    I_meas: np.ndarray
    height: int = 0
    width: int = 0
    order: str = "H_W_N"
    meta: Dict[str, Any] = field(default_factory=dict)