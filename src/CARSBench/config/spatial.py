from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .enums import CubeOrder

@dataclass
class BatchConfig:
    """
    Batch of independent spectra. Great for dataset generation.
    """
    batch_size: int = 1

@dataclass
class SpatialConfig:
    """
    Image/hyperspectral cube settings.
    Internally we simulate B=H*W spectra and reshape.
    """
    height: int = 64
    width: int = 64
    order: CubeOrder = CubeOrder.H_W_N

    # optional: later you can add spatial variability maps
    # e.g., vary alpha_nrb or peak amplitudes across pixels
    enable_spatial_maps: bool = False
    # placeholders for future expansion:
    # nrb_scale_map: Optional[str] = None
    # resonant_scale_map: Optional[str] = None