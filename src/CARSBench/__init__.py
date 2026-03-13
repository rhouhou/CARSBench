"""
CARSBench
=========

Benchmark and simulator framework for broadband Coherent Anti-Stokes Raman
Scattering (bCARS).

Main capabilities
-----------------

• Synthetic bCARS spectrum simulation
• Domain-shift benchmark generation
• Raman retrieval evaluation
• Hyperspectral cube simulation

Typical usage
-------------

>>> import carsbench as cb
>>> batch = cb.generate_dataset(100)
>>> cb.plot_sample(batch.samples[0])
"""

__version__ = "0.1.0"

# Public API
from .api import (
    generate_dataset,
    generate_multi_domain_dataset,
    list_domains,
)

# Common dataset structures
from .datasets.schema import (
    SpectrumSample,
    SampleBatch,
)

# Visualization helpers
from .viz.spectra import (
    plot_sample,
    plot_sample_with_latents,
)

# Benchmark utilities
from .benchmark.metrics import (
    rmse,
    mae,
    spectral_angle,
)

__all__ = [
    "generate_dataset",
    "generate_multi_domain_dataset",
    "list_domains",
    "SpectrumSample",
    "SampleBatch",
    "plot_sample",
    "plot_sample_with_latents",
    "rmse",
    "mae",
    "spectral_angle",
]