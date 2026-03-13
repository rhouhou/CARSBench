from __future__ import annotations

from CARSBench.datasets.schema import SpectrumSample


def get_raman_target(sample: SpectrumSample):
    """
    Primary supervised target for Raman retrieval tasks.
    """
    return sample.raman_target


def get_measured_spectrum(sample: SpectrumSample):
    """
    Measured input spectrum used by retrieval models.
    """
    return sample.spectrum


def get_axis(sample: SpectrumSample):
    """
    Spectral axis of the sample.
    """
    return sample.axis