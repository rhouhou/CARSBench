import numpy as np

def convolve_instrument_psf(I: np.ndarray, psf: np.ndarray) -> np.ndarray:
    # Same-length convolution via 'same' mode
    return np.convolve(I, psf, mode="same")