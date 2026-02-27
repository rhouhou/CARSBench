from enum import Enum

class WindowType(str, Enum):
    FULL = "full"
    FINGERPRINT = "fingerprint"
    CH = "ch"
    WIDE = "wide"

class LineShape(str, Enum):
    GAUSSIAN = "gaussian"
    VOIGT = "voigt"

class NRBMagFamily(str, Enum):
    SPLINE = "spline"
    POLY = "poly"
    EXP_TILT = "exp_tilt"

class NRBPhaseModel(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"

class NoiseModel(str, Enum):
    POISSON_READ = "poisson_read"
    GAUSSIAN_ONLY = "gaussian_only"
    NONE = "none"

class ResonantSource(str, Enum):
    SYNTHETIC_PEAKS = "synthetic_peaks"
    FROM_RAMAN_SPECTRUM = "from_raman_spectrum"  # future hook

class SaveFormat(str, Enum):
    NPZ = "npz"
    HDF5 = "hdf5"

class CubeOrder(str, Enum):
    H_W_N = "H_W_N"   # (H, W, N)
    N_H_W = "N_H_W"   # (N, H, W)