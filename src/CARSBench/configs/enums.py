from enum import Enum


class WindowMode(str, Enum):
    FULL = "full"
    WIDE = "wide"
    FINGERPRINT = "fingerprint"
    CH = "ch"


class ResonantMode(str, Enum):
    RANDOM = "random"
    COMPONENT = "component"


class NRBFamily(str, Enum):
    SPLINE = "spline"
    POLY = "poly"
    EXP_TILT = "exp_tilt"


class PhaseModel(str, Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"


class EnvelopeFamily(str, Enum):
    FLAT = "flat"
    TILTED = "tilted"
    SPLINE = "spline"


class BaselineFamily(str, Enum):
    NONE = "none"
    POLY = "poly"
    POLY_RIPPLE = "poly+ripple"


class GeneratorType(str, Enum):
    FREQUENCY = "frequency"
    TIME = "time"