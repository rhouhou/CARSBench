import numpy as np
from CARSBench.core.simulate import SimulationConfig, simulate

cfg = SimulationConfig(seed=42)
out = simulate(cfg)

print("nu:", out.nu_cm1.shape)
print("chi_r:", out.chi_r.shape, out.chi_r.dtype)
print("I_meas:", out.I_meas.shape, out.I_meas.min(), out.I_meas.max())