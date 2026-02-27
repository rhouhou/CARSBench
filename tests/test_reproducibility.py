import numpy as np
from CARSBench.core.simulate import SimulationConfig, simulate

def test_reproducible():
    out1 = simulate(SimulationConfig(seed=123))
    out2 = simulate(SimulationConfig(seed=123))
    assert np.allclose(out1.I_meas, out2.I_meas)