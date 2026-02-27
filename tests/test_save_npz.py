import os
from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.config.io import save_output

def test_save_npz(tmp_path):
    out = simulate(SimulationConfig(seed=0))
    p = tmp_path / "out"
    fname = save_output(out, str(p), fmt="npz")
    assert os.path.exists(fname)