from CARSBench.core.simulate import SimulationConfig, simulate

def test_shapes():
    out = simulate(SimulationConfig(seed=0))
    n = out.nu_cm1.size
    assert out.chi_r.size == n
    assert out.chi_nrb.size == n
    assert out.I_meas.size == n