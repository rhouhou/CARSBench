from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.core.batch import simulate_batch, simulate_image
from CARSBench.config.spatial import BatchConfig, SpatialConfig
from CARSBench.config.enums import CubeOrder

def test_presets_batch():
    cfg = SimulationConfig(seed=0, domain_preset="typical")
    cfg.batch = BatchConfig(batch_size=8)
    out = simulate_batch(cfg)
    assert out.I_meas.ndim == 2
    assert out.I_meas.shape[0] == 8

def test_presets_image():
    cfg = SimulationConfig(seed=0, domain_preset="typical")
    cfg.spatial = SpatialConfig(height=8, width=8, order=CubeOrder.H_W_N)
    cfg.batch = BatchConfig(batch_size=64)
    out = simulate_image(cfg)
    assert out.I_meas.ndim == 3
    assert out.I_meas.shape[0] == 8
    assert out.I_meas.shape[1] == 8