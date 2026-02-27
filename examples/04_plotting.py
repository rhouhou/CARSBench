from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.core.batch import simulate_batch, simulate_image
from CARSBench.config.spatial import BatchConfig, SpatialConfig
from CARSBench.config.enums import CubeOrder
from CARSBench.viz import PlotStyle, plot_spectrum, plot_batch_overlay, plot_image_pixel_spectrum, plot_image_band

# --- Single spectrum ---
cfg = SimulationConfig(seed=1, domain_preset="typical")
out = simulate(cfg)

style = PlotStyle(title="Typical domain: I_meas vs Im{chi_R}", xlim=(400, 3200))
plot_spectrum(out, kinds=("I_meas", "im_chi_r"), labels=("Measured intensity", "Im{chi_R}"), style=style)

# --- Batch overlay (variability view) ---
cfgb = SimulationConfig(seed=2, domain_preset="noisy")
cfgb.batch = BatchConfig(batch_size=50)
bout = simulate_batch(cfgb)

plot_batch_overlay(bout, kind="I_meas", max_lines=40, alpha=0.2, style=PlotStyle(title="Noisy domain: batch overlay"))

# --- Image cube ---
cfgi = SimulationConfig(seed=3, domain_preset="typical")
cfgi.spatial = SpatialConfig(height=16, width=16, order=CubeOrder.H_W_N)
cfgi.batch = BatchConfig(batch_size=16 * 16)
img = simulate_image(cfgi)

plot_image_pixel_spectrum(img, kind="I_meas", pixel=(5, 7), style=PlotStyle(title="Pixel spectrum"))
plot_image_band(img, kind="I_meas", nu_value=1000.0, style=PlotStyle(title="Band image at ~1000 cm^-1"))