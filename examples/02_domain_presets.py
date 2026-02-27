from CARSBench.core.simulate import SimulationConfig, simulate

presets = ["typical", "high_res", "noisy", "calibration_shifted"]

for preset in presets:
    cfg = SimulationConfig(seed=123, domain_preset=preset)
    out = simulate(cfg)

    print(f"\nPreset: {preset}")
    print("  nu range:", float(out.nu_cm1.min()), float(out.nu_cm1.max()))
    print("  I_meas stats:", float(out.I_meas.min()), float(out.I_meas.mean()), float(out.I_meas.max()))
    print("  resolved fwhm:", out.meta["resolved"]["instrument"]["fwhm_res_cm1"])
    print("  resolved noise S:", out.meta["resolved"]["noise"]["intensity_scale"])