import numpy as np
from CARSBench.core.simulate import SimulationConfig, simulate
from CARSBench.config.io import save_output, load_output_npz, load_output_hdf5


def reconstruct_output_from_npz(npz_dict):
    """
    Helper to reconstruct complex arrays from the saved NPZ dict.
    Returns: (nu, chi_r, chi_nrb, chi_total, I_true, I_instr, I_meas, intermediates)
    """
    nu = npz_dict["nu_cm1"]
    chi_r = npz_dict["chi_r_real"] + 1j * npz_dict["chi_r_imag"]
    chi_nrb = npz_dict["chi_nrb_real"] + 1j * npz_dict["chi_nrb_imag"]
    chi_total = npz_dict["chi_total_real"] + 1j * npz_dict["chi_total_imag"]
    I_true = npz_dict["I_true"]
    I_instr = npz_dict["I_instr"]
    I_meas = npz_dict["I_meas"]

    inter = {}
    for k in npz_dict.keys():
        if k.startswith("inter_"):
            inter[k.replace("inter_", "")] = npz_dict[k]
    return nu, chi_r, chi_nrb, chi_total, I_true, I_instr, I_meas, inter


# --- simulate and save ---
cfg = SimulationConfig(seed=7, domain_preset="typical")
out = simulate(cfg)

npz_path = save_output(out, "demo_run", fmt="npz")
print("Saved:", npz_path)

# --- load NPZ ---
loaded = load_output_npz(npz_path)
nu, chi_r, chi_nrb, chi_total, I_true, I_instr, I_meas, inter = reconstruct_output_from_npz(loaded)

print("Loaded NPZ shapes:", nu.shape, chi_r.shape, I_meas.shape)
print("Loaded inter keys:", list(inter.keys()))

# --- optionally save HDF5 (requires h5py) ---
# h5_path = save_output(out, "demo_run", fmt="hdf5")
# loaded_h5 = load_output_hdf5(h5_path)
# print("Loaded H5 keys:", loaded_h5.keys())