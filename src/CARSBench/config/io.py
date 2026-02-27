from __future__ import annotations
from typing import Optional, Dict, Any
import json
import numpy as np

from CARSBench.config.enums import SaveFormat
from CARSBench.output import SimulationOutput, BatchSimulationOutput, ImageSimulationOutput

def save_output(
    out: SimulationOutput,
    path: str,
    fmt: str | SaveFormat = SaveFormat.NPZ,
    save_meta_json: bool = True,
) -> str:
    fmt = SaveFormat(fmt)
    if fmt == SaveFormat.NPZ:
        return _save_npz(out, path, save_meta_json=save_meta_json)
    if fmt == SaveFormat.HDF5:
        return _save_hdf5(out, path)
    raise ValueError(f"Unknown save format: {fmt}")

def _save_npz(out: SimulationOutput, path: str, save_meta_json: bool = True) -> str:
    if not path.endswith(".npz"):
        path = path + ".npz"

    arrays = {
        "nu_cm1": out.nu_cm1,
        "chi_r_real": out.chi_r.real,
        "chi_r_imag": out.chi_r.imag,
        "chi_nrb_real": out.chi_nrb.real,
        "chi_nrb_imag": out.chi_nrb.imag,
        "chi_total_real": out.chi_total.real,
        "chi_total_imag": out.chi_total.imag,
        "I_true": out.I_true,
        "I_instr": out.I_instr,
        "I_meas": out.I_meas,
    }
    # intermediates (optional)
    for k, v in out.intermediates.items():
        arrays[f"inter_{k}"] = v

    np.savez_compressed(path, **arrays)

    if save_meta_json:
        meta_path = path.replace(".npz", ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(out.meta, f, indent=2, ensure_ascii=False)

    return path

def _save_hdf5(out: SimulationOutput, path: str) -> str:
    try:
        import h5py  # optional dependency
    except Exception as e:
        raise ImportError("To save HDF5, install: pip install bcarsim[hdf5]") from e

    if not (path.endswith(".h5") or path.endswith(".hdf5")):
        path = path + ".h5"

    with h5py.File(path, "w") as f:
        f.create_dataset("nu_cm1", data=out.nu_cm1)

        f.create_dataset("chi_r_real", data=out.chi_r.real)
        f.create_dataset("chi_r_imag", data=out.chi_r.imag)

        f.create_dataset("chi_nrb_real", data=out.chi_nrb.real)
        f.create_dataset("chi_nrb_imag", data=out.chi_nrb.imag)

        f.create_dataset("chi_total_real", data=out.chi_total.real)
        f.create_dataset("chi_total_imag", data=out.chi_total.imag)

        f.create_dataset("I_true", data=out.I_true)
        f.create_dataset("I_instr", data=out.I_instr)
        f.create_dataset("I_meas", data=out.I_meas)

        g = f.create_group("intermediates")
        for k, v in out.intermediates.items():
            g.create_dataset(k, data=v)

        # meta as JSON string attribute
        f.attrs["meta_json"] = json.dumps(out.meta)

    return path

def load_output_npz(path: str) -> Dict[str, np.ndarray]:
    return dict(np.load(path))

def load_output_hdf5(path: str) -> Dict[str, Any]:
    import json
    import h5py
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if k == "intermediates":
                out["intermediates"] = {kk: f[k][kk][()] for kk in f[k].keys()}
            else:
                out[k] = f[k][()]
        out["meta"] = json.loads(f.attrs.get("meta_json", "{}"))
    return out

def _is_image(out) -> bool:
    return hasattr(out, "height") and hasattr(out, "width") and hasattr(out, "order")

def _is_batch(out) -> bool:
    return hasattr(out, "meta") and isinstance(getattr(out, "I_meas", None), np.ndarray) and out.I_meas.ndim == 2