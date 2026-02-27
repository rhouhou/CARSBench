from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

# Import your outputs (adjust path if needed)
from CARSBench.output import SimulationOutput, BatchSimulationOutput, ImageSimulationOutput

SignalKind = Literal[
    "I_meas", "I_instr", "I_true",
    "im_chi_r", "re_chi_r", "abs_chi_r",
    "im_chi_nrb", "re_chi_nrb", "abs_chi_nrb",
    "im_chi_total", "re_chi_total", "abs_chi_total",
]

@dataclass
class PlotStyle:
    figsize: Tuple[float, float] = (10, 4)
    dpi: int = 120
    grid: bool = True
    title: Optional[str] = None
    xlabel: str = "Raman shift (cm$^{-1}$)"
    ylabel: str = "a.u."
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    legend: bool = True


def _extract_1d(out: SimulationOutput, kind: SignalKind) -> np.ndarray:
    if kind == "I_meas":
        return out.I_meas
    if kind == "I_instr":
        return out.I_instr
    if kind == "I_true":
        return out.I_true

    if kind == "im_chi_r":
        return out.chi_r.imag
    if kind == "re_chi_r":
        return out.chi_r.real
    if kind == "abs_chi_r":
        return np.abs(out.chi_r)

    if kind == "im_chi_nrb":
        return out.chi_nrb.imag
    if kind == "re_chi_nrb":
        return out.chi_nrb.real
    if kind == "abs_chi_nrb":
        return np.abs(out.chi_nrb)

    if kind == "im_chi_total":
        return out.chi_total.imag
    if kind == "re_chi_total":
        return out.chi_total.real
    if kind == "abs_chi_total":
        return np.abs(out.chi_total)

    raise ValueError(f"Unknown kind: {kind}")


def plot_spectrum(
    out: SimulationOutput,
    kinds: Sequence[SignalKind] = ("I_meas",),
    style: PlotStyle = PlotStyle(),
    labels: Optional[Sequence[str]] = None,
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot one or multiple 1D signals from a single SimulationOutput.
    """
    nu = out.nu_cm1
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = plt.gca()

    if labels is None:
        labels = list(kinds)

    for k, lab in zip(kinds, labels):
        y = _extract_1d(out, k)
        ax.plot(nu, y, label=lab)

    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if style.title:
        ax.set_title(style.title)
    if style.grid:
        ax.grid(True, alpha=0.3)

    if style.xlim:
        ax.set_xlim(*style.xlim)
    if style.ylim:
        ax.set_ylim(*style.ylim)

    if style.legend and len(kinds) > 1:
        ax.legend()

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_batch_overlay(
    bout: BatchSimulationOutput,
    kind: SignalKind = "I_meas",
    max_lines: int = 30,
    alpha: float = 0.25,
    style: PlotStyle = PlotStyle(),
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Overlay multiple spectra from a batch (B,N).
    Useful to visualize variability/noise across conditions.
    """
    nu = bout.nu_cm1
    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = plt.gca()

    # Build a fake single-output wrapper for extraction reuse
    # (We just handle the batch selection directly.)
    if kind.startswith("I_"):
        data = getattr(bout, kind)  # (B,N)
    else:
        if kind.endswith("chi_r"):
            arr = bout.chi_r
        elif kind.endswith("chi_nrb"):
            arr = bout.chi_nrb
        elif kind.endswith("chi_total"):
            arr = bout.chi_total
        else:
            raise ValueError(f"Unsupported kind for batch: {kind}")

        if kind.startswith("im_"):
            data = arr.imag
        elif kind.startswith("re_"):
            data = arr.real
        elif kind.startswith("abs_"):
            data = np.abs(arr)
        else:
            raise ValueError(f"Unsupported kind for batch: {kind}")

    B = data.shape[0]
    n_lines = min(B, max_lines)
    for i in range(n_lines):
        ax.plot(nu, data[i], alpha=alpha)

    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title or f"Batch overlay: {kind} (showing {n_lines}/{B})")
    if style.grid:
        ax.grid(True, alpha=0.3)
    if style.xlim:
        ax.set_xlim(*style.xlim)
    if style.ylim:
        ax.set_ylim(*style.ylim)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_image_pixel_spectrum(
    img: ImageSimulationOutput,
    kind: SignalKind = "I_meas",
    pixel: Tuple[int, int] = (0, 0),
    style: PlotStyle = PlotStyle(),
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a spectrum from a single pixel (row,col) of an image cube.
    Supports order H_W_N or N_H_W stored in img.order.
    """
    r, c = pixel
    nu = img.nu_cm1

    # Extract cube
    if kind.startswith("I_"):
        cube = getattr(img, kind)
    else:
        if kind.endswith("chi_r"):
            cube = img.chi_r
        elif kind.endswith("chi_nrb"):
            cube = img.chi_nrb
        elif kind.endswith("chi_total"):
            cube = img.chi_total
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        if kind.startswith("im_"):
            cube = np.imag(cube)
        elif kind.startswith("re_"):
            cube = np.real(cube)
        elif kind.startswith("abs_"):
            cube = np.abs(cube)
        else:
            raise ValueError(f"Unsupported kind: {kind}")

    if img.order == "H_W_N":
        y = cube[r, c, :]
    elif img.order == "N_H_W":
        y = cube[:, r, c]
    else:
        raise ValueError(f"Unknown img.order: {img.order}")

    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = plt.gca()
    ax.plot(nu, y)

    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    ax.set_title(style.title or f"{kind} at pixel (r={r}, c={c})")
    if style.grid:
        ax.grid(True, alpha=0.3)
    if style.xlim:
        ax.set_xlim(*style.xlim)
    if style.ylim:
        ax.set_ylim(*style.ylim)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_image_band(
    img: ImageSimulationOutput,
    kind: SignalKind = "I_meas",
    nu_value: float = 1000.0,
    style: PlotStyle = PlotStyle(figsize=(5, 5)),
    savepath: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Show a 2D image at a chosen Raman shift (nearest index).
    """
    nu = img.nu_cm1
    idx = int(np.argmin(np.abs(nu - nu_value)))

    # Extract cube (same logic as pixel plot)
    if kind.startswith("I_"):
        cube = getattr(img, kind)
    else:
        if kind.endswith("chi_r"):
            cube = img.chi_r
        elif kind.endswith("chi_nrb"):
            cube = img.chi_nrb
        elif kind.endswith("chi_total"):
            cube = img.chi_total
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        if kind.startswith("im_"):
            cube = np.imag(cube)
        elif kind.startswith("re_"):
            cube = np.real(cube)
        elif kind.startswith("abs_"):
            cube = np.abs(cube)
        else:
            raise ValueError(f"Unsupported kind: {kind}")

    if img.order == "H_W_N":
        im2d = cube[:, :, idx]
    elif img.order == "N_H_W":
        im2d = cube[idx, :, :]
    else:
        raise ValueError(f"Unknown img.order: {img.order}")

    fig = plt.figure(figsize=style.figsize, dpi=style.dpi)
    ax = plt.gca()
    m = ax.imshow(im2d, aspect="equal")
    ax.set_title(style.title or f"{kind} at {nu[idx]:.1f} cm$^{{-1}}$")
    fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    return fig