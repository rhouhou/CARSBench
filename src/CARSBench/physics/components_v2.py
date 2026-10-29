
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================
# Experimentally anchored peak
# ============================================================


@dataclass(frozen=True)
class PeakAnchorV2:
    """
    One experimentally anchored Raman resonance.

    Parameters
    ----------
    label:
        Unique peak identifier within a molecular subtype.

    center_cm1:
        Reference Raman-shift position.

    relative_intensity:
        Reference relative Raman intensity.

    hwhm_cm1:
        Lorentzian HWHM. May remain None until linewidth
        calibration has been completed.

    group_names:
        Correlated latent groups controlling this peak.

        A peak may participate in more than one latent factor.
        For example an unsaturated-lipid peak may participate in
        both a general lipid-chain group and an unsaturation group.

    presence_prob:
        Residual peak-specific presence probability. This should
        generally remain 1.0 for experimentally established major
        peaks. Molecular-state/group-level presence should be
        preferred over independent peak dropout.
    """

    label: str
    center_cm1: float
    relative_intensity: float

    hwhm_cm1: Optional[float] = None

    group_names: tuple[str, ...] = ()

    presence_prob: float = 1.0


# ============================================================
# Correlated peak group
# ============================================================


@dataclass(frozen=True)
class CorrelatedPeakGroupV2:
    """
    Shared latent factor controlling a set of Raman peaks.

    Rather than independently perturbing every Raman band, one
    latent variable modifies all peaks belonging to the group.

    Examples
    --------
    lipid_unsaturation:
        coupled variation of C=C-related bands.

    protein_secondary_structure:
        coupled variation of Amide-I / Amide-III-related bands.

    nucleic_acid_backbone:
        coupled variation of phosphate/backbone modes.
    """

    name: str

    # Shared multiplicative variation in log-amplitude space.
    amplitude_log_sigma: float = 0.0

    # Shared spectral-position shift.
    center_shift_sigma_cm1: float = 0.0

    # Whole-group presence probability.
    #
    # If controls_presence=True and the group is absent,
    # all member peaks assigned to this group are absent.
    joint_presence_prob: float = 1.0

    controls_presence: bool = False


# ============================================================
# Molecular subtype
# ============================================================


@dataclass(frozen=True)
class MolecularSubtypeV2:
    """
    Experimentally anchored molecular/subtype Raman model.

    Examples
    --------
    lipid / oleic_acid
    lipid / palmitic_acid
    protein / collagen
    nucleic_acid / b_dna
    carbohydrate / glycogen
    """

    name: str
    biochemical_class: str

    peaks: tuple[PeakAnchorV2, ...]

    correlated_groups: tuple[
        CorrelatedPeakGroupV2,
        ...
    ] = ()

    reference_ids: tuple[int, ...] = ()

    reference_wavelengths_nm: tuple[
        float,
        ...
    ] = ()

    notes: str = ""


# ============================================================
# Realized peak state
# ============================================================


@dataclass(frozen=True)
class RealizedPeakV2:

    label: str

    center_cm1: float
    relative_intensity: float

    hwhm_cm1: Optional[float]

    present: bool


# ============================================================
# Validation
# ============================================================


def validate_subtype_spec(
    spec: MolecularSubtypeV2,
) -> None:

    if not spec.name:
        raise ValueError(
            "Subtype name must not be empty."
        )

    if not spec.biochemical_class:
        raise ValueError(
            "biochemical_class must not be empty."
        )

    group_names = [
        group.name
        for group in spec.correlated_groups
    ]

    if len(group_names) != len(
        set(group_names)
    ):
        raise ValueError(
            "Correlated group names must be unique."
        )

    valid_groups = set(
        group_names
    )

    peak_labels = [
        peak.label
        for peak in spec.peaks
    ]

    if len(peak_labels) != len(
        set(peak_labels)
    ):
        raise ValueError(
            "Peak labels must be unique."
        )

    for peak in spec.peaks:

        if peak.center_cm1 <= 0:
            raise ValueError(
                f"{peak.label}: center must be positive."
            )

        if peak.relative_intensity < 0:
            raise ValueError(
                f"{peak.label}: intensity must be nonnegative."
            )

        if not (
            0.0
            <= peak.presence_prob
            <= 1.0
        ):
            raise ValueError(
                f"{peak.label}: presence_prob must lie in [0, 1]."
            )

        if (
            peak.hwhm_cm1 is not None
            and peak.hwhm_cm1 <= 0
        ):
            raise ValueError(
                f"{peak.label}: HWHM must be positive."
            )

        unknown_groups = (
            set(
                peak.group_names
            )
            - valid_groups
        )

        if unknown_groups:
            raise ValueError(
                f"{peak.label}: unknown groups "
                f"{sorted(unknown_groups)}"
            )

    for group in spec.correlated_groups:

        if group.amplitude_log_sigma < 0:
            raise ValueError(
                f"{group.name}: amplitude_log_sigma "
                "must be nonnegative."
            )

        if group.center_shift_sigma_cm1 < 0:
            raise ValueError(
                f"{group.name}: center shift sigma "
                "must be nonnegative."
            )

        if not (
            0.0
            <= group.joint_presence_prob
            <= 1.0
        ):
            raise ValueError(
                f"{group.name}: joint_presence_prob "
                "must lie in [0, 1]."
            )


# ============================================================
# Correlated sampling
# ============================================================


def sample_subtype_peak_state(
    spec: MolecularSubtypeV2,
    rng: np.random.Generator,
    residual_amplitude_log_sigma: float = 0.05,
    residual_center_sigma_cm1: float = 0.5,
) -> tuple[
    tuple[RealizedPeakV2, ...],
    dict,
]:
    """
    Sample one molecular subtype while preserving peak correlation.

    Model
    -----

    For peak j:

        log A_j =
            log A_j0
            + sum_g sigma_A,g * z_A,g
            + epsilon_A,j

    and:

        nu_j =
            nu_j0
            + sum_g sigma_nu,g * z_nu,g
            + epsilon_nu,j

    where z_g is shared by every peak assigned to group g.

    This creates physically structured covariance without trying
    to estimate a large empirical covariance matrix from the very
    small current Raman reference set.

    Independent residual perturbations remain deliberately small.
    """

    validate_subtype_spec(
        spec
    )

    if residual_amplitude_log_sigma < 0:
        raise ValueError(
            "residual_amplitude_log_sigma "
            "must be nonnegative."
        )

    if residual_center_sigma_cm1 < 0:
        raise ValueError(
            "residual_center_sigma_cm1 "
            "must be nonnegative."
        )

    groups = {
        group.name: group
        for group
        in spec.correlated_groups
    }

    group_state = {}

    for name, group in groups.items():

        if group.controls_presence:

            present = bool(
                rng.random()
                <= group.joint_presence_prob
            )

        else:
            present = True

        amplitude_latent = float(
            rng.normal()
        )

        center_latent = float(
            rng.normal()
        )

        group_state[
            name
        ] = {
            "present": present,
            "amplitude_latent": amplitude_latent,
            "center_latent": center_latent,
        }

    realized = []

    for peak in spec.peaks:

        present = bool(
            rng.random()
            <= peak.presence_prob
        )

        log_amplitude_shift = 0.0
        center_shift = 0.0

        for group_name in peak.group_names:

            group = groups[
                group_name
            ]

            state = group_state[
                group_name
            ]

            if (
                group.controls_presence
                and not state["present"]
            ):
                present = False

            log_amplitude_shift += (
                group.amplitude_log_sigma
                * state[
                    "amplitude_latent"
                ]
            )

            center_shift += (
                group.center_shift_sigma_cm1
                * state[
                    "center_latent"
                ]
            )

        residual_log_amplitude = float(
            rng.normal(
                0.0,
                residual_amplitude_log_sigma,
            )
        )

        residual_center_shift = float(
            rng.normal(
                0.0,
                residual_center_sigma_cm1,
            )
        )

        if peak.relative_intensity > 0:

            intensity = (
                peak.relative_intensity
                * np.exp(
                    log_amplitude_shift
                    + residual_log_amplitude
                )
            )

        else:
            intensity = 0.0

        center = (
            peak.center_cm1
            + center_shift
            + residual_center_shift
        )

        realized.append(
            RealizedPeakV2(
                label=peak.label,
                center_cm1=float(
                    center
                ),
                relative_intensity=float(
                    intensity
                ),
                hwhm_cm1=peak.hwhm_cm1,
                present=present,
            )
        )

    metadata = {
        "subtype": spec.name,
        "biochemical_class": (
            spec.biochemical_class
        ),
        "group_state": group_state,
        "residual_amplitude_log_sigma": float(
            residual_amplitude_log_sigma
        ),
        "residual_center_sigma_cm1": float(
            residual_center_sigma_cm1
        ),
    }

    return (
        tuple(realized),
        metadata,
    )
