
from __future__ import annotations

from typing import Dict

from .components_v2 import (
    CorrelatedPeakGroupV2,
    MolecularSubtypeV2,
    PeakAnchorV2,
    validate_subtype_spec,
)


# ============================================================
# IMPORTANT DESIGN RULE
# ============================================================
#
# This library contains experimentally anchored fingerprint-region
# Raman structures derived from the external Raman reference audit.
#
# Current coverage:
#     450--1800 cm^-1
#
# Correlated groups are currently STRUCTURAL ANNOTATIONS.
# Their variation magnitudes remain exactly zero until separately
# calibrated from experimental data or defensible literature priors.
#
# We therefore do NOT yet invent covariance strengths.
# ============================================================


def _peak(
    label: str,
    center: float,
    intensity: float,
    *groups: str,
) -> PeakAnchorV2:

    return PeakAnchorV2(
        label=label,
        center_cm1=float(center),
        relative_intensity=float(intensity),
        hwhm_cm1=None,
        group_names=tuple(groups),
        presence_prob=1.0,
    )


def _group(
    name: str,
) -> CorrelatedPeakGroupV2:

    return CorrelatedPeakGroupV2(
        name=name,

        # Not calibrated yet.
        amplitude_log_sigma=0.0,
        center_shift_sigma_cm1=0.0,

        # No stochastic group dropout yet.
        joint_presence_prob=1.0,
        controls_presence=False,
    )


# ============================================================
# Lipids
# ============================================================


def _oleic_acid() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="oleic_acid",
        biochemical_class="lipid",

        correlated_groups=(
            _group("lipid_chain"),
            _group("lipid_unsaturation"),
        ),

        peaks=(
            _peak("p866", 866.0, 0.207),
            _peak("p974", 973.5, 0.134),

            _peak(
                "p1082",
                1082.0,
                0.382,
                "lipid_chain",
            ),

            _peak("p1119", 1119.0, 0.133),

            _peak(
                "p1265",
                1265.0,
                0.319,
                "lipid_unsaturation",
            ),

            _peak(
                "p1304",
                1303.5,
                0.597,
                "lipid_chain",
            ),

            _peak(
                "p1442",
                1442.0,
                1.000,
                "lipid_chain",
            ),

            _peak(
                "p1656",
                1656.0,
                0.766,
                "lipid_unsaturation",
            ),
        ),

        reference_ids=(46, 72),
        reference_wavelengths_nm=(785.0, 532.0),

        notes=(
            "Experimentally anchored consensus from two "
            "RamanBiolib oleic-acid spectra."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _palmitic_acid() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="palmitic_acid",
        biochemical_class="lipid",

        correlated_groups=(
            _group("saturated_chain"),
        ),

        peaks=(
            _peak("p672", 672.0, 0.055),
            _peak("p896", 895.5, 0.298),
            _peak("p910", 910.5, 0.158),

            _peak(
                "p1065",
                1065.0,
                0.776,
                "saturated_chain",
            ),

            _peak("p1101", 1101.0, 0.227),

            _peak(
                "p1131",
                1130.5,
                0.927,
                "saturated_chain",
            ),

            _peak("p1177", 1176.5, 0.099),

            _peak(
                "p1298",
                1298.0,
                0.998,
                "saturated_chain",
            ),

            _peak("p1424", 1423.5, 0.486),

            _peak(
                "p1440",
                1440.5,
                0.814,
                "saturated_chain",
            ),

            _peak(
                "p1466",
                1466.0,
                0.493,
                "saturated_chain",
            ),
        ),

        reference_ids=(47, 73),
        reference_wavelengths_nm=(785.0, 532.0),

        notes=(
            "Experimentally anchored saturated fatty-acid consensus."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _stearic_acid() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="stearic_acid",
        biochemical_class="lipid",

        correlated_groups=(
            _group("saturated_chain"),
        ),

        peaks=(
            _peak("p893", 893.0, 0.059),
            _peak("p912", 912.0, 0.033),

            _peak(
                "p1064",
                1064.5,
                0.422,
                "saturated_chain",
            ),

            _peak("p1102", 1102.5, 0.149),

            _peak(
                "p1131",
                1131.0,
                0.362,
                "saturated_chain",
            ),

            _peak("p1176", 1175.5, 0.130),

            _peak(
                "p1298",
                1298.0,
                0.524,
                "saturated_chain",
            ),

            _peak("p1408", 1408.5, 0.189),

            _peak(
                "p1443",
                1443.0,
                1.000,
                "saturated_chain",
            ),

            _peak(
                "p1464",
                1464.5,
                0.565,
                "saturated_chain",
            ),
        ),

        reference_ids=(51, 77),
        reference_wavelengths_nm=(785.0, 532.0),

        notes=(
            "Experimentally anchored saturated fatty-acid consensus."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _trilinolein() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="trilinolein",
        biochemical_class="lipid",

        correlated_groups=(
            _group("lipid_chain"),
            _group("lipid_unsaturation"),
            _group("lipid_ester"),
        ),

        peaks=(
            _peak("p1076", 1076.0, 0.136),
            _peak("p1108", 1108.0, 0.080),

            _peak(
                "p1265",
                1265.0,
                0.351,
                "lipid_unsaturation",
            ),

            _peak(
                "p1302",
                1302.5,
                0.304,
                "lipid_chain",
            ),

            _peak(
                "p1440",
                1440.0,
                0.524,
                "lipid_chain",
            ),

            _peak(
                "p1656",
                1655.5,
                1.000,
                "lipid_unsaturation",
            ),

            _peak(
                "p1744",
                1744.0,
                0.085,
                "lipid_ester",
            ),
        ),

        reference_ids=(54, 87),
        reference_wavelengths_nm=(785.0, 1064.0),

        notes=(
            "Experimentally anchored unsaturated triglyceride consensus."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _cholesterol() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="cholesterol",
        biochemical_class="lipid",

        correlated_groups=(
            _group("sterol_ring"),
            _group("sterol_unsaturation"),
        ),

        peaks=(
            _peak(
                "p548",
                548.0,
                0.375,
                "sterol_ring",
            ),

            _peak(
                "p701",
                701.0,
                0.617,
                "sterol_ring",
            ),

            _peak(
                "p1087",
                1087.0,
                0.336,
                "sterol_ring",
            ),

            _peak(
                "p1130",
                1130.0,
                0.352,
                "sterol_ring",
            ),

            _peak(
                "p1178",
                1178.0,
                0.420,
                "sterol_ring",
            ),

            _peak("p1442", 1442.0, 1.000),

            _peak(
                "p1672",
                1672.0,
                0.577,
                "sterol_unsaturation",
            ),
        ),

        reference_ids=(64,),
        reference_wavelengths_nm=(532.0,),

        notes=(
            "Single-reference experimental anchor; "
            "variation not yet estimated."
        ),
    )

    validate_subtype_spec(spec)
    return spec


# ============================================================
# Proteins
# ============================================================


def _albumin() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="albumin",
        biochemical_class="protein",

        correlated_groups=(
            _group("protein_backbone"),
            _group("protein_amide"),
        ),

        peaks=(
            _peak("p512", 512.0, 0.234),
            _peak("p752", 751.5, 0.133),
            _peak("p828", 828.0, 0.195),
            _peak("p856", 856.0, 0.229),
            _peak("p1007", 1007.0, 0.724),
            _peak("p1036", 1036.0, 0.229),
            _peak("p1128", 1127.5, 0.180),

            _peak(
                "p1268",
                1268.0,
                0.269,
                "protein_amide",
            ),

            _peak(
                "p1321",
                1321.0,
                0.591,
                "protein_backbone",
            ),

            _peak(
                "p1340",
                1340.5,
                0.632,
                "protein_backbone",
            ),

            _peak(
                "p1454",
                1454.0,
                0.944,
                "protein_backbone",
            ),

            _peak("p1610", 1610.0, 0.396),

            _peak(
                "p1657",
                1657.0,
                0.969,
                "protein_amide",
            ),
        ),

        reference_ids=(99, 100, 101),
        reference_wavelengths_nm=(488.0, 532.0, 1064.0),

        notes=(
            "Consensus across three albumin reference spectra. "
            "Group labels are conservative structural annotations."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _collagen() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="collagen",
        biochemical_class="protein",

        correlated_groups=(
            _group("collagen_backbone"),
            _group("protein_amide"),
        ),

        peaks=(
            _peak("p538", 537.5, 0.146),
            _peak("p757", 757.0, 0.129),
            _peak("p820", 820.0, 0.214),
            _peak("p859", 859.0, 0.395),
            _peak("p883", 883.0, 0.376),
            _peak("p922", 922.0, 0.364),
            _peak("p940", 940.0, 0.346),
            _peak("p969", 969.0, 0.208),
            _peak("p1007", 1007.0, 0.251),
            _peak("p1038", 1037.5, 0.281),

            _peak(
                "p1273",
                1273.0,
                0.592,
                "protein_amide",
            ),

            _peak(
                "p1312",
                1312.0,
                0.318,
                "collagen_backbone",
            ),

            _peak(
                "p1453",
                1453.0,
                1.000,
                "collagen_backbone",
            ),

            _peak(
                "p1668",
                1668.0,
                0.964,
                "protein_amide",
            ),
        ),

        reference_ids=(105, 106, 107),
        reference_wavelengths_nm=(488.0, 532.0, 1064.0),

        notes=(
            "Consensus across three collagen reference spectra."
        ),
    )

    validate_subtype_spec(spec)
    return spec


# ============================================================
# Nucleic acids
# ============================================================


def _nucleic_acid_subtype(
    name: str,
    reference_id: int,
    wavelength: float,
    values: tuple[
        tuple[float, float],
        ...
    ],
) -> MolecularSubtypeV2:

    groups = (
        _group("nucleic_acid_backbone"),
        _group("nucleic_acid_bases"),
    )

    peaks = []

    for center, intensity in values:

        # Conservative structural annotation only.
        #
        # Around 1050--1150 cm^-1:
        #     backbone/phosphate-sensitive region
        #
        # Around 1200--1600 cm^-1:
        #     strongly base-containing region
        #
        # No variation strength is assigned yet.

        peak_groups = ()

        if 1050.0 <= center <= 1150.0:
            peak_groups = (
                "nucleic_acid_backbone",
            )

        elif 1200.0 <= center <= 1600.0:
            peak_groups = (
                "nucleic_acid_bases",
            )

        peaks.append(
            _peak(
                f"p{int(round(center))}",
                center,
                intensity,
                *peak_groups,
            )
        )

    spec = MolecularSubtypeV2(
        name=name,
        biochemical_class="nucleic_acid",

        correlated_groups=groups,

        peaks=tuple(peaks),

        reference_ids=(
            reference_id,
        ),

        reference_wavelengths_nm=(
            wavelength,
        ),

        notes=(
            "Single-reference structural anchor. "
            "Correlation magnitudes remain uncalibrated."
        ),
    )

    validate_subtype_spec(spec)
    return spec


def _a_dna() -> MolecularSubtypeV2:

    return _nucleic_acid_subtype(
        "a_dna",
        168,
        514.5,
        (
            (465, .103),
            (499, .892),
            (537, .067),
            (555, .076),
            (592, .026),
            (622, .167),
            (642, .106),
            (663, .243),
            (682, .135),
            (704, .330),
            (727, .588),
            (748, .255),
            (781, .034),
            (794, .248),
            (806, .287),
            (881, .566),
            (893, .958),
            (965, .628),
            (992, .242),
            (1011, .212),
            (1025, .281),
            (1055, .401),
            (1099, 1.000),
            (1145, .290),
            (1182, .100),
            (1209, .057),
            (1243, .161),
            (1255, .385),
            (1301, .756),
            (1322, .439),
            (1336, .450),
            (1373, .118),
            (1396, .124),
            (1418, .243),
            (1461, .446),
            (1483, .470),
            (1537, .260),
            (1574, .176),
            (1603, .098),
            (1612, .057),
            (1651, .010),
            (1669, .026),
            (1717, .015),
        ),
    )


def _b_dna() -> MolecularSubtypeV2:

    return _nucleic_acid_subtype(
        "b_dna",
        169,
        514.5,
        (
            (467, .133),
            (499, .935),
            (535, .297),
            (569, .092),
            (595, .147),
            (643, .094),
            (669, .340),
            (681, .230),
            (729, .830),
            (751, .232),
            (787, .058),
            (795, .224),
            (834, .448),
            (896, .713),
            (923, 1.000),
            (932, .736),
            (970, .115),
            (1000, .142),
            (1013, .200),
            (1028, .194),
            (1054, .306),
            (1094, .853),
            (1142, .259),
            (1182, .091),
            (1217, .055),
            (1237, .040),
            (1257, .207),
            (1292, .936),
            (1304, .889),
            (1319, .523),
            (1339, .235),
            (1376, .158),
            (1422, .240),
            (1445, .380),
            (1463, .450),
            (1489, .545),
            (1511, .461),
            (1538, .329),
            (1577, .163),
            (1610, .078),
            (1649, .048),
            (1669, .051),
            (1712, .051),
        ),
    )


def _trna() -> MolecularSubtypeV2:

    return _nucleic_acid_subtype(
        "trna",
        170,
        488.0,
        (
            (500, .116),
            (557, .130),
            (594, .121),
            (630, .090),
            (670, .180),
            (725, .268),
            (785, 1.000),
            (814, .591),
            (867, .110),
            (915, .083),
            (978, .070),
            (1000, .076),
            (1040, .063),
            (1100, .423),
            (1178, .112),
            (1242, .721),
            (1300, .629),
            (1321, .711),
            (1338, .768),
            (1375, .552),
            (1419, .169),
            (1485, .811),
            (1530, .101),
            (1575, .659),
        ),
    )


# ============================================================
# Glycogen
# ============================================================


def _glycogen() -> MolecularSubtypeV2:

    spec = MolecularSubtypeV2(
        name="glycogen",
        biochemical_class="carbohydrate",

        correlated_groups=(
            _group("glycogen_backbone"),
            _group("glycogen_deformation"),
        ),

        peaks=(
            _peak(
                "p484",
                484.0,
                0.683,
                "glycogen_backbone",
            ),

            _peak("p579", 579.0, 0.236),
            _peak("p596", 596.0, 0.149),

            _peak(
                "p856",
                856.0,
                0.512,
                "glycogen_backbone",
            ),

            _peak(
                "p941",
                941.0,
                0.594,
                "glycogen_backbone",
            ),

            _peak(
                "p1051",
                1051.0,
                0.548,
                "glycogen_backbone",
            ),

            _peak(
                "p1084",
                1084.0,
                0.843,
                "glycogen_backbone",
            ),

            _peak(
                "p1130",
                1130.0,
                0.923,
                "glycogen_backbone",
            ),

            _peak("p1262", 1262.0, 0.378),

            _peak(
                "p1337",
                1337.0,
                1.000,
                "glycogen_deformation",
            ),

            _peak(
                "p1387",
                1387.0,
                0.841,
                "glycogen_deformation",
            ),

            _peak(
                "p1460",
                1460.0,
                0.667,
                "glycogen_deformation",
            ),
        ),

        reference_ids=(188,),
        reference_wavelengths_nm=(1064.0,),

        notes=(
            "Experimentally anchored glycogen fingerprint reference."
        ),
    )

    validate_subtype_spec(spec)
    return spec


# ============================================================
# Library
# ============================================================


def build_experimental_subtype_library_v2(
) -> Dict[str, MolecularSubtypeV2]:
    """
    Build the experimentally anchored CARSBench v2 Raman library.

    Important
    ---------
    This library currently defines spectral structure only.

    It does NOT yet provide:
      - calibrated Lorentzian linewidths,
      - calibrated covariance magnitudes,
      - CH-stretch anchors,
      - absolute Raman cross sections.
    """

    library = {
        # Lipids
        "oleic_acid": _oleic_acid(),
        "palmitic_acid": _palmitic_acid(),
        "stearic_acid": _stearic_acid(),
        "trilinolein": _trilinolein(),
        "cholesterol": _cholesterol(),

        # Proteins
        "albumin": _albumin(),
        "collagen": _collagen(),

        # Nucleic acids
        "a_dna": _a_dna(),
        "b_dna": _b_dna(),
        "trna": _trna(),

        # Carbohydrate
        "glycogen": _glycogen(),
    }

    for spec in library.values():
        validate_subtype_spec(spec)

    return library
