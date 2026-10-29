
from pathlib import Path
import pandas as pd


REFERENCE_ROOT = Path(
    "/content/raman_reference/extracted"
)

csv_files = list(
    REFERENCE_ROOT.rglob("*.csv")
)

print("=" * 80)
print("RAMAN REFERENCE INVENTORY")
print("=" * 80)

for path in csv_files:
    print(path)


def locate(name):
    matches = [
        p for p in csv_files
        if p.name == name
    ]

    if not matches:
        raise FileNotFoundError(name)

    return matches[0]


metadata_path = locate(
    "metadata_db.csv"
)

spectra_path = locate(
    "raman_spectra_db.csv"
)

peaks_path = locate(
    "raman_peaks_db.csv"
)

metadata = pd.read_csv(
    metadata_path
)

spectra = pd.read_csv(
    spectra_path
)

peaks = pd.read_csv(
    peaks_path
)

print()
print("Metadata rows:", len(metadata))
print("Spectra rows:", len(spectra))
print("Peak-table rows:", len(peaks))

print()
print("Metadata columns:")
print(metadata.columns.tolist())

print()
print("Spectra columns:")
print(spectra.columns.tolist())

print()
print("Peak columns:")
print(peaks.columns.tolist())


# ------------------------------------------------------------
# Search candidate reference spectra for the current
# CARSBench biochemical classes.
# ------------------------------------------------------------

keywords = {
    "lipid": [
        "lipid",
        "fat",
        "fatty",
        "cholesterol",
        "triglycer",
        "phospholipid",
        "oleic",
        "palmit",
        "stear",
        "linole",
    ],

    "protein": [
        "protein",
        "albumin",
        "collagen",
        "keratin",
        "hemoglobin",
        "myoglobin",
    ],

    "nucleic_acid": [
        "dna",
        "rna",
        "nucleic",
        "adenine",
        "guanine",
        "cytosine",
        "thymine",
        "uracil",
        "nucleotide",
    ],

    "glycogen": [
        "glycogen",
        "glucose",
        "carbohydrate",
        "polysaccharide",
    ],
}


component_col = None

for candidate in [
    "component",
    "name",
    "compound",
]:
    if candidate in metadata.columns:
        component_col = candidate
        break

if component_col is None:
    raise RuntimeError(
        "Could not locate component-name column."
    )


text = metadata[
    component_col
].astype(str).str.lower()


for class_name, words in keywords.items():

    mask = False

    for word in words:
        mask = (
            mask
            | text.str.contains(
                word,
                regex=False,
                na=False,
            )
        )

    subset = metadata[
        mask
    ].copy()

    print()
    print("=" * 80)
    print(class_name.upper())
    print("=" * 80)

    if subset.empty:
        print("No candidates found.")
        continue

    columns_to_show = [
        c for c in [
            "id",
            component_col,
            "type",
            "source",
            "reference",
            "sample_composition",
            "laser_wavelength",
            "orig_spectral_resolution",
        ]
        if c in subset.columns
    ]

    print(
        subset[
            columns_to_show
        ].to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("SOURCE TYPES")
print("=" * 80)

if "source" in metadata.columns:
    print(
        metadata[
            "source"
        ].value_counts(
            dropna=False
        )
    )
