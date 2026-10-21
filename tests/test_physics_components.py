import numpy as np

from CARSBench.physics.components import (
    build_default_prototype_library,
    sample_prototype_variant,
)


def test_metadata_does_not_change_prototype_spectrum():
    axis = np.linspace(400.0, 3200.0, 1024)

    library = build_default_prototype_library()
    prototype = library.get("lipid")

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    chi_without_metadata = sample_prototype_variant(
        prototype=prototype,
        axis=axis,
        rng=rng1,
        return_metadata=False,
    )

    chi_with_metadata, metadata = sample_prototype_variant(
        prototype=prototype,
        axis=axis,
        rng=rng2,
        return_metadata=True,
    )

    np.testing.assert_allclose(
        chi_without_metadata,
        chi_with_metadata,
    )

    assert metadata["num_realized_peaks"] > 1
    assert len(metadata["realized_peaks"]) == metadata["num_realized_peaks"]

def test_lipid_prototype_has_multiple_realized_peaks():
    axis = np.linspace(400.0, 3200.0, 1024)

    library = build_default_prototype_library()
    prototype = library.get("lipid")

    rng = np.random.default_rng(123)

    _, metadata = sample_prototype_variant(
        prototype=prototype,
        axis=axis,
        rng=rng,
        return_metadata=True,
    )

    assert metadata["num_realized_peaks"] >= 6