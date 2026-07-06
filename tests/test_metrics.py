import numpy as np

from CARSBench import mae, rmse, spectral_angle


def test_rmse_zero_for_identical_arrays():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert rmse(y_pred, y_true) == 0.0


def test_mae_zero_for_identical_arrays():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    assert mae(y_pred, y_true) == 0.0


def test_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])

    expected = np.sqrt((1.0**2 + 0.0**2 + 1.0**2) / 3.0)

    np.testing.assert_allclose(rmse(y_pred, y_true), expected)


def test_mae_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 4.0])

    expected = (1.0 + 0.0 + 1.0) / 3.0

    np.testing.assert_allclose(mae(y_pred, y_true), expected)


def test_spectral_angle_identical_vectors_is_near_zero():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(spectral_angle(y_pred, y_true), 0.0, atol=1e-6)


def test_spectral_angle_orthogonal_vectors_is_pi_over_two():
    y_true = np.array([1.0, 0.0])
    y_pred = np.array([0.0, 1.0])

    np.testing.assert_allclose(
        spectral_angle(y_pred, y_true),
        np.pi / 2,
        atol=1e-8,
    )
