import numpy as np
import pytest

from usctbench.data.phase_delay import phase_slope_delays
from usctbench.data.arrival import water_relative_delays
from usctbench.metrics import compute_image_metrics


def test_phase_initialization_uses_only_three_training_frequencies():
    f = np.array([100e3, 150e3, 200e3, 250e3])
    delay = np.array([[1e-6, -2e-6], [0.5e-6, 2e-6]])
    reference = np.ones((4, 2, 2), complex) * (0.3 + 0.7j)
    observed = reference * np.exp(2j * np.pi * f[:, None, None] * delay)
    train = np.ones(observed.shape, bool)
    train[-1] = False
    result, weights, qc = phase_slope_delays(
        observed, reference, f, np.full((2, 2), 0.05), train
    )
    np.testing.assert_allclose(result, delay, atol=1e-18)
    changed = observed.copy()
    changed[-1] = np.nan
    altered, _, _ = phase_slope_delays(
        changed, reference, f, np.full((2, 2), 0.05), train
    )
    np.testing.assert_array_equal(result, altered)
    assert weights.min() > 0.99
    assert qc["heldout_values_used"] is False
    train[2] = False
    with pytest.raises(ValueError, match="three"):
        phase_slope_delays(observed, reference, f, np.full((2, 2), 0.05), train)


def test_bounded_water_lag_sign_nonzero_time_origin_and_missing_traces():
    time = np.arange(1200) * 1e-7 - 10e-6
    arrival = 0.06 / 1500 + 6e-6

    def pulse(centre):
        return np.exp(-(((time - centre) / 2e-6) ** 2)) * np.cos(
            2 * np.pi * 200e3 * (time - centre)
        )

    water = pulse(arrival)[:, None, None] * np.ones((1, 2, 2))
    object_pressure = pulse(arrival + 0.71e-6)[:, None, None] * np.ones((1, 2, 2))
    # A larger late pulse must not attract the direct-arrival correlation.
    object_pressure += 10 * pulse(arrival + 45e-6)[:, None, None]
    object_pressure[:, 0, 1] = np.nan
    delays, valid, weights, _ = water_relative_delays(
        object_pressure, water, time, np.full((2, 2), 0.06)
    )
    assert np.isnan(delays[0, 1]) and not valid[0, 1] and weights[0, 1] == 0
    np.testing.assert_allclose(delays[valid], 0.71e-6, atol=2e-8)


def test_metrics_reject_nonfinite_recon_instead_of_hiding_pixels():
    truth = np.full((12, 12), 1500.0)
    prediction = truth.copy()
    prediction[5, 5] = np.nan
    with pytest.raises(FloatingPointError):
        compute_image_metrics(prediction, truth)
    mask = np.ones(truth.shape, bool)
    mask[5, 5] = False
    assert compute_image_metrics(prediction, truth, mask=mask)["rmse"] == 0
