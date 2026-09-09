import numpy as np
import pytest

from usctbench.data.phase_delay import phase_slope_delays
from usctbench.data.arrival import arrival_observation_metadata, water_relative_delays
from usctbench.core.provenance import case_measurement_metadata
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


@pytest.mark.parametrize("shift", [-0.71e-6, 0.93e-6])
def test_envelope_onset_is_water_relative_and_amplitude_invariant(shift):
    time = np.arange(1800) * 5e-8 - 5e-6
    center = 0.06 / 1500 + 3e-6

    def pulse(offset):
        return np.exp(-(((time - center - offset) / 1e-6) ** 2)) * np.cos(
            2 * np.pi * 800e3 * (time - center - offset)
        )

    pressure = 0.2 * pulse(shift)[:, None, None]
    water = pulse(0)[:, None, None]
    delays, valid, weights, qc = water_relative_delays(
        pressure,
        water,
        time,
        np.array([[0.06]]),
        picker="envelope",
        pulse_duration_s=6e-6,
    )
    assert valid.all() and weights.min() > 0.9
    np.testing.assert_allclose(delays, shift, atol=1e-8)
    assert qc["ground_truth_used"] is False
    zeros, valid, _, _ = water_relative_delays(
        np.zeros_like(pressure),
        water,
        time,
        np.array([[0.06]]),
        picker="envelope",
    )
    assert not valid.any() and np.isnan(zeros).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"picker": "unknown"},
        {"envelope_fraction": 0},
        {"minimum_peak_snr": -1},
        {"pulse_duration_s": float("nan")},
        {"source_onset_s": float("nan")},
    ],
)
def test_arrival_configuration_fails_explicitly(settings):
    with pytest.raises(ValueError):
        water_relative_delays(
            np.zeros((20, 1, 1)),
            np.zeros((20, 1, 1)),
            np.arange(20) * 1e-7,
            np.array([[0.01]]),
            **settings,
        )


@pytest.mark.parametrize("picker", ["xcorr", "envelope", "aic"])
def test_source_trigger_is_not_confused_with_recording_time_origin(picker):
    t = np.arange(1800) * 5e-8
    centre = 0.06 / 1500 + 3e-6

    def pulse(shift):
        return (
            np.exp(-(((t - centre - shift) / 1e-6) ** 2))
            * np.cos(2 * np.pi * 800e3 * (t - centre - shift))
        )[:, None, None]

    p, w = pulse(0.71e-6), pulse(0)
    args = dict(picker=picker, pulse_duration_s=6e-6)
    a, valid, weights, qc = water_relative_delays(p, w, t, np.array([[0.06]]), **args)
    b, bv, bw, bqc = water_relative_delays(
        p, w, t + 0.005, np.array([[0.06]]), source_onset_s=0.005, **args
    )
    np.testing.assert_array_equal(valid, bv)
    np.testing.assert_allclose(a, b, atol=1e-15, rtol=0)
    np.testing.assert_allclose(weights, bw)
    metadata = arrival_observation_metadata(bqc)
    restored = case_measurement_metadata(metadata)["tof_observation_contract"]
    assert restored["geometric_first_arrival_certified"] is False
    assert restored["source_onset_s"] == 0.005
    assert restored["sign"] == "object_minus_water"
    assert "not_inverse_noise_variance" in restored["weight_meaning"]
    assert valid.all()


def test_float32_water_identity_has_no_artificial_subsample_lag():
    rng = np.random.default_rng(6)
    t = np.arange(1400) * 1e-7
    w = rng.normal(size=(1400, 2, 2)).astype(np.float32)
    delay, valid, _, _ = water_relative_delays(w, w, t, np.full((2, 2), 0.06))
    assert valid.all()
    np.testing.assert_allclose(delay, 0, atol=1e-19)


@pytest.mark.parametrize("shift", [-0.7e-6, 0.9e-6])
def test_modified_aic_on_causal_shift_control_and_not_a_gt_oracle(shift):
    time = np.arange(1600) * 5e-8
    u = time - 0.06 / 1500

    def trace(offset):
        q = np.maximum(u - offset, 0)
        return (
            np.where(
                u > offset,
                np.sin(2 * np.pi * 500e3 * q)
                * np.sin(np.pi * np.minimum(q / 6e-6, 1)) ** 2,
                0,
            )
        )[:, None, None]

    w = trace(0).astype(np.float32)
    p = 0.2 * trace(shift).astype(np.float32)
    delta, valid, _, qc = water_relative_delays(
        p, w, time, np.array([[0.06]]), picker="aic", pulse_duration_s=6e-6
    )
    assert valid.all()
    np.testing.assert_allclose(delta, shift, atol=2e-8, rtol=0)
    assert qc["ground_truth_used"] is False
    invalid = water_relative_delays(
        np.zeros_like(p),
        w,
        time,
        np.array([[0.06]]),
        picker="aic",
        pulse_duration_s=6e-6,
    )
    assert not invalid[1].any() and np.isnan(invalid[0]).all()
