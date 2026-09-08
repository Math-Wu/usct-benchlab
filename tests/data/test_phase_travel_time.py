import numpy as np
import pytest

from usctbench.data.phase_travel_time import water_relative_phase_delays
from usctbench.data.waveforms import pressure_spectrum


def test_known_phase_delay_cycles_amplitude_units_and_missing_values():
    f = 500000.0
    target = np.array([[0.2, 1.4, -2.3]]) * 1e-6
    anchor = target + 0.1e-6
    water = np.exp(1j * np.array([[0.2, 0.6, -2.8]]))
    pressure = water * np.exp(2j * np.pi * f * target)
    for units in (1.0, 1e-200, 1e200):
        delay, valid, qc = water_relative_phase_delays(
            units * pressure, units * water, f, anchor, np.full(target.shape, 0.1)
        )
        np.testing.assert_allclose(delay, target, atol=2e-21)
        assert valid.all() and qc["nonzero_cycle_channels"] == 2
    pressure[0, 1] = np.nan
    delay, valid, _ = water_relative_phase_delays(
        pressure, water, f, anchor, np.full(target.shape, 0.1)
    )
    assert not valid[0, 1] and np.isnan(delay[0, 1])


def test_channel_locality_under_heldout_mutation():
    rng = np.random.default_rng(72)
    f = 200000.0
    anchor = rng.uniform(-0.5e-6, 0.5e-6, (5, 5))
    water = np.exp(1j * rng.normal(size=(5, 5)))
    pressure = water * np.exp(2j * np.pi * f * anchor)
    distance = np.full((5, 5), 0.1)
    a, _, _ = water_relative_phase_delays(pressure, water, f, anchor, distance)
    train = np.ones((5, 5), dtype=bool)
    train[1, :] = False
    train[:, 1] = False
    pressure[~train] = np.nan
    water[~train] = np.nan
    anchor[~train] = np.nan
    b, _, _ = water_relative_phase_delays(pressure, water, f, anchor, distance)
    np.testing.assert_array_equal(a[train], b[train])


def test_delayed_real_pulse_and_nonzero_time_origin():
    t = -5e-6 + np.arange(2048) * 0.02e-6
    f, shift = 500000.0, 1.2e-6

    def pulse(t):
        return np.exp(-(((t - 10e-6) / 0.8e-6) ** 2)) * np.cos(
            2 * np.pi * f * (t - 10e-6)
        )

    water = pressure_spectrum(pulse(t)[:, None, None], t, [f])[0]
    pressure = pressure_spectrum(pulse(t - shift)[:, None, None], t, [f])[0]
    delay, valid, _ = water_relative_phase_delays(
        pressure, water, f, [[1.1e-6]], [[0.1]]
    )
    np.testing.assert_allclose(delay, shift, atol=1e-18)
    assert valid.all()


def test_half_cycle_ambiguity_and_weak_reference_are_missing_not_zero():
    delay, valid, _ = water_relative_phase_delays(
        np.ones((1, 2), complex),
        np.array([[1, 0]], complex),
        500000,
        np.array([[1e-6, 0]]),
        np.full((1, 2), 0.1),
    )
    assert not valid.any() and np.isnan(delay).all()


@pytest.mark.parametrize("frequency", [0, -1, np.nan, np.inf])
def test_bad_frequency(frequency):
    with pytest.raises(ValueError):
        water_relative_phase_delays([[1]], [[1]], frequency, [[0]], [[0.1]])


def test_malformed_shapes_and_mask():
    with pytest.raises(ValueError):
        water_relative_phase_delays([1], [1], 500000, [0], [0.1])
    with pytest.raises(ValueError):
        water_relative_phase_delays(
            [[1]], [[1]], 500000, [[0]], [[0.1]], valid_mask=[[1]]
        )
