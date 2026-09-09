import numpy as np
import pytest

from usctbench.evaluation.arrival_consistency import translated_reference_residual


def pulse_data():
    t = np.arange(1800) * 1e-7
    centre = 0.06 / 1500 + 6e-6

    def pulse(shift):
        return np.exp(-(((t - centre - shift) / 2e-6) ** 2)) * np.cos(
            2 * np.pi * 200e3 * (t - centre - shift)
        )

    return t, pulse


def test_translation_residual_is_pair_local_gain_invariant_and_not_a_picker_score():
    t, pulse = pulse_data()
    water = pulse(0)[:, None, None] * np.ones((1, 2, 2))
    p = 2 * pulse(1e-6)[:, None, None] * np.ones((1, 2, 2))
    kwargs = dict(
        time_s=t, delay_s=np.full((2, 2), 1e-6), distances_m=np.full((2, 2), 0.06)
    )
    residual, gain = translated_reference_residual(p, water, **kwargs)
    np.testing.assert_allclose(residual, 0, atol=1e-13)
    np.testing.assert_allclose(gain, 2, atol=1e-13)
    p[:, 1, 1] += pulse(4e-6)
    changed, _ = translated_reference_residual(p, water, **kwargs)
    assert changed[1, 1] > 0.1
    np.testing.assert_array_equal(changed[0], residual[0])


def test_missing_and_truncated_data_are_not_zero_residual():
    t, pulse = pulse_data()
    p = pulse(0)[:, None, None]
    kwargs = dict(delay_s=np.zeros((1, 1)), distances_m=np.full((1, 1), 0.06))
    r, _ = translated_reference_residual(p[:550], p[:550], t[:550], **kwargs)
    assert np.isnan(r).all()
    p[:] = 0
    r, _ = translated_reference_residual(p, p, t, **kwargs)
    assert np.isnan(r).all()
    with pytest.raises(ValueError):
        translated_reference_residual(p, p, t[::-1], **kwargs)
