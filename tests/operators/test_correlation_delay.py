import numpy as np
import pytest

from usctbench.core.schema import GridSpec, GeometrySpec
from usctbench.operators.forward.correlation_delay import (
    CorrelationDelayDerivative,
    CorrelationTravelTimeJacobian,
)
from usctbench.operators.forward.ray_born import RayBornOperator


def test_positive_delay_sign_and_quadratic_small_shift_error():
    f = np.linspace(1e5, 4e5, 12)
    omega = 2 * np.pi * f[:, None, None]
    w = np.exp(-(((f - 2.5e5) / 1e5) ** 2))[:, None, None] * np.ones((1, 2, 3))
    op = CorrelationDelayDerivative(f, w)
    errors = []
    for shift in (1e-8, 5e-9):
        result = op.forward(np.exp(1j * omega * shift) * np.ones(w.shape) - 1)
        errors.append(np.max(np.abs(result - shift)))
    assert errors[1] < 0.13 * errors[0]
    np.testing.assert_allclose(op.forward(1j * omega * np.ones(w.shape)), 1)


def test_real_adjoint_gain_invariance_and_bad_frequencies():
    rng = np.random.default_rng(71)
    f = np.linspace(1e5, 3e5, 7)
    water = rng.normal(size=(7, 2, 4)) + 1j * rng.normal(size=(7, 2, 4))
    water[:, 0, 0] = 0
    op = CorrelationDelayDerivative(f, water)
    data = rng.normal(size=water.shape) + 1j * rng.normal(size=water.shape)
    sensitivity = rng.normal(size=water.shape[1:])
    assert np.isnan(op.forward(data)[0, 0])
    lhs = np.nansum(op.forward(data) * sensitivity)
    rhs = np.vdot(data, op.adjoint(sensitivity)).real
    np.testing.assert_allclose(lhs, rhs, rtol=1e-13)
    scaled = CorrelationDelayDerivative(f, water * (3 + 2j))
    np.testing.assert_allclose(op.forward(data), scaled.forward(data))
    data[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        op.forward(data)
    with pytest.raises(ValueError, match="real"):
        op.adjoint(sensitivity.astype(complex))


def test_receiver_locality_and_insufficient_band_rejection():
    f = np.linspace(1e5, 4e5, 4)
    water = np.ones((4, 2, 2), dtype=complex)
    ratio = np.full(water.shape, 0.1j)
    baseline = CorrelationDelayDerivative(f, water).forward(ratio)
    water[:, 0, 0] *= np.arange(1, 5)
    ratio[:, 0, 0] *= 1e3
    changed = CorrelationDelayDerivative(f, water).forward(ratio)
    np.testing.assert_array_equal(baseline[1], changed[1])
    water[:2, 1, 0] = np.nan
    op = CorrelationDelayDerivative(f, water)
    assert not op.valid[1, 0]
    assert np.isnan(op.forward(ratio)[1, 0])


def test_born_composition_exact_adjoint():
    grid = GridSpec(shape=(12, 12), spacing_m=(0.001, 0.001), origin_m=(-0.006, -0.006))
    geom = GeometrySpec(
        tx_pos_m=[[-0.01, 0], [0, -0.01]], rx_pos_m=[[0.01, 0], [0, 0.01]]
    )
    f = np.linspace(1e5, 3e5, 4)
    born = RayBornOperator(grid, geom, f)
    obs = CorrelationDelayDerivative(f, born.background_data())
    op = CorrelationTravelTimeJacobian(born, obs)
    rng = np.random.default_rng(2)
    dm = rng.normal(size=grid.shape) * 1e-9
    y = rng.normal(size=4)
    np.testing.assert_allclose(
        np.dot(op.forward(dm), y),
        np.vdot(dm, op.adjoint(y)).real,
        rtol=1e-12,
        atol=1e-20,
    )
