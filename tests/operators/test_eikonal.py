import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators import adjoint_error
from usctbench.operators.forward.eikonal import EikonalForward, fast_march


def test_compiled_marching_matches_reference_discretization():
    from usctbench.operators.forward.eikonal import fast_march

    rng = np.random.default_rng(18)
    model = 1 / (1500 + rng.uniform(-30, 30, (21, 23)))
    source = np.array([2.31, 3.17])
    a = fast_march(model, (0.001, 0.0013), source, compiled=False)
    b = fast_march(model, (0.001, 0.0013), source, compiled=True)
    np.testing.assert_allclose(a.times, b.times, rtol=1e-13, atol=1e-18)
    np.testing.assert_array_equal(a.parents, b.parents)
    np.testing.assert_allclose(a.weights, b.weights, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("spatial_order", [1, 2])
def test_homogeneous_offgrid_water_and_exterior_geometry(spatial_order):
    grid = GridSpec(shape=(12, 15), spacing_m=(0.001, 0.0013), origin_m=(-0.006, -0.01))
    geometry = GeometrySpec(
        tx_pos_m=[[-0.008, -0.011]], rx_pos_m=[[0.009, 0.012], [0.002, 0.014]]
    )
    model = EikonalForward(grid, geometry, spatial_order=spatial_order)
    expected = (
        np.linalg.norm(geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None, :], axis=-1)
        / 1500
    )
    np.testing.assert_allclose(
        model.forward(np.full(grid.shape, 1 / 1500)), expected.ravel(), rtol=1e-13
    )


@pytest.mark.parametrize("spatial_order", [1, 2])
def test_discrete_jacobian_and_adjoint(spatial_order):
    rng = np.random.default_rng(83)
    grid = GridSpec(shape=(11, 13), spacing_m=(0.001, 0.0008))
    geom = GeometrySpec(
        tx_pos_m=[[-0.001, 0.0012], [0.0032, -0.002]],
        rx_pos_m=[[0.0092, 0.01], [0.011, 0.0082]],
    )
    model = EikonalForward(grid, geom, spatial_order=spatial_order)
    s = (1 + 0.02 * rng.random(grid.shape)) / 1500
    lin = model.linearize(s)
    v = rng.normal(size=grid.shape) / 1500
    data = rng.normal(size=4)
    assert adjoint_error(lin.jacobian, v, data) < 2e-12
    eps = 1e-6
    numerical = (model.forward(s + eps * v) - model.forward(s - eps * v)) / (2 * eps)
    np.testing.assert_allclose(
        lin.jacobian.forward(v), numerical, rtol=3e-5, atol=1e-12
    )


def test_refraction_converges_to_snell_layered_solution():
    # Crossing a flat interface: minimize the two-segment Fermat travel time.
    exact = minimize_scalar(
        lambda x: np.hypot(0.025, x) / 1450 + np.hypot(0.025, 0.07 - x) / 1750,
        bounds=(0, 0.07),
        method="bounded",
        options={"xatol": 1e-14},
    ).fun
    errors = []
    for n in (41, 81):
        h = 0.1 / (n - 1)
        yy = np.arange(n)[:, None] * h
        slowness = np.broadcast_to(
            np.where(yy < 0.05, 1 / 1450, 1 / 1750), (n, n)
        ).copy()
        source = np.array([0.025, 0.015]) / h
        tape = fast_march(slowness, (h, h), source)
        receiver = np.round(np.array([0.075, 0.085]) / h).astype(int)
        actual = tape.times.reshape(n, n)[tuple(receiver)]
        errors.append(abs(actual - exact))
    assert errors[1] < errors[0]
    assert errors[1] / exact < 0.035


def test_bent_prediction_is_not_fixed_straight_integral():
    grid = GridSpec(shape=(41, 41), spacing_m=(0.001, 0.001))
    geom = GeometrySpec(tx_pos_m=[[0.0205, -0.001]], rx_pos_m=[[0.0205, 0.043]])
    yy, xx = np.indices(grid.shape)
    s = np.full(grid.shape, 1 / 1500)
    s[(yy - 20) ** 2 + (xx - 20) ** 2 < 8**2] = 1 / 1100
    model = EikonalForward(grid, geom)
    bent_delay = model.forward(s)[0] - 0.044 / 1500
    straight_delay = np.sum(s[20] - 1 / 1500) * 0.001
    assert 0 < bent_delay < 0.95 * straight_delay


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_rejects_nonphysical_slowness(bad):
    with pytest.raises(ValueError):
        fast_march(np.full((4, 5), bad), (1, 1), np.array([1.2, 1.4]))


def test_second_order_improves_uncalibrated_point_source_and_converges():
    errors = {1: [], 2: []}
    exact = np.hypot(0.06, 0.06) / 1500
    for n in (41, 81, 161):
        h = 0.1 / (n - 1)
        for order in (1, 2):
            tape = fast_march(
                np.full((n, n), 1 / 1500),
                (h, h),
                np.array([0.02, 0.02]) / h,
                spatial_order=order,
            )
            receiver = np.rint(np.array([0.08, 0.08]) / h).astype(int)
            errors[order].append(abs(tape.times.reshape(n, n)[tuple(receiver)] - exact))
    assert np.all(np.diff(errors[2]) < 0)
    # The point-source singularity still limits global order; this is not an
    # assertion of quadratic convergence or a 30 ns accuracy certificate.
    assert errors[2][-1] < 0.25 * errors[1][-1]
    assert errors[2][-1] < 150e-9


def test_second_order_tape_is_causal_and_keeps_negative_far_weights():
    rng = np.random.default_rng(441)
    s = 1 / (1500 + rng.uniform(-20, 20, (31, 29)))
    tape = fast_march(s, (0.001, 0.0013), np.array([3.2, 2.4]), spatial_order=2)
    rank = np.empty(s.size, int)
    rank[tape.order] = np.arange(s.size)
    node, slot = np.nonzero(tape.parents >= 0)
    assert np.all(rank[tape.parents[node, slot]] < rank[node])
    assert np.any(tape.weights < 0)
    np.testing.assert_allclose(tape.weights.sum(axis=1)[node], 1, atol=1e-13)


@pytest.mark.parametrize("bad", [0, 3, True, "2", 2.0])
def test_rejects_invalid_order(bad):
    with pytest.raises(ValueError, match="spatial_order"):
        fast_march(np.ones((4, 5)), (1, 1), np.array([1.2, 1.4]), spatial_order=bad)


def test_smooth_linear_speed_gradient_matches_analytic_hyperbolic_metric():
    # For c(y)=c0+g*y, the eikonal metric is a scaled hyperbolic half-plane.
    source, receiver = np.array([0.02, 0.02]), np.array([0.08, 0.08])
    c0, gradient = 1400.0, 2000.0
    cs, cr = c0 + gradient * source[0], c0 + gradient * receiver[0]
    exact = (
        np.arccosh(1 + gradient**2 * np.sum((receiver - source) ** 2) / (2 * cs * cr))
        / gradient
    )
    errors = {1: [], 2: []}
    for n in (41, 81, 161):
        h = 0.1 / (n - 1)
        speed = np.broadcast_to(c0 + gradient * np.arange(n)[:, None] * h, (n, n))
        for order in (1, 2):
            tape = fast_march(1 / speed, (h, h), source / h, spatial_order=order)
            index = tuple(np.rint(receiver / h).astype(int))
            errors[order].append(abs(tape.times.reshape(n, n)[index] - exact))
    assert np.all(np.diff(errors[2]) < 0)
    assert errors[2][-1] < 0.3 * errors[1][-1]
    assert errors[2][-1] < 200e-9
