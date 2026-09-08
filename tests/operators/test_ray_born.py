import numpy as np
import pytest
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators import adjoint_error
from usctbench.operators.forward.ray_born import RayBornOperator
from usctbench.operators.forward.ray_born import RayBornForward
from usctbench.operators.forward.volume_integral import VolumeIntegralGreen


def test_volume_solver_checks_deadline_inside_source_solve():
    from usctbench.core.stopping import BudgetExhausted

    op = setup_operator(True)
    calls = []

    def deadline():
        calls.append(1)
        if len(calls) == 3:
            raise BudgetExhausted("time_budget")

    solver = VolumeIntegralGreen(
        op.grid,
        180e3,
        op.background,
        1500,
        rtol=1e-12,
        maxiter=20,
        budget_check=deadline,
    )
    with pytest.raises(BudgetExhausted, match="time_budget"):
        solver.fields(op.green_fields(0))
    assert len(calls) == 3


def test_conservative_transport_converges_for_radial_energy_flux():
    from usctbench.operators._marching import conservative_transport

    errors = []
    for n in (32, 64):
        h = 1 / n
        y, x = np.mgrid[:n, :n] * h
        distance = np.hypot(y + 0.2, x - 0.5)
        initial = -0.5 * np.log(distance.ravel())
        parents = np.zeros((n * n, 2), dtype=int)
        parents[:n] = -1
        actual = conservative_transport(
            np.argsort(distance.ravel()),
            parents,
            distance.ravel(),
            (n, n),
            np.array([h, h]),
            initial,
        )
        intensity = np.exp(2 * actual).reshape(n, n)
        errors.append(np.mean(np.abs(intensity[3:] * distance[3:] - 1)))
    assert errors[1] < 0.6 * errors[0]
    assert errors[1] < 0.03


def test_volume_convolution_is_linear_not_periodic():
    op = setup_operator()
    solver = VolumeIntegralGreen(
        op.grid, 180e3, op.background, 1500, rtol=1e-10, maxiter=20
    )
    impulse = np.zeros(op.grid.shape)
    impulse[-1, -1] = 1
    result = solver.convolve(impulse)
    points = np.indices(op.grid.shape).transpose(1, 2, 0)
    distance = np.linalg.norm(
        (points - np.array(op.grid.shape) + 1) * op.grid.spacing_m, axis=-1
    )
    exact = 0.25j * hankel1(
        0, 2 * np.pi * 180e3 / 1500 * np.maximum(distance, op.source_radius_m)
    )
    np.testing.assert_allclose(result[:-1], exact[:-1], atol=1e-14)


def test_volume_born_derivative_matches_nonlinear_finite_difference():
    base = setup_operator(True)
    forward = RayBornForward(
        base.grid,
        base.geometry,
        base.frequencies_hz,
        source_spectrum=base.source_spectrum,
        green_backend="volume_integral",
        green_solver_rtol=1e-12,
        max_cache_bytes=10**6,
    )
    model = 1 / base.background**2
    lin = forward.linearize(model)
    rng = np.random.default_rng(53)
    dm = rng.normal(size=model.shape) * 1e-9
    epsilon = 0.01
    numerical = (
        forward.forward(model + epsilon * dm) - forward.forward(model - epsilon * dm)
    ) / (2 * epsilon)
    np.testing.assert_allclose(
        lin.jacobian.forward(dm), numerical, rtol=2e-5, atol=1e-10
    )
    complex_data = rng.normal(size=lin.value.shape) + 1j * rng.normal(
        size=lin.value.shape
    )
    assert adjoint_error(lin.jacobian, dm, complex_data) < 1e-12
    assert lin.derivative_kind == "discrete_volume_integral_born_derivative"
    assert forward.green_solves > 0


def setup_operator(heterogeneous=False, cache=0):
    grid = GridSpec(shape=(9, 10), spacing_m=(0.001, 0.001), origin_m=(-0.0045, -0.005))
    geom = GeometrySpec(
        tx_pos_m=[[-0.009, -0.008], [0.009, -0.003]],
        rx_pos_m=[[0.005, 0.009], [-0.006, 0.007]],
    )
    background = np.full(grid.shape, 1500.0)
    if heterogeneous:
        yy, xx = np.indices(grid.shape)
        background += 35 * np.exp(-((yy - 4) ** 2 + (xx - 5) ** 2) / 8)
    return RayBornOperator(
        grid,
        geom,
        np.array([180e3, 240e3]),
        background_sound_speed_mps=background,
        source_spectrum=np.array([[1 + 0.2j, 0.7j], [0.8, -0.4j]]),
        max_cache_bytes=cache,
    )


def test_complex_real_adjoint_and_no_dense_jacobian():
    rng = np.random.default_rng(21)
    for heterogeneous in (False, True):
        op = setup_operator(heterogeneous)
        image = rng.normal(size=op.grid.shape) * 1e-9
        data = rng.normal(size=op.data_shape) + 1j * rng.normal(size=op.data_shape)
        assert adjoint_error(op, image, data) < 2e-12
        assert not op._cache
        assert op.green_method == (
            "eikonal_wkb_transport" if heterogeneous else "analytic_hankel"
        )


def test_normal_diagonal_matches_explicit_basis_columns():
    op = setup_operator()
    weights = np.arange(np.prod(op.data_shape)).reshape(op.data_shape) / 10
    diagonal = op.normal_diagonal(weights)
    for index in ((0, 0), (2, 7), (8, 9)):
        basis = np.zeros(op.grid.shape)
        basis[index] = 1
        expected = np.sum(weights * np.abs(op.forward(basis)) ** 2)
        np.testing.assert_allclose(diagonal[index], expected, rtol=1e-12)


def test_homogeneous_born_matches_independent_point_scatterer():
    op = setup_operator()
    point = (2, 7)
    dm = np.zeros(op.grid.shape)
    dm[point] = 1e-9
    coordinate = (
        np.array(op.grid.origin_m) + (np.array(point) + 0.5) * op.grid.spacing_m
    )
    result = op.forward(dm)
    for index, frequency in enumerate(op.frequencies_hz):
        k = 2 * np.pi * frequency / 1500
        gs = 0.25j * hankel1(
            0, k * np.linalg.norm(op.geometry.tx_pos_m - coordinate, axis=1)
        )
        gr = 0.25j * hankel1(
            0, k * np.linalg.norm(op.geometry.rx_pos_m - coordinate, axis=1)
        )
        expected = (
            (2 * np.pi * frequency) ** 2
            * op.area
            * dm[point]
            * gs[:, None]
            * gr[None, :]
            * op.source_spectrum[index, :, None]
        )
        np.testing.assert_allclose(result[index], expected, rtol=1e-13)
    # The scatterer is off the transmitter-receiver line, yet remains observable.
    assert np.all(np.abs(result) > 0)


def test_heterogeneous_background_changes_phase_and_spreading():
    uniform = setup_operator()
    refracted = setup_operator(True)
    assert np.max(np.abs(refracted.delay)) > 1e-8
    assert np.max(np.abs(refracted.log_amplitude_ratio)) > 1e-3
    assert not np.allclose(uniform.background_data(), refracted.background_data())
    np.testing.assert_allclose(
        refracted.predict(refracted.background), refracted.background_data()
    )


def test_born_error_is_quadratic_against_multiple_scattering_solve():
    # Independent discrete Lippmann-Schwinger solve, including all scattering orders.
    op = setup_operator()
    frequency = op.frequencies_hz[0]
    omega = 2 * np.pi * frequency
    points = np.indices(op.grid.shape).reshape(2, -1).T
    points = np.array(op.grid.origin_m) + (points + 0.5) * op.grid.spacing_m
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    gpp = 0.25j * hankel1(0, omega / 1500 * np.maximum(distances, op.source_radius_m))
    source_distances = np.linalg.norm(points - op.geometry.tx_pos_m[0], axis=-1)
    incident = (
        0.25j * hankel1(0, omega / 1500 * source_distances) * op.source_spectrum[0, 0]
    )
    receiver_distances = np.linalg.norm(
        op.geometry.rx_pos_m[:, None] - points[None], axis=-1
    )
    receiver_green = 0.25j * hankel1(0, omega / 1500 * receiver_distances)
    errors = []
    for scale in (1.0, 0.5):
        dm = np.full(op.grid.shape, 2e-10 * scale)
        potential = omega**2 * op.area * dm.ravel()
        full_field = np.linalg.solve(np.eye(op.n_pixels) - gpp * potential, incident)
        scattered = receiver_green @ (potential * full_field)
        born = op.forward(dm)[0, 0]
        errors.append(np.linalg.norm(scattered - born))
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_green_cache_budget_does_not_change_results():
    uncached = setup_operator(cache=0)
    cached = setup_operator(cache=100000)
    image = np.full(uncached.grid.shape, 1e-10)
    np.testing.assert_allclose(cached.forward(image), uncached.forward(image))
    assert sum(g.nbytes for g in cached._cache.values()) <= cached.max_cache_bytes
