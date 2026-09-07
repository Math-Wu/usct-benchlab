import numpy as np
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators import adjoint_error
from usctbench.operators.forward.ray_born import RayBornOperator


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
