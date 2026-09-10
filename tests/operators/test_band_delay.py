import numpy as np
import pytest

from usctbench.core.schema import GridSpec, GeometrySpec
from usctbench.operators.forward.band_delay import (
    BandCorrelationDelay,
    BandDelayLinearization,
    FiniteFrequencyTravelTimeForward,
)
from usctbench.operators.forward.ray_born import RayBornForward, RayBornOperator


def observation():
    f = np.linspace(100e3, 400e3, 19)
    bands = np.array([np.ones(len(f)), np.linspace(0.2, 1, len(f))])
    water = np.ones((len(f), 2, 3), complex)
    return BandCorrelationDelay(f, water, bands, -6e-6, 6e-6)


def test_finite_delay_sign_and_invariance_to_positive_channel_gain():
    op = observation()
    delays = np.array([[2e-6, -1e-6, 0], [0.5e-6, -2.5e-6, 4e-6]])
    ratio = np.exp(1j * op.omega[:, None, None] * delays)
    fit = op.linearize(ratio)
    assert fit.valid.all()
    np.testing.assert_allclose(
        fit.value, np.broadcast_to(delays, fit.value.shape), atol=1e-15
    )
    np.testing.assert_allclose(
        op.linearize(ratio * np.arange(1, 7).reshape(2, 3)).value, fit.value, atol=1e-15
    )
    np.testing.assert_allclose(fit.coherence, 1, atol=1e-12)


def test_peak_jacobian_and_real_complex_adjoint():
    op = observation()
    rng = np.random.default_rng(912)
    ratio = (1 + rng.normal(size=op.data_shape) * 0.1) * np.exp(
        1j * op.omega[:, None, None] * 1e-6
    )
    perturbation = rng.normal(size=ratio.shape) + 1j * rng.normal(size=ratio.shape)
    lin = op.linearize(ratio)
    epsilon = 1e-5
    fd = (
        op.linearize(ratio + epsilon * perturbation).value
        - op.linearize(ratio - epsilon * perturbation).value
    ) / (2 * epsilon)
    np.testing.assert_allclose(lin.forward(perturbation), fd, rtol=1e-6, atol=1e-15)
    y = rng.normal(size=lin.value.shape)
    np.testing.assert_allclose(
        np.vdot(lin.forward(perturbation), y).real,
        np.vdot(perturbation, lin.adjoint(y)).real,
        rtol=1e-12,
    )


def test_reported_local_certificate_matches_independent_spectral_derivatives():
    op = observation()
    rng = np.random.default_rng(216)
    ratio = (1 + 0.1 * rng.normal(size=op.data_shape)) * np.exp(
        1j * (op.omega[:, None, None] * 0.7e-6 + rng.normal(size=op.data_shape) * 0.05)
    )
    fit = op.linearize(ratio)
    assert fit.valid.all()
    omega = op.omega[None, :, None, None]
    rotated = op.power * ratio[None] * np.exp(-1j * omega * fit.value[:, None])
    slope = np.sum(omega * rotated.imag, axis=1)
    curvature = np.sum(omega**2 * rotated.real, axis=1)
    third_bound = np.sum(op.power * np.abs(ratio)[None] * omega**3, axis=1)
    np.testing.assert_allclose(
        fit.stationarity_error_s, np.abs(slope / curvature), rtol=1e-12, atol=1e-21
    )
    np.testing.assert_allclose(
        fit.local_concavity_margin, curvature - 2 * op.step * third_bound, rtol=1e-12
    )
    assert np.all(fit.stationarity_error_s < 1e-6 * op.step)
    assert np.all(fit.local_concavity_margin > 0)
    for offset in np.linspace(-2 * op.step, 2 * op.step, 17):
        phase = np.exp(-1j * omega * (fit.value[:, None] + offset))
        nearby_curvature = np.sum(
            omega**2 * (op.power * ratio[None] * phase).real, axis=1
        )
        assert np.all(nearby_curvature >= fit.local_concavity_margin)

    perturbation = rng.normal(size=ratio.shape) + 1j * rng.normal(size=ratio.shape)
    epsilon = 1e-5
    plus, minus = op.linearize(ratio + epsilon * perturbation), op.linearize(
        ratio - epsilon * perturbation
    )
    assert plus.valid.all() and minus.valid.all()
    np.testing.assert_allclose(
        fit.forward(perturbation),
        (plus.value - minus.value) / (2 * epsilon),
        rtol=1e-6,
        atol=1e-15,
    )


@pytest.mark.parametrize("oversample", [32, 128])
@pytest.mark.parametrize("imbalance", [0, -1e-8, 1e-8])
def test_unresolved_two_path_peaks_are_invalid_despite_high_sampled_gap(
    oversample, imbalance
):
    frequencies = np.array([100e3, 200e3, 400e3])
    angle = 0.01
    amplitude = 16 * np.cos(angle) * np.cos(2 * angle)
    water = np.sqrt([np.sqrt(2) * amplitude, 1, 1])[:, None, None]
    op = BandCorrelationDelay(
        frequencies, water, np.ones((1, 3)), -3e-6, 3e-6, oversample=oversample
    )
    path = np.exp(1j * op.omega * 1.25e-6)[:, None, None]
    ratio = (1 + imbalance) * path + (1 - imbalance) * path.conj()
    fit = op.linearize(ratio)
    # At zero imbalance C is proportional to A*cos(x)-cos(4*x), with
    # equal maxima at x=+/-angle, both inside the excluded competitor radius.
    peak_separation = 2 * angle / (2 * np.pi * frequencies[0])
    assert peak_separation < 2 * op.step
    assert np.all(fit.coherence > 0.88)
    assert np.all(fit.peak_gap > 0.99)
    assert np.all(fit.local_concavity_margin < 0)
    assert not fit.valid.any()
    assert np.isnan(fit.value).all()
    np.testing.assert_array_equal(fit.coefficient, 0)
    np.testing.assert_array_equal(fit.forward(np.full_like(ratio, np.nan)), 0)
    np.testing.assert_array_equal(fit.adjoint(np.full_like(fit.value, np.nan)), 0)


@pytest.mark.parametrize("oversample", [8, 32])
def test_flat_unique_peak_finishes_refinement_but_lacks_concavity_certificate(
    oversample,
):
    frequencies = np.array([100e3, 150e3, 400e3])
    op = BandCorrelationDelay(
        frequencies,
        np.ones((3, 1, 1)),
        np.ones((1, 3)),
        -3e-6,
        3e-6,
        oversample=oversample,
    )
    delta = 32 * 1.0001e-4 / (1 - 1.0001e-4)
    tau = op.lags[len(op.lags) // 2] + 0.49 * op.step
    ratio = np.array([16 + delta, 0, -1])[:, None, None] * np.exp(
        1j * op.omega[:, None, None] * tau
    )
    fit = op.linearize(ratio)
    # Six Newton steps leave a 5.705 ns correction for oversample=8.
    # Finishing refinement does not turn this conservative local test into a
    # necessary condition: the true peak is unique but still uncertified.
    assert np.all(fit.stationarity_error_s < 1e-15)
    assert np.all(fit.local_concavity_margin < 0)
    assert not fit.valid.any()
    assert np.isnan(fit.value).all()
    np.testing.assert_array_equal(fit.coefficient, 0)


def test_bound_clipping_never_exposes_an_implicit_peak_derivative():
    frequencies = np.linspace(100e3, 400e3, 19)
    op = BandCorrelationDelay(
        frequencies, np.ones((19, 1, 1)), np.ones((1, 19)), -3e-6, 3e-6
    )
    ratio = np.exp(1j * op.omega[:, None, None] * 3.2e-6)
    fit = op.linearize(ratio)
    assert np.all(fit.local_concavity_margin > 0)
    assert np.all(fit.stationarity_error_s > 1e-6 * op.step)
    assert not fit.valid.any()
    assert np.isnan(fit.value).all()
    np.testing.assert_array_equal(fit.coefficient, 0)


def test_local_concavity_does_not_claim_uniqueness_of_distant_peaks():
    frequencies = np.array([200e3, 250e3, 300e3])
    op = BandCorrelationDelay(
        frequencies, np.ones((3, 1, 1)), np.ones((1, 3)), -6e-6, 6e-6
    )
    fit = op.linearize(-np.ones((3, 1, 1), complex))
    assert fit.valid.all()
    assert np.all(fit.local_concavity_margin > 0)
    assert np.all(fit.peak_gap < 0.002)
    assert np.all(np.abs(fit.value) > 1.9e-6)


def test_optional_diagnostics_preserve_five_argument_composition():
    op = observation()
    fit = op.linearize(np.ones(op.data_shape, complex))
    composed = BandDelayLinearization(
        fit.value, fit.valid, fit.coherence, fit.peak_gap, fit.coefficient
    )
    assert composed.stationarity_error_s is None
    assert composed.local_concavity_margin is None
    np.testing.assert_array_equal(
        composed.forward(1j * np.ones(op.data_shape)),
        fit.forward(1j * np.ones(op.data_shape)),
    )


def test_bounds_bad_channels_and_receiver_locality():
    op = observation()
    ratio = np.ones(op.data_shape, complex)
    initial = op.linearize(ratio)
    ratio[:, 0, 0] *= np.exp(1j * op.omega * 1e-6)
    changed = op.linearize(ratio)
    np.testing.assert_array_equal(initial.value[:, 1], changed.value[:, 1])
    assert np.all(changed.value[:, 0, 0] > 0)
    ratio[:, 1, 1] = 0
    invalid = op.linearize(ratio)
    assert not np.any(invalid.valid[:, 1, 1])
    assert np.all(np.isnan(invalid.value[:, 1, 1]))
    assert np.isfinite(invalid.adjoint(np.where(invalid.valid, 1, np.nan))).all()
    with pytest.raises(ValueError, match="ordered"):
        BandCorrelationDelay(
            op.frequencies_hz, ratio, np.ones((1, len(op.omega))), 1, 0
        )


def test_sparse_frequency_grid_creates_late_sensitivity_replicas():
    # A smooth pulse has negligible sensitivity this far outside its support.
    # Coarse spectral sampling creates a replica at 1 / df inside this interval.
    norms = {}
    for count in (15, 61):
        frequencies = np.linspace(200e3, 800e3, count)
        water = np.exp(-0.5 * ((frequencies - 500e3) / 100e3) ** 2)[:, None, None]
        op = BandCorrelationDelay(frequencies, water, np.ones((1, count)), -6e-6, 6e-6)
        lin = op.linearize(np.ones_like(water, dtype=complex))
        kernel = [
            lin.forward(np.exp(2j * np.pi * frequencies[:, None, None] * lag))
            for lag in np.linspace(20e-6, 27e-6, 101)
        ]
        norms[count] = np.linalg.norm(kernel)
    assert norms[61] < norms[15] * 1e-4


@pytest.mark.parametrize("device", [None, 0])
def test_nonlinear_full_chain_at_heterogeneous_background(device):
    if device is not None:
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    grid = GridSpec(shape=(10, 10), spacing_m=(0.001, 0.001), origin_m=(-0.005, -0.005))
    geom = GeometrySpec(
        tx_pos_m=[[-0.008, 0], [0, -0.008]], rx_pos_m=[[0.008, 0], [0, 0.008]]
    )
    f = np.linspace(100e3, 300e3, 9)
    water = RayBornOperator(grid, geom, f).background_data()
    obs = BandCorrelationDelay(f, water, np.ones((1, len(f))), -3e-6, 3e-6)
    pressure = RayBornForward(
        grid,
        geom,
        f,
        green_backend="volume_integral",
        green_solver_rtol=1e-12,
        max_cache_bytes=2**24,
        green_device=device,
    )
    forward = FiniteFrequencyTravelTimeForward(pressure, obs, water)
    yy, xx = np.indices(grid.shape)
    speed = 1500 + 30 * np.exp(-((yy - 4) ** 2 + (xx - 5) ** 2) / 4)
    model = 1 / speed**2
    lin = forward.linearize(model)
    rng = np.random.default_rng(85)
    dm = rng.normal(size=grid.shape) * 1e-9
    epsilon = 1e-2
    fd = (
        forward.forward(model + epsilon * dm) - forward.forward(model - epsilon * dm)
    ) / (2 * epsilon)
    np.testing.assert_allclose(lin.jacobian.forward(dm), fd, rtol=1e-4, atol=1e-15)
    y = rng.normal(size=lin.value.shape)
    np.testing.assert_allclose(
        np.vdot(lin.jacobian.forward(dm), y).real,
        np.vdot(dm, lin.jacobian.adjoint(y)).real,
        rtol=1e-12,
    )
    assert forward.background_builds == 3
    if device is not None:
        cpu = RayBornForward(
            grid, geom, f, green_backend="volume_integral", green_solver_rtol=1e-12
        )
        cpu_linearization = cpu.linearize(model)
        np.testing.assert_allclose(
            pressure.forward(model), cpu_linearization.value, rtol=1e-9, atol=1e-12
        )
        gpu_linearization = pressure.linearize(model)
        np.testing.assert_allclose(
            gpu_linearization.jacobian.forward(dm),
            cpu_linearization.jacobian.forward(dm),
            rtol=1e-8,
            atol=1e-14,
        )


def test_water_initialized_nonlinear_inversion_decreases_travel_time_misfit():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import AlgorithmConfig, MeasurementSpec, USCTCase
    from usctbench.solvers.nonlinear import nonlinear_least_squares

    grid = GridSpec(shape=(12, 12), spacing_m=(0.001, 0.001), origin_m=(-0.006, -0.006))
    angles = np.arange(8) * 2 * np.pi / 8
    positions = 0.01 * np.column_stack([np.sin(angles), np.cos(angles)])
    geom = GeometrySpec(tx_pos_m=positions, rx_pos_m=positions)
    f = np.linspace(100e3, 350e3, 9)
    water = RayBornOperator(grid, geom, f).background_data()
    active = np.linalg.norm(positions[:, None] - positions[None], axis=-1) > 0.01
    obs = BandCorrelationDelay(
        f, np.where(active[None], water, 0), np.ones((1, len(f))), -3e-6, 3e-6
    )
    pressure = RayBornForward(
        grid,
        geom,
        f,
        green_backend="volume_integral",
        green_solver_rtol=1e-10,
        max_cache_bytes=2**24,
    )
    forward = FiniteFrequencyTravelTimeForward(pressure, obs, water)
    yy, xx = np.indices(grid.shape)
    truth = 1500 + 20 * np.exp(-((yy - 5) ** 2 + (xx - 6) ** 2) / 5)
    observed = forward.forward(1 / truth**2)
    case = USCTCase(
        case_id="band_sanity",
        grid=grid,
        geometry=geom,
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed[0]),
    )
    config = AlgorithmConfig(
        parameters={
            "stopping": {"max_iterations": 4, "restore_best_validation": False},
            "evaluation": {"receiver_fraction": 0.125, "seed": 42},
        }
    )
    control = InversionControl(
        case, config, observed, default_iterations=4, valid_mask=active[None]
    )
    initial = np.full(grid.shape, 1 / 1500**2)
    result, metrics = nonlinear_least_squares(
        forward,
        control,
        initial=initial,
        bounds=(1400, 1600),
        inner_iterations=8,
        damping=0.01,
    )
    assert metrics["data_residual_reduction"] > 0.8
    assert np.linalg.norm(1 / np.sqrt(result) - truth) < np.linalg.norm(1500 - truth)
    assert metrics["nonlinear_background_updates"] > 0
    accepted = [r["objective"] for r in metrics["line_search_history"] if r["accepted"]]
    assert np.all(np.diff(accepted) < 0)
