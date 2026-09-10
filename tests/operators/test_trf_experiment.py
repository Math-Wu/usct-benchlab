"""Reference optimizer controls, separate from physical-image acceptance."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import least_squares, minimize

from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GridSpec,
    GeometrySpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization

pytestmark = pytest.mark.skipif(
    "callback" not in inspect.signature(least_squares).parameters,
    reason="experimental TRF requires SciPy >= 1.16",
)


def load_solver_module():
    path = Path(__file__).resolve().parents[2] / "scripts/_finite_frequency_trf.py"
    spec = importlib.util.spec_from_file_location("finite_frequency_trf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solver():
    return load_solver_module().solve_trust_region


def test_tv_loss_preserves_data_rows_and_has_consistent_derivatives():
    loss = load_solver_module().SmoothTVLoss(2, 0.2)
    z = np.array([0.2, 4.0, 1e-8, 0.3, 100.0])
    rho = loss(z)
    np.testing.assert_array_equal(rho[:, :2], np.array([z[:2], [1.0, 1.0], [0.0, 0.0]]))
    h = z * 1e-5
    difference = (loss(z + h) - loss(z - h)) / (2 * h)
    np.testing.assert_allclose(difference[0], rho[1], rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(difference[1], rho[2], rtol=1e-3, atol=1e-8)
    np.testing.assert_array_equal(loss(np.zeros(5))[0], 0)
    assert np.all(rho[0, 2:] < z[2:])


class DiagonalObservation:
    """Algebraic nonlinear control, deliberately NOT an ultrasound forward model."""

    def __init__(self):
        self.background_builds = 0
        self.eikonal_solves = 0

    def linearize(self, model):
        self.background_builds += 1
        scale = 1 / 1500**2
        derivative = 4e-6 * model / scale**2

        class Jacobian:
            def forward(self, dm):
                return (derivative * dm)[None]

            def adjoint(self, y):
                return derivative * y[0]

        return Linearization(
            (2e-6 * (model / scale) ** 2)[None],
            Jacobian(),
            derivative_kind="diagonal_control",
        )


def setup(max_elapsed_s=None):
    grid = GridSpec(shape=(4, 4), spacing_m=(0.001, 0.001))
    positions = [[0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01]]
    geometry = GeometrySpec(tx_pos_m=positions, rx_pos_m=positions)
    rng = np.random.default_rng(41)
    model = 1 / rng.uniform(1440, 1520, grid.shape) ** 2
    observed = DiagonalObservation().linearize(model).value
    case = USCTCase(
        case_id="trf_control",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed[0]),
    )
    config = AlgorithmConfig(
        parameters={
            "stopping": {
                "max_iterations": 8,
                "restore_best_validation": False,
                "max_elapsed_s": max_elapsed_s,
            },
            "evaluation": {
                "receiver_fraction": 0.25,
                "seed": 42,
                "exclude_reciprocal": True,
            },
        }
    )
    return case, config, observed, model


def test_trf_converges_and_ignores_heldout_updates():
    solve = load_solver()
    case, config, observed, truth = setup()
    initial = np.full(case.grid.shape, 1 / 1500**2)
    outputs = []
    for corrupt in (False, True):
        control = InversionControl(case, config, observed.copy(), default_iterations=8)
        if corrupt:
            control.observed[~control.split.train] += 1e5
        result, metrics = solve(
            DiagonalObservation(),
            control,
            initial=initial,
            bounds=(1400, 1600),
            damping=0,
            regularization="laplacian",
            inner_iterations=12,
        )
        outputs.append(result)
        assert metrics["data_relative_residual"] < 1e-5
        assert metrics["optimizer_terminal"]["optimality"] < 1e-5
        np.testing.assert_allclose(
            result[control.split.train[0]], truth[control.split.train[0]], rtol=1e-4
        )
    np.testing.assert_array_equal(outputs[0], outputs[1])


def test_trf_budget_returns_atomic_fallback():
    case, config, observed, _ = setup(max_elapsed_s=1e-9)
    initial = np.full(case.grid.shape, 1 / 1500**2)
    control = InversionControl(case, config, observed, default_iterations=8)
    result, metrics = load_solver()(
        DiagonalObservation(),
        control,
        initial=initial,
        bounds=(1400, 1600),
        damping=0,
        regularization="laplacian",
        inner_iterations=12,
    )
    np.testing.assert_array_equal(result, initial)
    assert metrics["stopping"]["completed_iterations"] == 0
    assert metrics["optimizer_terminal"] is None


def test_tv_trf_matches_independent_dense_objective_and_reports_actual_cost():
    from usctbench.operators.model_space import SpatialGradient

    case, config, observed, _ = setup()
    config.parameters["stopping"].update(
        max_iterations=60,
        objective_rtol=None,
        update_rtol=None,
    )
    control = InversionControl(case, config, observed, default_iterations=60)
    scale, epsilon, damping = 1 / 1500**2, 2 * 5 / 1500**3, 25.0
    initial = np.full(case.grid.shape, scale)
    result, metrics = load_solver()(
        DiagonalObservation(),
        control,
        initial=initial,
        bounds=(1400, 1600),
        damping=damping,
        regularization=SpatialGradient(case.grid, 0.002),
        inner_iterations=32,
        tv_transition=epsilon,
    )

    def reference(x):
        x = x.reshape(case.grid.shape)
        residual = 2e-6 * (1 + x) ** 2 - observed[0]
        precision = control.precision[0]
        value = 0.5 * np.sum(precision * residual**2)
        grad = precision * residual * 4e-6 * (1 + x)
        m = scale * x
        # Independent edge differences/divergence, not the production sparse L.
        for axis in (0, 1):
            edge = 2 * np.diff(m, axis=axis)
            root = np.sqrt(1 + (edge / epsilon) ** 2)
            value += damping * np.sum(edge**2 / (root + 1))
            force = scale * 2 * damping * edge / root
            if axis == 0:
                grad[:-1] -= force
                grad[1:] += force
            else:
                grad[:, :-1] -= force
                grad[:, 1:] += force
        return float(value * 1e12), grad.ravel() * 1e12

    bounds = [(1 / 1600**2 / scale - 1, 1 / 1400**2 / scale - 1)] * initial.size
    dense = minimize(
        reference,
        np.zeros(initial.size),
        jac=True,
        bounds=bounds,
        method="L-BFGS-B",
        options={"gtol": 1e-8, "ftol": 1e-14, "maxiter": 500},
    )
    assert dense.success, dense.message
    x = (result / scale - 1).ravel()
    np.testing.assert_allclose(reference(x)[0], dense.fun, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(x, dense.x, atol=2e-5)
    history = metrics["iteration_history"]
    assert all(b["objective"] <= a["objective"] for a, b in zip(history, history[1:]))
    assert history[-1]["objective"] == pytest.approx(reference(x)[0] / 1e12, rel=1e-12)
    assert metrics["optimizer_settings"]["data_loss"] == "quadratic"
    assert (
        metrics["optimizer_settings"]["regularization_loss"] == "smooth_anisotropic_tv"
    )


def test_zero_inner_step_is_not_nonlinear_stationarity(monkeypatch):
    import usctbench.solvers.nonlinear as nonlinear

    case, config, observed, _ = setup()
    initial = np.full(case.grid.shape, 1 / 1500**2)
    control = InversionControl(case, config, observed, default_iterations=8)
    monkeypatch.setattr(
        nonlinear, "normal_step", lambda *a, **kw: np.zeros_like(initial)
    )
    result, metrics = nonlinear.nonlinear_least_squares(
        DiagonalObservation(),
        control,
        initial=initial,
        bounds=(1400, 1600),
        damping=0,
    )
    assert not np.array_equal(result, initial)
    assert metrics["data_residual_reduction"] > 0
    assert metrics["stop_reason"] not in {"stationary_update", "stationary_gradient"}
    assert any(
        x["proposal"] == "projected_gradient" and x["accepted"]
        for x in metrics["line_search_history"]
    )


@pytest.mark.parametrize("optimizer", ["trf", "gauss_newton"])
def test_solver_recovers_identifiable_wave_equation_model(optimizer):
    """An overdetermined, noiseless physics gate, not a clinical-quality claim."""
    from usctbench.operators.forward.band_delay import (
        BandCorrelationDelay,
        FiniteFrequencyTravelTimeForward,
    )
    from usctbench.operators.forward.ray_born import RayBornOperator, RayBornForward
    from usctbench.operators.model_space import BilinearBasis

    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_finite_frequency_traveltime.py"
    )
    spec = importlib.util.spec_from_file_location("trf_physics_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grid = GridSpec(shape=(12, 12), spacing_m=(0.001, 0.001), origin_m=(-0.006, -0.006))
    angles = np.arange(12) * 2 * np.pi / 12
    positions = 0.01 * np.column_stack([np.sin(angles), np.cos(angles)])
    geometry = GeometrySpec(tx_pos_m=positions, rx_pos_m=positions)
    frequencies = np.linspace(100e3, 350e3, 13)
    water = RayBornOperator(grid, geometry, frequencies).background_data()
    valid = np.linalg.norm(positions[:, None] - positions[None], axis=-1) > 0.01
    observation = BandCorrelationDelay(
        frequencies,
        np.where(valid[None], water, 0),
        np.ones((1, len(frequencies))),
        -3e-6,
        3e-6,
    )
    fine = FiniteFrequencyTravelTimeForward(
        RayBornForward(
            grid,
            geometry,
            frequencies,
            green_backend="volume_integral",
            green_solver_rtol=1e-11,
        ),
        observation,
        water,
    )
    basis = BilinearBasis(grid, (3, 3))
    forward = module.CoefficientForward(fine, basis)
    truth = (
        1
        / np.array([[1502.0, 1498, 1503], [1496, 1520, 1506], [1505, 1492, 1501]]) ** 2
    )
    observed = forward.forward(truth)
    case = USCTCase(
        case_id="trf_physics",
        grid=grid,
        geometry=geometry,
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed[0]),
    )
    config = AlgorithmConfig(
        parameters={
            "stopping": {
                "max_iterations": 15,
                "restore_best_validation": False,
                "update_rtol": None,
                "objective_rtol": None,
            },
            "evaluation": {"receiver_fraction": 0.125, "seed": 42},
        }
    )
    control = InversionControl(
        case, config, observed, default_iterations=15, valid_mask=valid[None]
    )
    initial = np.full(basis.shape, 1 / 1500**2)
    lin = forward.linearize(initial)
    dense = np.column_stack(
        [
            lin.jacobian.forward(e.reshape(basis.shape))[control.split.train]
            for e in np.eye(initial.size)
        ]
    )
    assert np.linalg.matrix_rank(dense) == initial.size
    assert np.linalg.cond(dense) < 100
    if optimizer == "trf":
        solve = load_solver()
    else:
        from usctbench.solvers.nonlinear import nonlinear_least_squares

        solve = nonlinear_least_squares
    result, metrics = solve(
        forward,
        control,
        initial=initial,
        bounds=(1400, 1600),
        damping=0,
        regularization="identity",
        inner_iterations=20,
    )
    assert metrics["data_relative_residual"] < 1e-4
    assert np.sqrt(np.mean((1 / np.sqrt(result) - 1 / np.sqrt(truth)) ** 2)) < 0.01
    assert metrics["evaluation"]["receiver"]["weighted_relative_residual"] < 1e-3
    assert metrics["stop_reason"] == "stationary_gradient"
