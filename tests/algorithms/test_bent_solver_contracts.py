"""Native Bent optimizer contracts; dense fixtures isolate optimization from PDEs."""

import numpy as np
import pytest

import usctbench.algorithms.bent_ray as bent
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.model_space import BilinearBasis
from usctbench.solvers.least_squares import regularizer


def _config(**parameters):
    settings = {
        "iterations": 4,
        "inner_iterations": 80,
        "damping": 0.0,
        "regularization": "identity",
        "stopping": {
            "objective_rtol": None,
            "update_rtol": None,
            "restore_best_validation": False,
        },
    }
    stopping = parameters.pop("stopping", {})
    settings["stopping"].update(stopping)
    settings.update(parameters)
    return AlgorithmConfig(parameters=settings)


def _case(shape, data):
    nt, nr = data.shape
    return USCTCase(
        case_id="bent_solver_contract",
        grid=GridSpec(shape=shape, spacing_m=(0.001, 0.0008)),
        geometry=GeometrySpec(
            tx_pos_m=[[-0.001, 0.0012 + i * 0.001] for i in range(nt)],
            rx_pos_m=[[0.0092, 0.0052 + i * 0.001] for i in range(nr)],
        ),
        measurement=MeasurementSpec(domain="features", delta_tof_s=data),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def _offset(case):
    return (
        np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None, :], axis=-1
        )
        / 1500
    )


class _DenseForward:
    def __init__(self, case, matrix):
        self.grid = case.grid
        self.ray_shape = case.measurement.delta_tof_s.shape
        self.n_rays = int(np.prod(self.ray_shape))
        self.spatial_order = 1
        self.matrix = matrix
        self.offset = _offset(case)
        self.states = []
        self.adjoint_calls = 0
        self.trial_failure = None

    def linearize(self, state):
        self.states.append(state.copy())
        value = self.offset.ravel() + self.matrix @ (state - 1 / 1500).ravel()
        if self.trial_failure is not None and len(self.states) > 1:
            value = self.trial_failure(value)
        parent = self

        class Jacobian:
            grid = parent.grid
            ray_shape = parent.ray_shape
            n_rays = parent.n_rays

            def forward(self, x):
                return parent.matrix @ x.ravel()

            def adjoint(self, y):
                parent.adjoint_calls += 1
                return (parent.matrix.T @ y.ravel()).reshape(parent.grid.shape)

        return Linearization(value, Jacobian())


def _dense(monkeypatch, matrix, data, shape=(2, 2)):
    case = _case(shape, np.asarray(data).reshape(2, -1))
    forward = _DenseForward(case, matrix)
    monkeypatch.setattr(bent, "EikonalForward", lambda *a, **kw: forward)
    return case, forward


def _assert_armijo(result):
    costs = {
        row["iteration"]: row["objective"]
        for row in result.metrics["iteration_history"]
    }
    for row in result.metrics["line_search_history"]:
        if row["accepted"]:
            assert row["projected_directional_derivative"] < 0
            assert row["objective"] <= (
                costs[row["iteration"] - 1]
                + 1e-4 * row["projected_directional_derivative"]
            )


@pytest.mark.parametrize("reduced", [False, True])
def test_uphill_gaussian_smoothing_recovers_regularized_dense_solution(
    monkeypatch, reduced
):
    diagonal = np.sqrt([1000.0, 1, 1, 1])
    matrix = np.diag(diagonal)
    data = -1e-5 * np.array([-3.0, 0.6, 0.6, -0.5]) / diagonal
    case, _ = _dense(monkeypatch, matrix, data)
    result = bent.BentRayGNAdapter().run(
        case,
        _config(
            damping=0.2,
            smooth_sigma=1,
            **({"model_grid_shape": [2, 2]} if reduced else {}),
        ),
    )
    expected = np.linalg.solve(matrix.T @ matrix + 0.2 * np.eye(4), matrix.T @ data)
    assert result.status == "success", result.failure_reason
    np.testing.assert_allclose(
        (1 / result.sound_speed_mps - 1 / 1500).ravel(), expected, atol=2e-14
    )
    check = result.metrics["direction_checks"][0]
    assert (
        check["raw_directional_derivative"]
        < 0
        < check["smoothed_directional_derivative"]
    )
    assert check["smoothing_rejected"]
    assert result.metrics["stop_reason"] == "stationary_gradient"
    assert result.metrics["gradient_checks"][-1]["relative_to_initial"] <= 1e-8
    _assert_armijo(result)


@pytest.mark.parametrize("kind", ["identity", "laplacian"])
@pytest.mark.parametrize("reduced", [False, True])
def test_weighted_regularized_solution_matches_fine_grid_dense_reference(
    monkeypatch, kind, reduced
):
    rng = np.random.default_rng(241)
    shape = (4, 5)
    matrix = rng.normal(size=(24, 20))
    data = matrix @ rng.normal(scale=2e-6, size=20) + rng.normal(scale=1e-6, size=24)
    case, _ = _dense(monkeypatch, matrix, data, shape)
    weights = rng.uniform(0.2, 0.9, (2, 12))
    case.measurement.ray_weights = weights
    config = _config(
        damping=0.3,
        regularization=kind,
        gradient_rtol=1e-6,
        evaluation={"receiver_indices": [1]},
        **({"model_grid_shape": [2, 3]} if reduced else {}),
    )
    basis = BilinearBasis(case.grid, (2, 3)).matrix.toarray() if reduced else np.eye(20)
    penalty = np.column_stack(
        [regularizer(column.reshape(shape), kind).ravel() for column in basis.T]
    )
    train_weights = weights.copy()
    train_weights[:, 1] = 0
    design = matrix @ basis
    expected = np.linalg.solve(
        design.T @ (train_weights.ravel()[:, None] * design)
        + 0.3 * penalty.T @ penalty,
        design.T @ (train_weights.ravel() * data),
    )
    result = bent.BentRayGNAdapter().run(case, config)
    assert result.status == "success", result.failure_reason
    np.testing.assert_allclose(
        (1 / result.sound_speed_mps - 1 / 1500).ravel(), basis @ expected, atol=2e-11
    )
    assert result.metrics["gradient_space"] == ("coefficients" if reduced else "pixels")
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 1.0
    other = bent.BentRayGNAdapter().run(altered, config)
    np.testing.assert_array_equal(other.sound_speed_mps, result.sound_speed_mps)
    assert other.metrics["gradient_checks"] == result.metrics["gradient_checks"]
    _assert_armijo(result)


def test_descent_smoothing_that_fails_armijo_retries_raw_newton(monkeypatch):
    case, _ = _dense(monkeypatch, np.eye(4), [1e-6, -1e-6, 2e-6, -2e-6])
    monkeypatch.setattr(bent, "_gaussian_smooth", lambda update, sigma: update * 1e6)
    result = bent.BentRayGNAdapter().run(case, _config(iterations=1, smooth_sigma=1))
    assert result.status == "success", result.failure_reason
    assert not result.metrics["direction_checks"][0]["smoothing_rejected"]
    attempts = result.metrics["line_search_history"]
    assert len(attempts) == 11
    assert attempts[-1]["proposal"] == "unsmoothed_newton"
    assert attempts[-1]["accepted"]
    _assert_armijo(result)


def test_clipping_rejects_uphill_newton_and_uses_projected_gradient(monkeypatch):
    hessian = np.eye(4)
    hessian[:2, :2] = [[3, -8], [-8, 22]]
    matrix = np.linalg.cholesky(hessian).T
    initial = np.full((2, 2), 1 / 1500)
    initial[0, 0] = 1 / 1700
    gradient = np.array([1, -2, 0, 0]) * 1e-6
    data = matrix @ (initial - 1 / 1500).ravel() - np.linalg.solve(matrix.T, gradient)
    case, forward = _dense(monkeypatch, matrix, data)
    result = bent.BentRayGNAdapter().run(
        case, _config(iterations=1, initial_sound_speed_mps=1 / initial)
    )
    assert result.status == "success", result.failure_reason
    attempts = result.metrics["line_search_history"]
    assert result.metrics["direction_checks"][0]["raw_directional_derivative"] < 0
    assert attempts[0]["projected_directional_derivative"] > 0
    assert attempts[0]["rejection"] == "non_descent_projected_step"
    assert attempts[-1]["proposal"] == "projected_gradient"
    assert attempts[-1]["accepted"]
    assert len(forward.states) == 1 + sum(
        row["objective"] is not None for row in attempts
    )
    _assert_armijo(result)


def test_zero_inner_update_does_not_claim_stationarity(monkeypatch):
    case, _ = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    monkeypatch.setattr(
        bent,
        "normal_step",
        lambda jac, residual, current, *a, **kw: np.zeros_like(current),
    )
    result = bent.BentRayGNAdapter().run(case, _config(iterations=1))
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "max_iterations"
    assert result.metrics["line_search_history"][-1]["proposal"] == "projected_gradient"
    assert result.metrics["data_residual_reduction"] > 0


def test_bound_kkt_stationarity_requires_no_newton_or_trial(monkeypatch):
    initial = np.full((2, 2), 1 / 1700)
    case, forward = _dense(monkeypatch, np.eye(4), (initial - 1 / 1500 - 1e-6).ravel())
    result = bent.BentRayGNAdapter().run(case, _config(initial_sound_speed_mps=1700))
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "stationary_projected_gradient"
    assert len(forward.states) == 1
    assert forward.adjoint_calls == 1
    np.testing.assert_allclose(result.sound_speed_mps, 1700)


@pytest.mark.parametrize("recover", [False, True])
def test_nonfinite_trials_are_rejected_without_changing_checkpoint(
    monkeypatch, recover
):
    case, forward = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    forward.trial_failure = lambda value: (
        value if recover and len(forward.states) > 2 else np.full_like(value, np.nan)
    )
    result = bent.BentRayGNAdapter().run(case, _config(iterations=1, smooth_sigma=0.5))
    assert (
        result.metrics["line_search_history"][0]["rejection"]
        == "nonfinite_trial_prediction_or_objective"
    )
    if recover:
        assert result.status == "success", result.failure_reason
        assert result.metrics["data_residual_reduction"] > 0
        _assert_armijo(result)
    else:
        assert result.status == "failed"
        assert result.metrics["stop_reason"] == "numerical_failure"
        assert result.metrics["recovered_checkpoint_only"]
        assert len(result.metrics["iteration_history"]) == 1
        np.testing.assert_array_equal(result.sound_speed_mps, np.full((2, 2), 1500.0))
        assert result.metrics["data_residual_norm"] == pytest.approx(2e-6)


def test_disabling_backtracking_does_not_allow_ascent(monkeypatch):
    case, _ = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    result = bent.BentRayGNAdapter().run(
        case, _config(line_search=False, step_length=10)
    )
    assert result.status == "failed"
    assert result.metrics["stop_reason"] == "line_search_failed"
    assert len(result.metrics["line_search_history"]) == 2
    assert not any(row["accepted"] for row in result.metrics["line_search_history"])
    np.testing.assert_allclose(result.sound_speed_mps, 1500)


def test_unrepresentable_step_is_not_an_accepted_update(monkeypatch):
    case, forward = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    result = bent.BentRayGNAdapter().run(
        case, _config(step_length=1e-300, smooth_sigma=1)
    )
    assert result.status == "failed"
    assert result.metrics["stop_reason"] == "line_search_failed"
    assert result.metrics["iterations"] == 0
    assert len(forward.states) == 1
    assert all(
        row["rejection"] == "non_descent_projected_step"
        for row in result.metrics["line_search_history"]
    )
    np.testing.assert_array_equal(result.sound_speed_mps, np.full((2, 2), 1500.0))


@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("inner_solver", ["normal_cg", "lsmr", "lsqr"])
def test_budget_after_rejected_trial_preserves_image_and_prediction(
    monkeypatch, reduced, inner_solver
):
    case, forward = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    result = bent.BentRayGNAdapter().run(
        case,
        _config(
            damping=0.2,
            step_length=10,
            inner_solver=inner_solver,
            # Augmented solvers additionally measure an operator scale; count
            # that call while still exhausting the budget AFTER a rejected trial.
            stopping={"max_forward_calls": 4 if inner_solver == "normal_cg" else 5},
            **({"model_grid_shape": [2, 2]} if reduced else {}),
        ),
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "forward_calls_budget"
    assert len(forward.states) == 2
    assert len(result.metrics["iteration_history"]) == 1
    assert len(result.metrics["line_search_history"]) == 1
    assert not result.metrics["line_search_history"][0]["accepted"]
    np.testing.assert_allclose(result.sound_speed_mps, 1500)
    assert result.metrics["data_residual_norm"] == pytest.approx(2e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e308])
def test_nonfinite_inner_update_or_norm_preserves_checkpoint(monkeypatch, bad):
    case, forward = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    monkeypatch.setattr(
        bent,
        "normal_step",
        lambda jac, residual, current, *a, **kw: np.full_like(current, bad),
    )
    result = bent.BentRayGNAdapter().run(case, _config())
    assert result.status == "failed"
    assert result.metrics["stop_reason"] == "numerical_failure"
    assert len(forward.states) == 1
    np.testing.assert_allclose(result.sound_speed_mps, 1500)


@pytest.mark.parametrize("scale", [1e-6, 1e6])
def test_relative_gradient_stopping_is_invariant_to_objective_scale(monkeypatch, scale):
    matrix = np.diag(np.sqrt([1000.0, 1, 1, 1]))
    data = -1e-5 * np.array([-3.0, 0.6, 0.6, -0.5]) / np.diag(matrix)
    case, _ = _dense(monkeypatch, matrix * scale, data * scale)
    result = bent.BentRayGNAdapter().run(case, _config(damping=0.2 * scale**2))
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == "stationary_gradient"
    assert result.metrics["iterations"] == 1
    assert result.metrics["gradient_checks"][0]["relative_to_initial"] == 1
    expected = np.linalg.solve(matrix.T @ matrix + 0.2 * np.eye(4), matrix.T @ data)
    np.testing.assert_allclose(
        (1 / result.sound_speed_mps - 1 / 1500).ravel(), expected, atol=2e-14
    )


@pytest.mark.parametrize("counter", ["forward_calls", "adjoint_calls"])
@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4])
def test_gradient_and_trials_obey_budgets_and_return_complete_iterates(
    monkeypatch, counter, limit
):
    case, forward = _dense(monkeypatch, np.eye(4), np.full(4, 1e-6))
    result = bent.BentRayGNAdapter().run(
        case, _config(damping=0.2, stopping={f"max_{counter}": limit})
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["stop_reason"] == f"{counter}_budget"
    work = result.metrics["stopping"]["work"]
    assert work[counter] == limit
    assert work["adjoint_calls"] == forward.adjoint_calls
    assert work.get("forward", 0) + work.get("line_search", 0) == len(forward.states)
    if result.metrics["iteration_history"]:
        delta = 1 / result.sound_speed_mps - 1 / 1500
        residual = delta - case.measurement.delta_tof_s
        expected_cost = 0.5 * np.sum(residual**2) + 0.1 * np.sum(delta**2)
        assert result.metrics["iteration_history"][-1]["objective"] == pytest.approx(
            expected_cost, abs=1e-24
        )
        assert result.metrics["data_residual_norm"] == pytest.approx(
            np.linalg.norm(residual), abs=1e-17
        )
    else:
        np.testing.assert_allclose(result.sound_speed_mps, 1500)


@pytest.mark.parametrize("reference", [None, 1500.0, 1490.0])
def test_differential_reference_override_must_match_metadata(reference):
    case = _case((3, 3), np.ones((2, 2)) * 1e-8)
    if reference is None:
        case.metadata.clear()
    else:
        case.metadata["reference_sound_speed_mps"] = reference
    result = bent.BentRayGNAdapter().run(case, _config(reference_sound_speed_mps=1600))
    assert result.status == "failed"
    assert "measurement reference" in result.failure_reason
    assert result.sound_speed_mps is None
    matching = 1500 if reference is None else reference
    result = bent.BentRayGNAdapter().run(
        case,
        _config(reference_sound_speed_mps=matching, stopping={"max_forward_calls": 0}),
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["measurement_reference_sound_speed_mps"] == matching


def test_absolute_tof_allows_model_reference_override():
    case = _case((3, 3), np.ones((2, 2)) * 1e-8)
    case.measurement.tof_s = _offset(case)
    case.measurement.delta_tof_s = None
    result = bent.BentRayGNAdapter().run(
        case, _config(reference_sound_speed_mps=1600, stopping={"max_forward_calls": 0})
    )
    assert result.status == "success", result.failure_reason
    assert result.metrics["measurement_reference_sound_speed_mps"] is None
    np.testing.assert_allclose(result.sound_speed_mps, 1600)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_gradient_tolerance_must_be_finite_nonnegative(bad):
    case = _case((3, 3), np.ones((2, 2)) * 1e-8)
    result = bent.BentRayGNAdapter().run(case, _config(gradient_rtol=bad))
    assert result.status == "failed"
    assert "gradient_rtol" in result.failure_reason


@pytest.mark.parametrize("spatial_order", [1, 2])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("kind", ["identity", "laplacian"])
def test_native_regularized_objective_directional_derivative_and_roi(
    monkeypatch, spatial_order, reduced, kind
):
    rng = np.random.default_rng(83)
    case = _case((7, 8), np.zeros((2, 4)))
    forward = EikonalForward(case.grid, case.geometry, spatial_order=spatial_order)
    truth = (1 + rng.uniform(-0.02, 0.02, case.grid.shape)) / 1500
    case.measurement.delta_tof_s = forward.forward(truth).reshape(2, 4) - _offset(case)
    case.measurement.ray_weights = rng.uniform(0.2, 0.9, (2, 4))
    roi = np.ones(case.grid.shape, bool)
    roi[[0, -1], :] = False
    case.grid.roi_mask = roi
    basis = BilinearBasis(case.grid, (3, 4)) if reduced else None
    parameters = (
        {"model_grid_shape": [3, 4]}
        if reduced
        else {
            "initial_sound_speed_mps": 1500 + rng.uniform(-20, 20, case.grid.shape),
            "roi_update_only": True,
        }
    )
    records = []
    original_step = bent.normal_step

    def record_step(jacobian, residual, current, control, **kwargs):
        update = original_step(jacobian, residual, current, control, **kwargs)
        records.append((current.copy(), update.copy(), control))
        return update

    monkeypatch.setattr(bent, "normal_step", record_step)
    config = _config(
        iterations=2,
        inner_iterations=12,
        damping=1e-5,
        regularization=kind,
        step_length=0.3,
        smooth_sigma=0.5,
        eikonal_order=spatial_order,
        evaluation={"receiver_indices": [1]},
        **parameters,
    )
    result = bent.BentRayGNAdapter().run(case, config)
    assert result.status == "success", result.failure_reason
    assert len(records) == 2
    current, direction, control = records[-1]
    fine = current if basis is None else basis.forward(current)
    tangent = direction if basis is None else basis.forward(direction)
    assert np.linalg.norm(regularizer(fine, kind)) > 0

    def objective(alpha):
        delta = fine + alpha * tangent
        prediction = forward.forward(1 / 1500 + delta).reshape(2, 4) - _offset(case)
        residual = np.where(control.split.train, control.safe_observed - prediction, 0)
        return 0.5 * np.sum(control.precision * residual**2) + 0.5e-5 * np.sum(
            regularizer(delta, kind) ** 2
        )

    epsilon = 1e-3
    finite_difference = (objective(epsilon) - objective(-epsilon)) / (2 * epsilon)
    slope = result.metrics["direction_checks"][-1]["raw_directional_derivative"]
    np.testing.assert_allclose(slope, finite_difference, rtol=2e-4, atol=1e-23)
    if not reduced:
        np.testing.assert_array_equal(
            result.sound_speed_mps[~roi], np.full(np.sum(~roi), 1500.0)
        )
    altered = case.model_copy(deep=True)
    altered.measurement.delta_tof_s[:, 1] += 1.0
    other = bent.BentRayGNAdapter().run(altered, config)
    np.testing.assert_array_equal(result.sound_speed_mps, other.sound_speed_mps)
    assert result.metrics["direction_checks"] == other.metrics["direction_checks"]
    _assert_armijo(result)
