"""Nonlinear outer-loop scale, feasible-step and acceptance regressions."""

import math

import numpy as np
import pytest
from scipy.linalg import norm

import usctbench.algorithms.bent_ray as bent
import usctbench.solvers.nonlinear as nonlinear
from usctbench.algorithms._control import InversionControl
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.base import Linearization


class _DiagonalForward:
    def __init__(self, initial, diagonal):
        self.initial, self.diagonal = initial, diagonal
        self.n_rays = initial.size
        self.spatial_order = 1
        self.background_builds = self.eikonal_solves = 0

    def forward(self, image):
        return self.diagonal * image

    def adjoint(self, data):
        return self.diagonal * data.reshape(self.initial.shape)

    def linearize(self, model):
        self.background_builds += 1
        return Linearization(self.forward(model - self.initial), self)


@pytest.fixture(params=["nonlinear", "bent"])
def run_outer(request, monkeypatch):
    def run(*, blocked=False, direction="exact"):
        initial = np.full(
            (2, 2), 1 / 1500 ** (2 if request.param == "nonlinear" else 1)
        )
        diagonal = np.full(initial.shape, 1e-80)
        observed = np.arange(1.0, 5.0).reshape(initial.shape) * 1e-88
        if blocked:
            # The ordinary gradient remains large; only its feasible part is tiny.
            diagonal[0, 0], observed[0, 0] = -1.0, 1.0
        forward = _DiagonalForward(initial, diagonal)
        case = USCTCase(
            case_id="outer_scale_guard",
            grid=GridSpec(shape=initial.shape, spacing_m=(0.001, 0.001)),
            geometry=GeometrySpec(
                tx_pos_m=[[-0.01, 0], [-0.01, 0.001]],
                rx_pos_m=[[0.01, 0], [0.01, 0.001]],
            ),
            # Absolute data avoid adding/subtracting a macroscopic water offset.
            measurement=MeasurementSpec(domain="features", tof_s=observed),
        )
        options = {"rtol": 1e-9}
        config = AlgorithmConfig(
            parameters={
                "iterations": 1,
                "inner_solver": "lsqr",
                "inner_options": options,
                "damping": 0.0,
                "regularization": "identity",
                "sound_speed_bounds_mps": [1400, 1500],
                "stopping": {
                    "objective_rtol": None,
                    "update_rtol": None,
                    "restore_best_validation": False,
                },
            }
        )
        calls = []

        def inner_step(jacobian, residual, current, control, **settings):
            calls.append(settings)
            assert current.shape == initial.shape
            if direction == "tiny":
                return np.full_like(current, 1e-170)
            if direction == "zero":
                return np.zeros_like(current)
            return residual / diagonal

        if request.param == "nonlinear":
            monkeypatch.setattr(nonlinear, "normal_step", inner_step)
            control = InversionControl(case, config, observed, default_iterations=1)
            state, metrics = nonlinear.nonlinear_least_squares(
                forward,
                control,
                initial=initial,
                bounds=(1400, 1500),
                inner_solver="lsqr",
                inner_options=options,
                regularization="identity",
                max_update_mps=100,
            )
        else:
            monkeypatch.setattr(bent, "normal_step", inner_step)
            monkeypatch.setattr(bent, "EikonalForward", lambda *a, **kw: forward)
            result = bent.BentRayGNAdapter().run(case, config)
            expected_status = "failed" if direction == "tiny" else "success"
            assert result.status == expected_status, result.failure_reason
            state, metrics = 1 / result.sound_speed_mps, result.metrics
        assert state.shape == initial.shape
        return initial, state, metrics, calls, -diagonal * observed

    return run


@pytest.mark.parametrize("blocked", [False, True], ids=["ordinary", "projected"])
def test_tiny_representable_gradient_reaches_inner_solve(run_outer, blocked):
    initial, state, metrics, calls, gradient = run_outer(blocked=blocked)
    feasible = gradient.copy()
    if blocked:
        feasible[0, 0] = 0
    assert np.any(feasible != 0) and np.linalg.norm(feasible) == 0
    assert len(calls) == 1
    assert calls[0]["method"] == "lsqr" and calls[0]["rtol"] == 1e-9
    first = metrics["gradient_checks"][0]
    assert first["norm"] == pytest.approx(
        math.hypot(*gradient.ravel()), rel=1e-14, abs=0
    )
    assert first["relative_to_initial"] == 1.0
    assert metrics["iterations"] == 1
    assert np.all(state.ravel()[1:] > initial.ravel()[1:])
    if blocked:
        assert state[0, 0] == initial[0, 0]
    assert any(row["accepted"] for row in metrics["line_search_history"])


@pytest.mark.parametrize("direction", ["tiny", "zero"])
def test_only_exact_zero_direction_uses_fallback_scale(run_outer, direction):
    initial, state, metrics, calls, _ = run_outer(direction=direction)
    assert len(calls) == 1
    assert metrics["gradient_checks"][0]["norm"] > 0
    if direction == "tiny":
        np.testing.assert_array_equal(state, initial)
        assert metrics["stop_reason"] == "line_search_failed"
        assert not any(row["accepted"] for row in metrics["line_search_history"])
    else:
        assert np.all(state > initial)
        assert any(
            row["accepted"] and row["proposal"] == "projected_gradient"
            for row in metrics["line_search_history"]
        )


class _DenseForward:
    def __init__(self, matrix, reference):
        self.matrix, self.reference = matrix, reference
        self.states = []
        self.background_builds = self.eikonal_solves = 0

    def forward(self, image):
        return (self.matrix @ image.ravel()).reshape(1, -1)

    def adjoint(self, data):
        return (self.matrix.T @ data.ravel()).reshape(self.reference.shape)

    def linearize(self, model):
        self.states.append(model.copy())
        self.background_builds += 1
        return Linearization(self.forward(model - self.reference), self)


@pytest.fixture
def solve(monkeypatch):
    original_step = nonlinear.normal_step
    directions = []

    def capture_step(*args, **kwargs):
        direction = original_step(*args, **kwargs)
        directions.append(direction.copy())
        return direction

    monkeypatch.setattr(nonlinear, "normal_step", capture_step)

    def run(matrix, observed, *, initial=None, **options):
        if initial is None:
            initial = np.full((1, matrix.shape[1]), 1 / 1500**2)
        observed = np.asarray(observed).reshape(1, -1)
        case = USCTCase(
            case_id="feasible_displacement_backtracking",
            grid=GridSpec(shape=initial.shape, spacing_m=(0.001, 0.001)),
            geometry=GeometrySpec(
                tx_pos_m=[[-0.01, 0]],
                rx_pos_m=[[0.01, i * 0.001] for i in range(observed.size)],
            ),
            measurement=MeasurementSpec(domain="features", tof_s=observed),
        )
        config = AlgorithmConfig(
            parameters={
                "stopping": {
                    "max_iterations": 1,
                    "objective_rtol": None,
                    "update_rtol": None,
                    "restore_best_validation": False,
                },
            }
        )
        control = InversionControl(case, config, observed, default_iterations=1)
        forward = _DenseForward(matrix, initial)
        settings = dict(
            bounds=(1400, 1600),
            regularization="identity",
            inner_solver="lsmr",
            inner_iterations=64,
            max_backtracks=8,
        )
        settings.update(options)
        state, metrics = nonlinear.nonlinear_least_squares(
            forward, control, initial=initial, **settings
        )
        return state, metrics, forward, directions, control

    return run


def _original_first_trial(initial, proposal, step_length, bounds, max_update, roi=None):
    candidate = np.clip(
        initial + step_length * proposal, 1 / bounds[1] ** 2, 1 / bounds[0] ** 2
    )
    speed = 1 / np.sqrt(initial)
    candidate = (
        1 / np.clip(1 / np.sqrt(candidate), speed - max_update, speed + max_update) ** 2
    )
    return candidate if roi is None else np.where(roi, candidate, initial)


def test_separate_prior_matches_dense_solution_and_keeps_roi_exterior(solve):
    initial = np.full((1, 4), 1 / 1500**2)
    prior = initial + np.array([[-1.0, 2.0, -1.0, 1.0]]) * 1e-9
    matrix = np.diag([300.0, 200, 100, 50])
    observed = matrix @ np.array([2.0, -1.0, 1.0, 2.0]) * 1e-9
    damping = 5000.0
    roi = np.array([[True, True, False, False]])
    state, metrics, _, _, _ = solve(
        matrix,
        observed,
        initial=initial,
        prior_reference=prior,
        damping=damping,
        roi=roi,
    )
    selected = matrix[:, roi.ravel()]
    delta = np.linalg.solve(
        selected.T @ selected + damping * np.eye(roi.sum()),
        selected.T @ observed + damping * (prior - initial)[roi],
    )
    expected = initial.copy()
    expected[roi] += delta
    np.testing.assert_allclose(state, expected, rtol=1e-12)
    np.testing.assert_array_equal(state[~roi], initial[~roi])
    residual = matrix @ (state - initial).ravel() - observed
    objective = 0.5 * (residual @ residual + damping * np.sum((state - prior) ** 2))
    assert metrics["iteration_history"][-1]["objective"] == pytest.approx(
        objective, rel=1e-12
    )
    assert metrics["prior_reference_source"] == "explicit"


def _assert_displacement_trials(initial, states, rows, first, step_length):
    assert len(states) == len(rows) > 0
    np.testing.assert_array_equal(states[0], first)
    full_displacement = first - initial
    for index, (state, row) in enumerate(zip(states, rows)):
        fraction = 0.5**index
        assert row["step_length"] == step_length
        assert row["backtracking_fraction"] == fraction
        expected = first if index == 0 else initial + fraction * full_displacement
        np.testing.assert_array_equal(state, expected)
        displacement = state - initial
        assert row["displacement_norm"] == pytest.approx(norm(displacement.ravel()))
        assert row["changed_pixel_count"] == np.count_nonzero(displacement)
        if index:
            assert row["displacement_norm"] == pytest.approx(
                0.5 * rows[index - 1]["displacement_norm"], rel=2e-12, abs=0
            )
            assert row["displacement_norm"] < rows[index - 1]["displacement_norm"]


def _assert_armijo(metrics):
    initial_cost = metrics["iteration_history"][0]["objective"]
    for row in metrics["line_search_history"]:
        if row["accepted"]:
            assert row["projected_directional_derivative"] < 0
            assert row["objective"] <= (
                initial_cost + 1e-4 * row["projected_directional_derivative"]
            )


@pytest.mark.parametrize("inner_iterations", [64, 256])
def test_certified_quadratic_step_recovers_from_saturated_first_trial(
    solve, inner_iterations
):
    matrix = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    state, metrics, forward, directions, _ = solve(
        matrix, [1e-3, 1e-3], inner_iterations=inner_iterations
    )
    assert metrics["inner_solver_history"][0]["converged"]
    assert metrics["iterations"] == 1
    assert metrics["iteration_history"][-1]["objective"] < 0.51e-6
    rows = metrics["line_search_history"]
    assert len(rows) == 5 and all(row["proposal"] == "newton_proposal" for row in rows)
    first = _original_first_trial(forward.reference, directions[0], 1, (1400, 1600), 12)
    _assert_displacement_trials(forward.reference, forward.states[1:], rows, first, 1)
    np.testing.assert_array_equal(state, forward.states[-1])
    assert rows[-1]["accepted"]
    assert np.max(np.abs(1 / np.sqrt(first) - 1500)) == pytest.approx(12)
    for candidate in forward.states[1:]:
        assert np.all(candidate >= 1 / 1600**2)
        assert np.all(candidate <= 1 / 1400**2)
        assert np.max(np.abs(1 / np.sqrt(candidate) - 1500)) <= 12 + 1e-12
    _assert_armijo(metrics)


@pytest.mark.parametrize("step_length", [0.4, 3.0])
def test_linear_unconstrained_step_length_is_applied_once(solve, step_length):
    matrix = 1e8 * np.eye(4)
    truth_step = np.array([[1, -2, 3, -4]]) * 1e-10
    state, metrics, forward, directions, _ = solve(
        matrix, matrix @ truth_step.ravel(), step_length=step_length
    )
    rows = metrics["line_search_history"]
    assert len(rows) == (1 if step_length < 1 else 2)
    first = _original_first_trial(
        forward.reference, directions[0], step_length, (1400, 1600), 12
    )
    assert np.max(np.abs(1 / np.sqrt(first) - 1500)) < 12
    _assert_displacement_trials(
        forward.reference, forward.states[1:], rows, first, step_length
    )
    np.testing.assert_allclose(
        state - forward.reference,
        step_length * rows[-1]["backtracking_fraction"] * truth_step,
        rtol=2e-12,
        atol=1e-22,
    )
    assert metrics["iteration_history"][-1]["objective"] < (
        metrics["iteration_history"][0]["objective"]
    )
    constraints = metrics["line_search_constraints"]
    assert (
        constraints["strategy"] == "project_once_then_backtrack_feasible_displacement"
    )
    assert constraints["step_length_applies_to"] == "raw_proposal_before_projection"
    assert (
        constraints["backtracking_fraction_applies_to"]
        == "initial_feasible_displacement"
    )
    assert constraints["sound_speed_bounds_mps"] == [1400, 1600]
    assert constraints["max_update_mps"] == 12
    assert constraints["displacement_parameter"] == "squared_slowness"
    _assert_armijo(metrics)


def test_each_rejected_halving_shrinks_both_proposals_and_preserves_checkpoint(solve):
    matrix = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    state, metrics, forward, _, control = solve(matrix, [1e-9, 1e-3])
    rows = metrics["line_search_history"]
    assert metrics["stop_reason"] == "line_search_failed"
    assert len(rows) == 16 and len(forward.states) == 17
    assert not any(row["accepted"] for row in rows)
    for start in (0, 8):
        trial_states = forward.states[1 + start : 9 + start]
        _assert_displacement_trials(
            forward.reference, trial_states, rows[start : start + 8], trial_states[0], 1
        )
    np.testing.assert_array_equal(state, forward.reference)
    np.testing.assert_array_equal(control.last_prediction, np.zeros((1, 2)))
    assert len(metrics["iteration_history"]) == 1


def test_roi_is_fixed_for_the_initial_projection_and_every_halving(solve):
    matrix = np.eye(4)
    matrix[:2, :2] = 1e8 * np.array([[1.0, 1.0], [0.0, 1e-6]])
    initial = 1 / np.array([[1500.0, 1500], [1490, 1510]]) ** 2
    roi = np.array([[True, True], [False, False]])
    state, metrics, forward, directions, _ = solve(
        matrix, [1e-3, 1e-3, 1, 1], initial=initial, roi=roi, step_length=3
    )
    first = _original_first_trial(initial, directions[0], 3, (1400, 1600), 12, roi)
    _assert_displacement_trials(
        initial, forward.states[1:], metrics["line_search_history"], first, 3
    )
    for candidate in forward.states:
        np.testing.assert_array_equal(candidate[~roi], initial[~roi])
        assert np.all(1 / np.sqrt(candidate[roi]) >= 1400)
        assert np.all(1 / np.sqrt(candidate[roi]) <= 1600)
    assert metrics["line_search_constraints"]["roi_fixed"]
    assert all(
        row["changed_pixel_count"] == 2 for row in metrics["line_search_history"]
    )
    assert np.any(state[roi] != initial[roi])
    _assert_armijo(metrics)


def test_projected_uphill_newton_uses_projected_gradient_without_evaluation(solve):
    hessian = np.array([[3.0, -8], [-8, 22]])
    matrix = np.linalg.cholesky(hessian).T
    initial = np.array([[1 / 1600**2, 1 / 1500**2]])
    gradient = np.array([[1.0, -2]]) * 1e-6
    observed = -np.linalg.solve(matrix.T, gradient.ravel())
    state, metrics, forward, directions, _ = solve(matrix, observed, initial=initial)
    rows = metrics["line_search_history"]
    assert metrics["direction_checks"][0]["raw_directional_derivative"] < 0
    newton_rows = [row for row in rows if row["proposal"] == "newton_proposal"]
    assert len(newton_rows) == 8
    assert all(row["projected_directional_derivative"] > 0 for row in newton_rows)
    assert all(row["rejection"] == "non_descent_projected_step" for row in newton_rows)
    assert all(row["objective"] is None for row in newton_rows)
    for index, row in enumerate(newton_rows):
        assert row["changed_pixel_count"] == 1
        assert row["backtracking_fraction"] == 0.5**index
        assert row["displacement_norm"] == pytest.approx(
            0.5**index * newton_rows[0]["displacement_norm"], rel=2e-12, abs=0
        )
    assert rows[-1]["proposal"] == "projected_gradient" and rows[-1]["accepted"]
    feasible_gradient = gradient.copy()
    feasible_gradient[0, 0] = 0
    fallback = -feasible_gradient * (
        norm(directions[0].ravel()) / norm(feasible_gradient.ravel())
    )
    first = _original_first_trial(initial, fallback, 1, (1400, 1600), 12)
    np.testing.assert_array_equal(forward.states[1], first)
    assert len(forward.states) == 1 + sum(row["objective"] is not None for row in rows)
    assert state[0, 0] == initial[0, 0]
    _assert_armijo(metrics)
