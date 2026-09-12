"""Independent optimization references for the registered straight CGLS path."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import lsq_linear, minimize, minimize_scalar

from usctbench.algorithms._control import InversionControl
from usctbench.algorithms.ray import StraightRayCGLSAlgorithm
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    MeasurementSpec,
    USCTCase,
)
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.solvers.projected_cg import projected_cg


def huber_case():
    # The two active Siddon rows are exactly [0.001] and [0.002] meters.
    return USCTCase(
        case_id="one_pixel_huber_reference",
        grid=GridSpec(shape=(1, 1), spacing_m=(0.001, 0.002)),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=[[-0.001, 0.0005], [0.0005, -0.001]],
            rx_pos_m=[[0.002, 0.0005], [0.0005, 0.003]],
        ),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=[[1e-8, 0], [0, 4e-8]],
            valid_mask=np.eye(2, dtype=bool),
        ),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def bounded_case():
    # A grazing/short first ray is bound-active; the second pixel remains free.
    return USCTCase(
        case_id="two_pixel_bound_reference",
        grid=GridSpec(shape=(1, 2), spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=[[0.0001, 0.0001], [-0.001, 0.0015]],
            rx_pos_m=[[0.00011, 0.0001], [0.002, 0.0015]],
        ),
        measurement=MeasurementSpec(
            domain="features",
            delta_tof_s=[[1e-6, 0], [0, 2.5e-9]],
            valid_mask=np.eye(2, dtype=bool),
        ),
        metadata={"reference_sound_speed_mps": 1500.0},
    )


def parameters(robust):
    if robust:
        return {
            "iterations": 30,
            "robust_loss": "huber",
            "huber_delta_s": 1e-8,
            "irls_iterations": 3,
        }
    return {
        "iterations": 30,
        "sound_speed_bounds_mps": [1 / (1 / 1500 + 1e-5), 1 / (1 / 1500 - 1e-5)],
    }


def run(case, params):
    result = StraightRayCGLSAlgorithm().run(case, AlgorithmConfig(parameters=params))
    assert result.status == "success", result.failure_reason
    assert case.ground_truth.sound_speed_mps is None
    return result


def perturbation(result):
    return 1 / result.sound_speed_mps - 1 / 1500


@pytest.mark.parametrize("stages", [1, 3, 30])
def test_huber_inner_convergence_refreshes_before_global_stopping(stages):
    result = run(huber_case(), {**parameters(True), "irls_iterations": stages})
    # In units x=delta_s/1e-5, min rho(x-1)+rho(2*x-4) has x=1.8.
    np.testing.assert_allclose(perturbation(result), [[1.8e-5]], rtol=1e-10)
    assert result.metrics["iteration_history"][-1]["objective"] == pytest.approx(
        4e-17, rel=1e-10, abs=1e-28
    )
    assert result.metrics["stopping"]["work"]["irls_reweights"] >= 2
    assert result.metrics["iterations"] <= 30
    assert result.metrics["solver_optimality"]["verified"]
    assert result.metrics["solver_optimality"]["converged"]


@pytest.mark.parametrize("weight", [0.25, 1.0])
def test_weighted_huber_matches_independent_scalar_minimization(weight):
    case = huber_case()
    case.measurement.ray_weights = np.diag([weight, 1.0])

    def objective(x):
        residual = np.abs(np.array([x - 1, 2 * x - 4]))
        rho = np.where(residual <= 1, 0.5 * residual**2, residual - 0.5)
        return np.array([weight, 1.0]) @ rho

    expected = minimize_scalar(
        objective, method="bounded", bounds=(-5, 5), options={"xatol": 1e-13}
    )
    assert expected.success
    result = run(case, parameters(True))
    actual = perturbation(result).item() / 1e-5
    assert actual == pytest.approx(expected.x, rel=1e-7)
    assert objective(actual) == pytest.approx(expected.fun, rel=1e-12)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_bound_active_cgls_matches_bvls_and_projected_kkt(sign, preconditioned):
    case = bounded_case()
    case.measurement.delta_tof_s *= sign
    # Rescale both image and time units before asking SciPy for a reference.
    matrix = np.diag([0.01, 1.0])
    data = sign * np.array([100.0, 0.25])
    expected = lsq_linear(matrix, data, bounds=(-1, 1), method="bvls", tol=1e-13)
    assert expected.success
    result = run(
        case, {**parameters(False), "coverage_preconditioning": preconditioned}
    )
    x = perturbation(result).ravel() / 1e-5
    np.testing.assert_allclose(x, expected.x, rtol=1e-9, atol=1e-10)
    gradient = matrix.T @ (matrix @ x - data)
    kkt = x - np.clip(x - gradient, -1, 1)
    assert np.linalg.norm(kkt, np.inf) < 1e-10
    assert result.metrics["stop_reason"] != "line_search_failed"
    assert result.metrics["solver_optimality"]["converged"]


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("unit_scale", [1e-6, 1.0, 1e6])
def test_stationarity_is_relative_not_an_absolute_physical_gradient_cutoff(
    robust, unit_scale
):
    case = huber_case() if robust else bounded_case()
    case.grid.spacing_m = tuple(np.array(case.grid.spacing_m) * unit_scale)
    case.geometry.tx_pos_m *= unit_scale
    case.geometry.rx_pos_m *= unit_scale
    case.measurement.delta_tof_s *= unit_scale
    params = parameters(robust)
    if robust:
        params["huber_delta_s"] *= unit_scale
    result = run(case, params)
    expected = [[1.8e-5]] if robust else [[1e-5, 2.5e-6]]
    np.testing.assert_allclose(perturbation(result), expected, rtol=1e-9, atol=1e-15)
    certificate = result.metrics["solver_optimality"]
    assert result.metrics["iterations"] >= 2
    assert certificate["verified"] and certificate["converged"]
    initial_norm = certificate["initial_projected_gradient_norm"]
    assert initial_norm > 0
    assert certificate["projected_gradient_tolerance"] == pytest.approx(
        initial_norm * 1e-10, rel=1e-12, abs=0
    )
    assert certificate["projected_gradient_norm"] / initial_norm <= 1e-10


@pytest.mark.parametrize("robust", [False, True])
def test_coupled_weighted_bounded_penalty_matches_independent_lbfgsb(robust):
    case = bounded_case()
    case.geometry.tx_pos_m = np.array([[0.0005, -0.001], [-0.001, 0.0005]])
    case.geometry.rx_pos_m = np.array([[0.0005, 0.003], [0.002, 0.0005]])
    case.measurement.delta_tof_s = np.diag([3e-8, -0.25e-8])
    case.measurement.ray_weights = np.diag([0.4, 1.0])
    matrix = np.array([[1.0, 1.0], [1.0, 0.0]])
    weights = np.array([0.4, 1.0])

    def objective(x):
        residual = matrix @ x - np.array([3.0, -0.25])
        if robust:
            loss = np.where(abs(residual) <= 1, 0.5 * residual**2, abs(residual) - 0.5)
            derivative = np.clip(residual, -1, 1)
        else:
            loss, derivative = 0.5 * residual**2, residual
        return (
            weights @ loss + 0.1 * (x @ x),
            matrix.T @ (weights * derivative) + 0.2 * x,
        )

    expected = minimize(
        objective,
        np.zeros(2),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1, 1)] * 2,
        options={"gtol": 1e-12, "ftol": 1e-15},
    )
    assert expected.success
    params = {
        **parameters(False),
        "iterations": 100,
        "damping": 0.2e-6,
        "stopping": {"update_rtol": None, "objective_rtol": None},
    }
    if robust:
        params.update(robust_loss="huber", huber_delta_s=1e-8, irls_iterations=3)
    result = run(case, params)
    x = perturbation(result).ravel() / 1e-5
    np.testing.assert_allclose(x, expected.x, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(x, [0.125 if robust else 0.34375, 1.0], rtol=1e-8)
    gradient = objective(x)[1]
    assert np.linalg.norm(x - np.clip(x - gradient, -1, 1), np.inf) < 1e-9
    assert result.metrics["solver_optimality"]["converged"]
    costs = np.array([row["objective"] for row in result.metrics["iteration_history"]])
    assert np.all(np.diff(costs) <= 1e-12 * costs[:-1])


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("budget", ["max_forward_calls", "max_adjoint_calls"])
@pytest.mark.parametrize("limit", [0, 1, 2])
def test_retry_work_obeys_global_budgets_and_atomic_checkpoints(robust, budget, limit):
    case = huber_case() if robust else bounded_case()
    result = run(case, {**parameters(robust), "stopping": {budget: limit}})
    stop = result.metrics["stopping"]
    assert stop["work"][budget.removeprefix("max_")] <= limit
    assert not result.metrics["solver_optimality"]["converged"]
    assert stop["completed_iterations"] <= 30
    if "evaluation" in result.metrics:
        op = StraightRayProjector.from_case(case)
        residual = op.forward(perturbation(result)).reshape(op.ray_shape)
        residual -= case.measurement.delta_tof_s
        expected = np.linalg.norm(residual[case.measurement.valid_mask])
        assert result.metrics["data_residual_norm"] == pytest.approx(
            expected, rel=1e-9, abs=1e-18
        )


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_irls_restarts_do_not_extend_iteration_budget(iterations):
    result = run(huber_case(), {**parameters(True), "iterations": iterations})
    assert result.metrics["iterations"] <= iterations
    assert result.metrics["stop_reason"] == "max_iterations"
    assert not result.metrics["solver_optimality"]["converged"]


def heldout_case():
    case = huber_case()
    case.geometry.rx_pos_m = np.vstack(
        [case.geometry.rx_pos_m, case.geometry.rx_pos_m[0]]
    )
    case.measurement.delta_tof_s = np.array([[1e-8, 0, 1.5e-8], [0, 4e-8, 0]])
    case.measurement.valid_mask = np.array([[True, False, True], [False, True, False]])
    case.measurement.ray_weights = np.ones((2, 3))
    return case


def test_huber_reweighting_and_optimality_exclude_heldout_values_and_weights():
    case = heldout_case()
    params = {
        **parameters(True),
        "coverage_preconditioning": True,
        "evaluation": {"receiver_indices": [2]},
        "stopping": {"restore_best_validation": False},
    }
    first = run(case, params)
    changed = case.model_copy(deep=True)
    changed.measurement.delta_tof_s[:, 2] += 1e-4
    changed.measurement.ray_weights[:, 2] = 0.125
    second = run(changed, params)
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)
    np.testing.assert_array_equal(
        first.metrics["residual_curve"], second.metrics["residual_curve"]
    )
    assert first.metrics["solver_optimality"] == second.metrics["solver_optimality"]
    assert first.metrics["stopping"]["work"] == second.metrics["stopping"]["work"]


def test_optimality_certificate_belongs_to_selected_validation_checkpoint():
    result = run(
        heldout_case(),
        {**parameters(True), "evaluation": {"receiver_indices": [2]}},
    )
    np.testing.assert_allclose(perturbation(result), [[1.5e-5]], rtol=1e-10)
    assert result.metrics["stop_reason"] == "stationary_gradient"
    certificate = result.metrics["solver_optimality"]
    assert (
        certificate["iteration"]
        == result.metrics["stopping"]["selected_iteration"]
        == 1
    )
    assert certificate["verified"]
    assert not certificate["converged"]


@pytest.mark.parametrize("kind", ["identity", "laplacian"])
@pytest.mark.parametrize("restricted", [False, True])
@pytest.mark.parametrize("preconditioned", [False, True])
def test_quadratic_weights_roi_and_penalty_match_dense_augmented_reference(
    kind, restricted, preconditioned
):
    rng = np.random.default_rng(904)
    shape = (2, 3)
    matrix = rng.uniform(0.01, 1, (12, 6))
    observed = rng.normal(size=(12, 1))
    observed[2] = np.nan
    weights = rng.uniform(0.1, 1, (12, 1))
    weights[4] = 0
    roi = (
        np.array([[True, False, True], [True, True, False]])
        if restricted
        else np.ones(shape, bool)
    )
    penalty = np.eye(6)
    if kind == "laplacian":
        penalty = np.zeros((6, 6))
        for y, x in np.ndindex(shape):
            i = y * shape[1] + x
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                    penalty[i, yy * shape[1] + xx] = 1
                    penalty[i, i] -= 1

    class DenseOperator:
        def forward(self, x):
            return matrix @ x.ravel()

        def adjoint(self, data):
            return (matrix.T @ data.ravel()).reshape(shape)

    case = SimpleNamespace(
        geometry=SimpleNamespace(tx_pos_m=None, rx_pos_m=None),
        grid=SimpleNamespace(roi_mask=roi),
        ground_truth=SimpleNamespace(sound_speed_mps=None),
    )
    config = AlgorithmConfig(
        parameters={
            "iterations": 30,
            "stopping": {"update_rtol": None, "objective_rtol": None},
        }
    )
    control = InversionControl(
        case, config, observed, weights=weights, default_iterations=30
    )
    actual, metrics = projected_cg(
        DenseOperator(),
        control,
        initial=np.zeros(shape),
        reference=np.ones(shape),
        project=lambda x: np.where(roi, x, 0),
        to_image=lambda x: x,
        damping=0.4,
        regularization=kind,
        roi=roi,
        regularization_roi=roi,
        preconditioner=rng.uniform(0.2, 3, shape) if preconditioned else None,
    )
    rows = np.isfinite(observed.ravel()) & (weights.ravel() > 0)
    columns = roi.ravel()
    design = np.vstack(
        [
            np.sqrt(weights[rows]) * matrix[rows][:, columns],
            np.sqrt(0.4) * penalty[:, columns],
        ]
    )
    target = np.r_[np.sqrt(weights[rows].ravel()) * observed[rows].ravel(), np.zeros(6)]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    np.testing.assert_allclose(actual.ravel()[columns], expected, rtol=1e-9, atol=1e-11)
    assert metrics["stop_reason"] == "stationary_gradient"
    assert metrics["solver_optimality"]["verified"]
    assert metrics["solver_optimality"]["converged"]
    np.testing.assert_array_equal(actual[~roi], 0)
    assert metrics["stop_reason"] in {"stationary_gradient", "max_iterations"}
