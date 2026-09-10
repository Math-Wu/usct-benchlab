"""Independent dense references for the shared matrix-free inverse solvers."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from usctbench.core.stopping import StopPolicy, WorkLedger
from usctbench.solvers.least_squares import normal_step, regularizer


class DenseJacobian:
    def __init__(self, matrix, shape):
        self.matrix, self.shape = matrix, shape

    def forward(self, x):
        return self.matrix @ x.ravel()

    def adjoint(self, y):
        return (self.matrix.conj().T @ y).real.reshape(self.shape)


def controls(precision):
    work = WorkLedger(StopPolicy())
    return SimpleNamespace(
        observed=np.zeros(len(precision)),
        precision=precision,
        work=work,
        call=work.call,
    )


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("penalty", ["identity", "laplacian", (0.2, 1.3)])
def test_normal_step_matches_dense_augmented_lstsq(complex_data, penalty):
    rng = np.random.default_rng(641)
    shape = (3, 4)
    matrix = rng.normal(size=(40, 12))
    b = rng.normal(size=40)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        b = b + 1j * rng.normal(size=b.shape)
    weights = rng.uniform(0.2, 2, len(b))
    weights[::5] = 0
    roi = np.ones(shape, bool)
    roi[0, 0] = roi[-1, -1] = False
    current = rng.normal(size=shape)
    damping = 0.23
    reg = np.column_stack(
        [regularizer(e.reshape(shape), penalty).ravel() for e in np.eye(12)]
    )
    design = np.vstack(
        [
            np.sqrt(weights)[:, None] * matrix.real,
            np.sqrt(weights)[:, None] * matrix.imag,
            np.sqrt(damping) * reg,
        ]
    )[:, roi.ravel()]
    target = np.concatenate(
        [
            np.sqrt(weights) * b.real,
            np.sqrt(weights) * b.imag,
            -np.sqrt(damping) * reg @ current.ravel(),
        ]
    )
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    control = controls(weights)
    actual = normal_step(
        DenseJacobian(matrix, shape),
        weights * b,
        current,
        control,
        iterations=30,
        damping=damping,
        regularization=penalty,
        roi=roi,
    )
    np.testing.assert_allclose(actual[roi], expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(actual[~roi], 0)
    gradient = design.T @ (design @ actual[roi] - target)
    assert np.linalg.norm(gradient) < 1e-8
    assert control.inner_solver_history[-1]["converged"]
    assert control.inner_solver_history[-1]["true_relative_residual"] < 1e-7


def test_normal_cg_nonzero_squared_rhs_underflow_is_not_convergence():
    matrix = np.array([[1e-90]])
    residual = np.array([1e-90])
    jacobian = DenseJacobian(matrix, (1, 1))
    rhs = jacobian.adjoint(residual)
    assert rhs.item() > 0
    assert np.vdot(rhs, rhs).real == 0
    # The one-dimensional system has condition number one and exact step one.
    expected = residual[0] / matrix[0, 0]
    control = controls(np.ones(1))
    actual = normal_step(
        jacobian,
        residual,
        np.zeros((1, 1)),
        control,
        iterations=8,
        method="normal_cg",
    )
    record = control.inner_solver_history[-1]
    assert not record["converged"]
    assert record["stop_reason"] == "arithmetic_precision_limit"
    assert record["iterations"] == 0
    assert record["true_relative_residual"] == 1.0
    assert record["recursive_relative_residual"] == 1.0
    assert record["initial_normal_residual_norm"] == rhs.item()
    assert record["physical_normal_residual_norm"] == rhs.item()
    np.testing.assert_array_equal(actual, 0)
    assert abs(actual.item() - expected) == 1.0


def test_gaussian_postprocessing_can_turn_newton_step_into_ascent():
    # A convex, diagonal quadratic: smoothing a Newton step is not in general
    # equivalent to a positive-definite gradient preconditioner.
    gradient = np.array([-3.0, 0.6, 0.6, -0.5])
    hessian = np.diag([1000.0, 1.0, 1.0, 1.0])
    step = -np.linalg.solve(hessian, gradient).reshape(2, 2)
    smoothed = gaussian_filter(step, 1, mode="nearest")
    assert gradient @ step.ravel() < 0
    assert gradient @ smoothed.ravel() > 0


def test_normal_step_masks_nonfinite_excluded_jacobian_rows():
    class PartialJacobian(DenseJacobian):
        def forward(self, x):
            out = super().forward(x)
            out[-1] = np.nan
            return out

    matrix = np.vstack([np.eye(4), np.ones(4)])
    precision = np.array([1.0, 1, 1, 1, 0])
    result = normal_step(
        PartialJacobian(matrix, (2, 2)),
        np.array([1.0, 2, 3, 4, 0]),
        np.zeros((2, 2)),
        controls(precision),
        iterations=8,
    )
    np.testing.assert_allclose(result.ravel(), [1, 2, 3, 4])


def test_normal_step_does_not_silently_return_on_invalid_curvature():
    class BrokenAdjoint(DenseJacobian):
        def adjoint(self, y):
            return -super().adjoint(y)

    with pytest.raises(FloatingPointError, match="curvature"):
        normal_step(
            BrokenAdjoint(np.eye(4), (2, 2)),
            np.ones(4),
            np.zeros((2, 2)),
            controls(np.ones(4)),
            iterations=8,
        )


def test_nonlinear_solver_rejects_uphill_smoothing_and_recovers_dense_solution():
    from usctbench.algorithms._control import InversionControl
    from usctbench.core.schema import (
        AlgorithmConfig,
        GridSpec,
        GeometrySpec,
        MeasurementSpec,
        USCTCase,
    )
    from usctbench.operators.base import Linearization
    from usctbench.solvers.nonlinear import nonlinear_least_squares

    shape = (2, 2)
    initial = np.full(shape, 1 / 1500**2)
    matrix = np.diag(np.sqrt([1000.0, 1, 1, 1])) * 1e-6 / initial[0, 0]
    normalized_gradient = np.array([-3.0, 0.6, 0.6, -0.5])
    observed = (-0.02e-6 * normalized_gradient / np.sqrt([1000.0, 1, 1, 1])).reshape(
        shape
    )

    class Forward:
        background_builds = eikonal_solves = 0

        def linearize(self, model):
            self.background_builds += 1
            jacobian = DenseJacobian(matrix, shape)
            return Linearization(
                jacobian.forward(model - initial).reshape(shape),
                SimpleNamespace(
                    forward=jacobian.forward,
                    adjoint=lambda y: jacobian.adjoint(y.ravel()),
                ),
            )

    positions = [[-0.01, 0], [0.01, 0]]
    case = USCTCase(
        case_id="convex_newton_smoothing_counterexample",
        grid=GridSpec(shape=shape, spacing_m=(0.001, 0.001)),
        geometry=GeometrySpec(tx_pos_m=positions, rx_pos_m=positions),
        measurement=MeasurementSpec(domain="features", delta_tof_s=observed),
    )
    config = AlgorithmConfig(
        parameters={
            "stopping": {
                "max_iterations": 3,
                "target_relative_residual": 1e-8,
                "restore_best_validation": False,
            }
        }
    )
    control = InversionControl(case, config, observed, default_iterations=3)
    result, metrics = nonlinear_least_squares(
        Forward(),
        control,
        initial=initial,
        bounds=(1400, 1600),
        inner_iterations=8,
        damping=0,
        smooth_sigma=1,
    )
    expected = initial + np.linalg.solve(matrix, observed.ravel()).reshape(shape)
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    assert metrics["data_relative_residual"] < 1e-8
    assert metrics["direction_checks"][0]["smoothing_rejected"]
    assert metrics["direction_checks"][0]["raw_directional_derivative"] < 0
    assert all(
        row["projected_directional_derivative"] < 0
        for row in metrics["line_search_history"]
        if row["accepted"]
    )


def test_saved_iterate_audit_checks_complete_objective_gradient():
    import importlib.util
    from pathlib import Path
    from usctbench.operators.base import Linearization

    path = Path(__file__).resolve().parents[2] / "scripts/_inverse_solver_audit.py"
    spec = importlib.util.spec_from_file_location("inverse_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = np.diag([1000.0, 500, 400, 600])
    initial = np.full((2, 2), 1 / 1500**2)
    state = initial * np.array([[1.01, 0.99], [0.97, 1.02]])
    jacobian = DenseJacobian(matrix, state.shape)
    forward = SimpleNamespace(
        linearize=lambda m: Linearization(matrix @ (m - initial).ravel(), jacobian)
    )
    control = controls(np.array([1.0, 0.5, 0, 1]))
    control.observed = np.array([1e-6, -2e-6, 1e3, 3e-6])
    control.split = SimpleNamespace(train=control.precision > 0)
    control.weighted_residual = (
        lambda p: np.where(control.split.train, control.observed - p, 0)
        * control.precision
    )
    report = module.audit_iterate(
        forward,
        control,
        state=state,
        reference=initial,
        bounds=(1300, 1700),
        damping=1e3,
        regularization=(0.5, 1.3),
        smooth_sigma=1,
        gradient_steps=(1.0, 0.01, 0.001),
    )
    assert not report["ground_truth_used"]
    assert (
        max(x["relative_error"] for x in report["objective_gradient_differences"])
        < 1e-10
    )
    assert report["gradient_fd_requested_steps"] == [1.0, 0.01, 0.001]
    assert [r["step"] for r in report["objective_gradient_differences"]] == [
        1.0,
        0.01,
        0.001,
    ]
    assert all(x["inner"]["converged"] for x in report["normal_subproblems"])
    for invalid in ([], [0.0], [-1.0], [np.nan], [[1.0]]):
        with pytest.raises(ValueError, match="gradient steps"):
            module.audit_iterate(
                None,
                None,
                state=None,
                reference=None,
                bounds=None,
                damping=None,
                regularization=None,
                gradient_steps=invalid,
            )
