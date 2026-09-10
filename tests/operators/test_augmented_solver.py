"""Independent dense/SVD contracts for matrix-free augmented least squares."""

from types import SimpleNamespace

import numpy as np
import pytest

from usctbench.core.stopping import BudgetExhausted, StopPolicy, WorkLedger
from usctbench.solvers.least_squares import normal_step


class DenseJacobian:
    def __init__(self, matrix, shape, data_shape=None):
        self.matrix = np.asarray(matrix)
        self.shape = shape
        self.data_shape = data_shape or (self.matrix.shape[0],)
        self.forward_calls = 0
        self.adjoint_calls = 0

    def forward(self, image):
        self.forward_calls += 1
        assert np.isrealobj(image)
        return (self.matrix @ np.asarray(image).ravel()).reshape(self.data_shape)

    def adjoint(self, data):
        self.adjoint_calls += 1
        return (self.matrix.conj().T @ np.asarray(data).ravel()).real.reshape(
            self.shape
        )


class RectangularPenalty:
    def __init__(self, matrix, shape):
        self.matrix = np.asarray(matrix)
        self.shape = shape

    def forward(self, image):
        return self.matrix @ np.asarray(image).ravel()

    def adjoint(self, data):
        return (self.matrix.T @ np.asarray(data).ravel()).reshape(self.shape)


@pytest.fixture(params=["lsmr", "lsqr"])
def method(request):
    return request.param


def controls(precision, *, policy=None, clock=None):
    work = WorkLedger(policy or StopPolicy(), **({"clock": clock} if clock else {}))
    return SimpleNamespace(
        observed=np.zeros(np.shape(precision)),
        precision=np.asarray(precision),
        work=work,
        call=work.call,
    )


def dense_augmented(matrix, residual, precision, current, penalty, damping, roi=None):
    """Assemble real rows directly, excluding masked data before arithmetic."""
    valid = np.asarray(precision).ravel() > 0
    root = np.sqrt(np.asarray(precision).ravel()[valid])
    data_matrix = root[:, None] * matrix[valid]
    data_rhs = root * np.asarray(residual).ravel()[valid]
    design = np.vstack((data_matrix.real, data_matrix.imag, np.sqrt(damping) * penalty))
    target = np.concatenate(
        (
            data_rhs.real,
            data_rhs.imag,
            -np.sqrt(damping) * (penalty @ current.ravel()),
        )
    )
    active = np.ones(current.size, dtype=bool) if roi is None else roi.ravel()
    return design[:, active], target


def svd_minimum_norm(design, target):
    left, values, right = np.linalg.svd(design, full_matrices=False)
    keep = values > np.finfo(float).eps * max(design.shape) * values[0]
    return right[keep].T @ ((left[:, keep].T @ target) / values[keep])


def assert_record(control, method, design, target, step, rtol):
    record = control.inner_solver_history[-1]
    gradient = design.T @ (design @ np.asarray(step).ravel() - target)
    denominator = max(np.linalg.norm(design.T @ target), np.finfo(float).tiny)
    relative = np.linalg.norm(gradient) / denominator
    assert record["method"] == f"augmented_{method}"
    assert np.isfinite(record["true_relative_residual"])
    np.testing.assert_allclose(
        record["true_relative_residual"], relative, rtol=5e-6, atol=2e-14
    )
    assert isinstance(record["converged"], (bool, np.bool_))
    if record["converged"]:
        assert relative <= rtol
        assert record["true_relative_residual"] <= rtol
    assert isinstance(record["stop_reason"], str) and record["stop_reason"].strip()
    return record


def laplacian_matrix(shape, axis_weights=(1.0, 1.0)):
    # Independent edge differences give the replicated-edge, symmetric stencil.
    matrix = np.zeros((np.prod(shape), np.prod(shape)))
    for index in np.ndindex(shape):
        first = np.ravel_multi_index(index, shape)
        for axis, weight in enumerate(axis_weights):
            neighbor = list(index)
            neighbor[axis] += 1
            if neighbor[axis] >= shape[axis]:
                continue
            second = np.ravel_multi_index(tuple(neighbor), shape)
            matrix[first, first] -= weight
            matrix[second, second] -= weight
            matrix[first, second] += weight
            matrix[second, first] += weight
    return matrix


@pytest.mark.parametrize(
    ("complex_data", "regularization"),
    [
        pytest.param(False, "identity", id="real-identity"),
        pytest.param(True, "laplacian", id="complex-laplacian"),
        pytest.param(False, (0.2, 1.3), id="real-anisotropic"),
    ],
)
def test_weighted_augmented_solution_matches_dense_reference(
    method, complex_data, regularization
):
    rng = np.random.default_rng(2107)
    shape = (3, 4)
    matrix = rng.normal(size=(40, 12))
    residual = rng.normal(size=40)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        residual = residual + 1j * rng.normal(size=residual.shape)
    precision = rng.uniform(0.05, 4, 40)
    precision[::6] = 0
    current = rng.normal(size=shape)
    penalty = (
        np.eye(current.size)
        if regularization == "identity"
        else laplacian_matrix(
            shape, regularization if isinstance(regularization, tuple) else (1, 1)
        )
    )
    damping = 0.37
    design, target = dense_augmented(
        matrix, residual, precision, current, penalty, damping
    )
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    control = controls(precision.reshape(5, 8))
    weighted = (precision * residual).reshape(5, 8)
    original_current, original_weighted = current.copy(), weighted.copy()
    step = normal_step(
        DenseJacobian(matrix, shape, (5, 8)),
        weighted,
        current,
        control,
        iterations=100,
        damping=damping,
        regularization=regularization,
        method=method,
        rtol=1e-11,
        atol=1e-14,
        btol=1e-14,
    )
    assert step.shape == shape and np.isrealobj(step)
    np.testing.assert_allclose(step.ravel(), expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(weighted, original_weighted)
    assert assert_record(control, method, design, target, step, 1e-11)["converged"]


@pytest.mark.parametrize("rows", [3, 15], ids=["underdetermined", "rank_deficient"])
def test_unpreconditioned_rank_deficiency_returns_svd_minimum_norm(method, rows):
    rng = np.random.default_rng(2241)
    shape = (2, 3)
    matrix = rng.normal(size=(rows, 3)) @ rng.normal(size=(3, 6))
    matrix[:, 4] = 2 * matrix[:, 1]
    matrix[:, 5] = 0
    residual = rng.normal(size=rows)
    precision = rng.uniform(0.2, 2, rows)
    current = rng.normal(size=shape)
    design, target = dense_augmented(matrix, residual, precision, current, np.eye(6), 0)
    expected = svd_minimum_norm(design, target)
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        precision * residual,
        current,
        control,
        iterations=100,
        method=method,
        preconditioner="none",
        rtol=1e-12,
        atol=0,
        btol=0,
        conlim=1e16,
    )
    np.testing.assert_allclose(step.ravel(), expected, rtol=2e-11, atol=2e-12)
    _, values, right = np.linalg.svd(design, full_matrices=True)
    rank = np.count_nonzero(
        values > values[0] * max(design.shape) * np.finfo(float).eps
    )
    assert rank == 3
    np.testing.assert_allclose(right[rank:] @ step.ravel(), 0, atol=2e-12)
    assert step.ravel()[-1] == 0
    assert assert_record(control, method, design, target, step, 1e-12)["converged"]


@pytest.mark.parametrize("units", [1e-12, 1.0, 1e12])
def test_ill_conditioned_solution_is_invariant_to_data_units(method, units):
    rng = np.random.default_rng(3119)
    shape = (2, 4)
    left, _ = np.linalg.qr(rng.normal(size=(16, 8)))
    right, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    singular_values = np.geomspace(1, 1e-6, 8)
    matrix = units * ((left * singular_values) @ right.T)
    truth = rng.normal(size=8)
    residual = matrix @ truth
    precision = np.ones(16)
    current = np.zeros(shape)
    design, target = dense_augmented(matrix, residual, precision, current, np.eye(8), 0)
    expected = svd_minimum_norm(design, target)
    np.testing.assert_allclose(expected, truth, rtol=2e-9, atol=2e-10)
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        residual,
        current,
        control,
        iterations=160,
        method=method,
        preconditioner="none",
        rtol=1e-10,
        atol=0,
        btol=0,
        conlim=1e14,
    )
    np.testing.assert_allclose(step.ravel(), expected, rtol=2e-8, atol=2e-9)
    assert assert_record(control, method, design, target, step, 1e-10)["converged"]


@pytest.mark.parametrize("units", [1e-12, 1e12])
@pytest.mark.parametrize("preconditioner", ["none", "column_rms"])
def test_global_unit_scaling_preserves_data_and_rectangular_penalty_balance(
    method, units, preconditioner
):
    rng = np.random.default_rng(3209)
    shape = (2, 4)
    matrix = rng.normal(size=(20, 8)) + 1j * rng.normal(size=(20, 8))
    residual = rng.normal(size=20) + 1j * rng.normal(size=20)
    precision = rng.uniform(0.1, 3, 20)
    precision[::5] = 0
    current = rng.normal(size=shape)
    penalty = rng.normal(size=(5, 8))
    roi = np.ones(shape, dtype=bool)
    roi[0, 0] = roi[-1, -1] = False
    baseline, baseline_target = dense_augmented(
        matrix, residual, precision, current, penalty, 0.6, roi
    )
    expected = svd_minimum_norm(baseline, baseline_target)
    matrix, residual, damping = units * matrix, units * residual, units**2 * 0.6
    design, target = dense_augmented(
        matrix, residual, precision, current, penalty, damping, roi
    )
    np.testing.assert_allclose(
        svd_minimum_norm(design, target), expected, rtol=2e-12, atol=2e-13
    )
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        precision * residual,
        current,
        control,
        iterations=100,
        damping=damping,
        regularization=RectangularPenalty(penalty, shape),
        roi=roi,
        method=method,
        preconditioner=preconditioner,
        preconditioner_probes=16,
        preconditioner_seed=13,
        rtol=1e-11,
        atol=1e-14,
        btol=1e-14,
    )
    np.testing.assert_allclose(step[roi], expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(step[~roi], 0)
    assert assert_record(control, method, design, target, step[roi], 1e-11)["converged"]


@pytest.mark.parametrize("complex_data", [False, True])
def test_masked_nonfinite_residuals_and_forward_rows_are_excluded(method, complex_data):
    rng = np.random.default_rng(4503)
    shape = (2, 3)
    matrix = rng.normal(size=(18, 6))
    residual = rng.normal(size=18)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        residual = residual + 1j * rng.normal(size=residual.shape)
    precision = rng.uniform(0.2, 2, 18)
    precision[::3] = 0
    excluded = precision == 0
    weighted = precision * residual
    weighted[excluded] = np.nan

    class PartiallyDefinedJacobian(DenseJacobian):
        def forward(self, image):
            out = super().forward(image)
            out[excluded] = np.nan
            return out

        def adjoint(self, data):
            assert np.isfinite(data).all()
            np.testing.assert_array_equal(np.asarray(data)[excluded], 0)
            return super().adjoint(data)

    current = rng.normal(size=shape)
    design, target = dense_augmented(
        matrix, residual, precision, current, np.eye(6), 0.4
    )
    expected = svd_minimum_norm(design, target)
    control = controls(precision)
    step = normal_step(
        PartiallyDefinedJacobian(matrix, shape),
        weighted,
        current,
        control,
        iterations=80,
        damping=0.4,
        method=method,
        rtol=1e-11,
        atol=1e-14,
        btol=1e-14,
    )
    assert np.isfinite(step).all()
    np.testing.assert_allclose(step.ravel(), expected, rtol=2e-10, atol=2e-11)
    assert assert_record(control, method, design, target, step, 1e-11)["converged"]


@pytest.mark.parametrize(
    ("complex_data", "preconditioner"),
    [
        pytest.param(False, "none", id="real-unpreconditioned"),
        pytest.param(True, "column_rms", id="complex-column-rms"),
    ],
)
def test_roi_restricts_columns_but_retains_full_current_in_rectangular_penalty(
    method, complex_data, preconditioner
):
    rng = np.random.default_rng(5227)
    shape = (3, 4)
    matrix = rng.normal(size=(28, 12))
    residual = rng.normal(size=28)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        residual = residual + 1j * rng.normal(size=residual.shape)
    precision = rng.uniform(0.1, 3, 28)
    precision[::7] = 0
    roi = np.ones(shape, dtype=bool)
    roi[0, :2] = roi[-1, -1] = False
    current = rng.normal(size=shape)
    current[~roi] = [3, -4, 5]
    penalty = rng.normal(size=(7, 12))
    damping = 0.7
    design, target = dense_augmented(
        matrix, residual, precision, current, penalty, damping, roi
    )
    expected = svd_minimum_norm(design, target)
    _, projected_target = dense_augmented(
        matrix, residual, precision, np.where(roi, current, 0), penalty, damping, roi
    )
    assert np.linalg.norm(expected - svd_minimum_norm(design, projected_target)) > 0.1
    control = controls(precision)
    original_current, original_roi = current.copy(), roi.copy()
    step = normal_step(
        DenseJacobian(matrix, shape),
        precision * residual,
        current,
        control,
        iterations=120,
        damping=damping,
        regularization=RectangularPenalty(penalty, shape),
        roi=roi,
        method=method,
        preconditioner=preconditioner,
        preconditioner_probes=16,
        preconditioner_seed=21,
        rtol=1e-11,
        atol=1e-14,
        btol=1e-14,
    )
    np.testing.assert_allclose(step[roi], expected, rtol=2e-10, atol=2e-11)
    np.testing.assert_array_equal(step[~roi], 0)
    np.testing.assert_array_equal(current, original_current)
    np.testing.assert_array_equal(roi, original_roi)
    assert assert_record(control, method, design, target, step[roi], 1e-11)["converged"]


@pytest.mark.parametrize("complex_data", [False, True])
def test_right_preconditioner_preserves_unique_solution_and_accounts_for_calls(
    method, complex_data
):
    rng = np.random.default_rng(6131)
    shape = (2, 4)
    matrix = rng.normal(size=(24, 8))
    residual = rng.normal(size=24)
    if complex_data:
        matrix = matrix + 1j * rng.normal(size=matrix.shape)
        residual = residual + 1j * rng.normal(size=residual.shape)
    matrix = matrix * np.geomspace(1e-2, 1e2, 8)
    precision = rng.uniform(0.1, 2, 24)
    current = rng.normal(size=shape)
    design, target = dense_augmented(
        matrix, residual, precision, current, np.eye(8), 0.05
    )
    expected = svd_minimum_norm(design, target)
    solutions = []
    for preconditioner in ("none", "column_rms", "column_rms"):
        jacobian = DenseJacobian(matrix, shape)
        control = controls(precision)
        step = normal_step(
            jacobian,
            precision * residual,
            current,
            control,
            iterations=160,
            damping=0.05,
            method=method,
            preconditioner=preconditioner,
            preconditioner_probes=16,
            preconditioner_seed=7,
            rtol=1e-11,
            atol=0,
            btol=0,
            conlim=1e14,
        )
        np.testing.assert_allclose(step.ravel(), expected, rtol=2e-9, atol=2e-10)
        assert assert_record(control, method, design, target, step, 1e-11)["converged"]
        assert control.work.counts["forward_calls"] == jacobian.forward_calls
        assert control.work.counts["adjoint_calls"] == jacobian.adjoint_calls
        assert jacobian.forward_calls > 0 and jacobian.adjoint_calls > 0
        solutions.append(step)
    np.testing.assert_allclose(solutions[0], solutions[1], rtol=2e-9, atol=2e-10)
    np.testing.assert_array_equal(solutions[1], solutions[2])


@pytest.mark.parametrize(
    ("rhs_kind", "preconditioner"),
    [
        ("zero_data", "none"),
        ("orthogonal_data", "column_rms"),
        ("penalty_balance", "column_rms"),
    ],
)
def test_zero_normal_rhs_returns_zero_step_and_converges(
    method, rhs_kind, preconditioner
):
    shape = (2, 2)
    matrix = np.vstack((np.eye(4), np.zeros((1, 4))))
    residual = np.zeros(5)
    current = np.arange(1.0, 5.0).reshape(shape)
    damping = 0.0
    if rhs_kind == "orthogonal_data":
        residual[-1] = 3
    elif rhs_kind == "penalty_balance":
        damping = 1.0
        residual[:4] = current.ravel()
    precision = np.ones(5)
    design, target = dense_augmented(
        matrix, residual, precision, current, np.eye(4), damping
    )
    np.testing.assert_array_equal(design.T @ target, 0)
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        residual,
        current,
        control,
        iterations=20,
        damping=damping,
        method=method,
        preconditioner=preconditioner,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(step, np.zeros(shape))
    record = assert_record(control, method, design, target, step, 1e-12)
    assert record["converged"] and record["true_relative_residual"] == 0


@pytest.mark.parametrize(
    ("iterations", "preconditioner"), [(1, "none"), (2, "column_rms")]
)
def test_iteration_cap_cannot_claim_true_convergence(
    method, iterations, preconditioner
):
    rng = np.random.default_rng(7213)
    shape = (3, 4)
    matrix = rng.normal(size=(30, 12)) * np.geomspace(0.1, 100, 12)
    residual = rng.normal(size=30)
    precision = rng.uniform(0.2, 2, 30)
    current = rng.normal(size=shape)
    design, target = dense_augmented(
        matrix, residual, precision, current, np.eye(12), 0.2
    )
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        precision * residual,
        current,
        control,
        iterations=iterations,
        damping=0.2,
        method=method,
        preconditioner=preconditioner,
        preconditioner_seed=11,
        rtol=1e-12,
        atol=0,
        btol=0,
        conlim=1e14,
    )
    record = assert_record(control, method, design, target, step, 1e-12)
    assert record["true_relative_residual"] > 1e-6
    assert not record["converged"]
    assert record["iterations"] == iterations
    assert "iteration" in record["stop_reason"]
    assert control.work.counts.get("inner_iterations", 0) == iterations


def test_scipy_loose_stopping_tolerance_is_not_a_convergence_certificate(method):
    rng = np.random.default_rng(7319)
    shape = (2, 3)
    matrix = rng.normal(size=(14, 6)) * np.geomspace(1, 100, 6)
    residual = rng.normal(size=14)
    precision = np.ones(14)
    current = np.zeros(shape)
    design, target = dense_augmented(matrix, residual, precision, current, np.eye(6), 0)
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        residual,
        current,
        control,
        iterations=100,
        method=method,
        rtol=1e-12,
        atol=0.9,
        btol=0.9,
    )
    record = assert_record(control, method, design, target, step, 1e-12)
    assert record["true_relative_residual"] > 1e-6
    assert not record["converged"]


@pytest.mark.parametrize("all_data_masked", [False, True])
def test_rectangular_penalty_nullspace_matches_same_target_minimum_step_norm(
    method, all_data_masked
):
    rng = np.random.default_rng(7507)
    shape = (2, 3)
    penalty = np.diff(np.eye(6), axis=0)
    matrix = penalty[[0, 2, 4]]
    residual = rng.normal(size=3)
    precision = np.zeros(3) if all_data_masked else np.array([0.5, 2.0, 1.5])
    current = rng.normal(size=shape) + 7.0
    damping = 0.3
    design, target = dense_augmented(
        matrix, residual, precision, current, penalty, damping
    )
    expected = svd_minimum_norm(design, target)
    assert np.linalg.matrix_rank(design) == 5
    np.testing.assert_allclose(design @ np.ones(6), 0, atol=1e-15)
    weighted = precision * residual
    if all_data_masked:
        weighted[:] = np.nan
    steps = []
    # A common-nullspace shift changes neither this target nor the minimum-norm
    # step. It must NOT be removed by minimizing the norm of current + step.
    for shift in (0.0, 13.0):
        shifted = current + shift
        control = controls(precision)
        step = normal_step(
            DenseJacobian(matrix, shape),
            weighted,
            shifted,
            control,
            iterations=100,
            damping=damping,
            regularization=RectangularPenalty(penalty, shape),
            method=method,
            preconditioner="none",
            rtol=1e-11,
            atol=1e-14,
            btol=1e-14,
            conlim=1e14,
        )
        np.testing.assert_allclose(step.ravel(), expected, rtol=2e-10, atol=2e-11)
        np.testing.assert_allclose(step.sum(), 0, atol=2e-12)
        np.testing.assert_allclose((shifted + step).mean(), shifted.mean(), atol=2e-12)
        assert assert_record(control, method, design, target, step, 1e-11)["converged"]
        steps.append(step)
    np.testing.assert_allclose(steps[0], steps[1], rtol=2e-10, atol=2e-11)


def test_condition_limit_returns_uncertified_step_with_explicit_reason(method):
    shape = (2, 4)
    matrix = np.diag(np.geomspace(1, 1e5, 8))
    residual = np.arange(1.0, 9.0)
    precision = np.ones(8)
    current = np.zeros(shape)
    design, target = dense_augmented(matrix, residual, precision, current, np.eye(8), 0)
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, shape),
        residual,
        current,
        control,
        iterations=100,
        method=method,
        rtol=1e-14,
        atol=0,
        btol=0,
        conlim=2,
    )
    assert np.isfinite(step).all()
    record = assert_record(control, method, design, target, step, 1e-14)
    assert not record["converged"]
    assert record["true_relative_residual"] > 1e-6
    assert record["stop_reason"] == "condition_limit"
    assert record["scipy_stop_code"] == 3
    assert 0 < record["iterations"] < 100
    assert np.linalg.norm(step.ravel() - svd_minimum_norm(design, target)) > 1e-4


@pytest.mark.parametrize(
    ("preconditioner", "counter", "limit"),
    [
        pytest.param("none", "forward_calls", 0, id="before-first-forward"),
        pytest.param("none", "adjoint_calls", 0, id="before-first-adjoint"),
        pytest.param("none", "forward_calls", 2, id="solver-forward-cap"),
        pytest.param("none", "adjoint_calls", 3, id="solver-adjoint-cap"),
        pytest.param("column_rms", "adjoint_calls", 3, id="preconditioner-setup-cap"),
    ],
)
def test_operator_budgets_include_preconditioner_setup(
    method, preconditioner, counter, limit
):
    rng = np.random.default_rng(8101)
    jacobian = DenseJacobian(rng.normal(size=(30, 16)), (4, 4))
    control = controls(np.ones(30), policy=StopPolicy(**{f"max_{counter}": limit}))
    current = rng.normal(size=(4, 4))
    original = current.copy()
    with pytest.raises(BudgetExhausted, match=f"{counter}_budget"):
        normal_step(
            jacobian,
            rng.normal(size=30),
            current,
            control,
            iterations=100,
            method=method,
            preconditioner=preconditioner,
            preconditioner_probes=8,
            rtol=1e-12,
            atol=0,
            btol=0,
        )
    assert control.work.counts[counter] == limit
    assert control.work.counts["forward_calls"] == jacobian.forward_calls
    assert control.work.counts["adjoint_calls"] == jacobian.adjoint_calls
    assert not any(
        row["converged"] for row in getattr(control, "inner_solver_history", [])
    )
    np.testing.assert_array_equal(current, original)


# Expired budgets stop before backend dispatch or preconditioner setup.
@pytest.mark.parametrize(
    ("method", "preconditioner"), [("lsmr", "none"), ("lsqr", "column_rms")]
)
def test_elapsed_budget_is_checked_before_operator_work(method, preconditioner):
    control = controls(
        np.ones(4), policy=StopPolicy(max_elapsed_s=1), clock=lambda: 0.0
    )
    control.work.clock = lambda: 1.0
    jacobian = DenseJacobian(np.eye(4), (2, 2))
    with pytest.raises(BudgetExhausted, match="time_budget"):
        normal_step(
            jacobian,
            np.ones(4),
            np.zeros((2, 2)),
            control,
            iterations=20,
            method=method,
            preconditioner=preconditioner,
        )
    assert jacobian.forward_calls == jacobian.adjoint_calls == 0
    assert (
        control.work.counts["forward_calls"]
        == control.work.counts["adjoint_calls"]
        == 0
    )


# Validation is shared before backend dispatch; repeat inputs, not backends.
@pytest.mark.parametrize("method", ["lsmr"])
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("iterations", 0),
        ("iterations", -1),
        ("iterations", 1.5),
        ("iterations", True),
        ("damping", -1.0),
        ("damping", np.nan),
        ("damping", np.inf),
        ("rtol", -1.0),
        ("rtol", np.nan),
        ("rtol", np.inf),
        ("atol", -1.0),
        ("atol", np.nan),
        ("atol", np.inf),
        ("btol", -1.0),
        ("btol", np.nan),
        ("btol", np.inf),
        ("conlim", -1.0),
        ("conlim", np.nan),
        ("preconditioner", "unknown"),
        ("preconditioner_probes", 0),
        ("preconditioner_probes", -1),
        ("preconditioner_probes", 1.5),
        ("preconditioner_probes", True),
        ("preconditioner_seed", -1),
        ("preconditioner_seed", 1.5),
    ],
)
def test_invalid_augmented_settings_fail_before_operator_work(method, name, value):
    jacobian = DenseJacobian(np.eye(4), (2, 2))
    control = controls(np.ones(4))
    settings = {"iterations": 20, "method": method, "preconditioner": "column_rms"}
    settings[name] = value
    with pytest.raises((ValueError, TypeError)):
        normal_step(jacobian, np.ones(4), np.zeros((2, 2)), control, **settings)
    assert jacobian.forward_calls == jacobian.adjoint_calls == 0


# The same seed guard must also run before the stationary-RHS early return.
@pytest.mark.parametrize("method", ["lsmr"])
@pytest.mark.parametrize("seed", [-1, 1.5])
def test_invalid_preconditioner_seed_is_rejected_even_for_stationary_rhs(method, seed):
    with pytest.raises((ValueError, TypeError)):
        normal_step(
            DenseJacobian(np.eye(4), (2, 2)),
            np.zeros(4),
            np.zeros((2, 2)),
            controls(np.ones(4)),
            iterations=20,
            method=method,
            preconditioner="column_rms",
            preconditioner_seed=seed,
        )


def test_unknown_method_is_rejected():
    jacobian = DenseJacobian(np.eye(4), (2, 2))
    with pytest.raises(ValueError, match="method|solver"):
        normal_step(
            jacobian,
            np.ones(4),
            np.zeros((2, 2)),
            controls(np.ones(4)),
            iterations=20,
            method="not_a_solver",
        )
    assert jacobian.forward_calls == jacobian.adjoint_calls == 0


def test_existing_default_remains_normal_cg():
    matrix = np.diag([1.0, 2, 3, 4])
    precision = np.array([0.25, 1.0, 2.0, 4.0])
    residual = np.array([1.0, -2.0, 3.0, -4.0])
    current = np.ones((2, 2))
    control = controls(precision)
    step = normal_step(
        DenseJacobian(matrix, current.shape),
        precision * residual,
        current,
        control,
        iterations=20,
        damping=0.5,
    )
    design, target = dense_augmented(
        matrix, residual, precision, current, np.eye(4), 0.5
    )
    np.testing.assert_allclose(
        step.ravel(), svd_minimum_norm(design, target), rtol=1e-11, atol=1e-12
    )
    assert control.inner_solver_history[-1]["method"] == "real_parameter_normal_cg"


def test_small_normal_residual_is_not_parameter_accuracy(method):
    matrix = np.diag([1.0, 1e-6])
    truth = np.ones(2)
    observed = matrix @ truth
    legacy_control = controls(np.ones(2))
    legacy = normal_step(
        DenseJacobian(matrix, (1, 2)),
        observed,
        np.zeros((1, 2)),
        legacy_control,
        iterations=20,
    )
    assert legacy_control.inner_solver_history[-1]["true_relative_residual"] < 1e-11
    assert np.linalg.norm(legacy.ravel() - truth) > 0.9
    control = controls(np.ones(2))
    step = normal_step(
        DenseJacobian(matrix, (1, 2)),
        observed,
        np.zeros((1, 2)),
        control,
        iterations=20,
        method=method,
    )
    np.testing.assert_allclose(step.ravel(), truth, rtol=1e-8, atol=1e-9)
    assert control.inner_solver_history[-1]["converged"]


def test_condition_stop_is_not_convergence_even_with_loose_normal_target(method):
    matrix = np.diag(np.geomspace(1, 1e5, 8))
    control = controls(np.ones(8))
    normal_step(
        DenseJacobian(matrix, (2, 4)),
        np.arange(1.0, 9),
        np.zeros((2, 4)),
        control,
        iterations=100,
        method=method,
        conlim=2,
        rtol=1.0,
        atol=0,
        btol=0,
    )
    record = control.inner_solver_history[-1]
    assert record["normal_residual_target_met"]
    assert record["scipy_stop_code"] == 3
    assert not record["converged"]
    assert record["stop_reason"] == "condition_limit"


def test_cancellation_cannot_produce_false_zero_normal_certificate(method):
    control = controls(np.ones(2))
    step = normal_step(
        DenseJacobian(np.ones((2, 1)), (1, 1)),
        np.array([1e15, -1e15 + 2]),
        np.zeros((1, 1)),
        control,
        iterations=40,
        method=method,
    )
    value = step.item()
    record = control.inner_solver_history[-1]
    exact_relative = abs(value - 1)
    assert exact_relative > 1e-3
    assert not record["converged"]
    np.testing.assert_allclose(
        record["true_relative_residual"], exact_relative, atol=1e-15
    )
    np.testing.assert_allclose(
        record["quadratic_objective_reduction"], 2 * value - value**2, atol=1e-14
    )


@pytest.mark.parametrize("preconditioner", ["none", "column_rms"])
def test_large_left_nullspace_rhs_does_not_overflow_scaling(method, preconditioner):
    control = controls(np.ones(2))
    step = normal_step(
        DenseJacobian(np.array([[1.0], [0.0]]), (1, 1)),
        np.array([1.0, 1e80]),
        np.zeros((1, 1)),
        control,
        iterations=40,
        method=method,
        preconditioner=preconditioner,
    )
    np.testing.assert_allclose(step.item(), 1, atol=1e-12)
    assert control.inner_solver_history[-1]["converged"]


def test_tiny_nonzero_normal_rhs_does_not_underflow_to_stationarity(method):
    control = controls(np.ones(1))
    step = normal_step(
        DenseJacobian(np.array([[1e-170]]), (1, 1)),
        np.array([1e-170]),
        np.zeros((1, 1)),
        control,
        iterations=40,
        method=method,
    )
    np.testing.assert_allclose(step.item(), 1, atol=1e-12)
    record = control.inner_solver_history[-1]
    assert record["converged"]
    assert record["iterations"] > 0
