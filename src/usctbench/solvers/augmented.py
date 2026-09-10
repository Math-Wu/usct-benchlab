"""Golub-Kahan least squares without explicitly iterating the normal equations.

The physical step solves [sqrt(W) J; sqrt(damping) L] d =
[weighted_residual / sqrt(W); -sqrt(damping) L current]. Complex data are
real-stacked because the model parameters are real. No ridge is silently added.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import LinearOperator, lsmr, lsqr

from usctbench.core.stopping import BudgetExhausted


def augmented_step(
    jacobian,
    residual,
    current,
    control,
    *,
    iterations,
    damping=0.0,
    regularization="identity",
    roi=None,
    method="lsmr",
    rtol=1e-7,
    atol=1e-10,
    btol=1e-10,
    conlim=1e8,
    preconditioner="none",
    preconditioner_probes=8,
    preconditioner_seed=0,
):
    """Return an inexact GN step and append an independently checked certificate.

    ``iterations`` is a work cap, never a convergence declaration. ``rtol``
    applies to the *unscaled* physical normal residual relative to its initial
    value. SciPy's atol/btol instead test augmented backward errors. Either
    stopping test can fail while the other passes; both are reported.

    Optional column_rms scaling uses training-only adjoint Rademacher probes.
    It transforms BOTH data and penalty columns. In a rank-deficient problem
    it changes which minimum-norm solution is selected, not the objective.
    """
    from usctbench.solvers.least_squares import regularizer

    if method not in {"lsmr", "lsqr"}:
        raise ValueError("augmented method must be lsmr or lsqr")
    for name, value in (
        ("iterations", iterations),
        ("preconditioner_probes", preconditioner_probes),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("damping", damping),
        ("rtol", rtol),
        ("atol", atol),
        ("btol", btol),
        ("conlim", conlim),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if preconditioner not in {"none", "column_rms"}:
        raise ValueError("preconditioner must be none or column_rms")
    if (
        isinstance(preconditioner_seed, bool)
        or not isinstance(preconditioner_seed, (int, np.integer))
        or preconditioner_seed < 0
    ):
        raise ValueError("preconditioner_seed must be a nonnegative integer")
    current = np.asarray(current)
    if np.iscomplexobj(current) or not np.isfinite(current).all():
        raise ValueError("current must be a finite real model")
    current = current.astype(float, copy=False)
    active = np.ones(current.shape, bool) if roi is None else np.asarray(roi, bool)
    if active.shape != current.shape:
        raise ValueError("ROI shape must match model")
    precision = np.asarray(control.precision, dtype=float)
    if (
        precision.shape != control.observed.shape
        or not np.isfinite(precision).all()
        or np.any(precision < 0)
    ):
        raise ValueError("precision must match observations and be finite/nonnegative")
    residual = np.asarray(residual).reshape(precision.shape)
    selected = precision > 0
    sqrt_w = np.sqrt(precision[selected])
    complex_data = np.iscomplexobj(residual) or np.iscomplexobj(control.observed)
    ndata, nmodel = sqrt_w.size, int(active.sum())
    data_rows = ndata * (2 if complex_data else 1)
    sqrt_damping = np.sqrt(damping)
    reg_current = (
        np.asarray(regularizer(current, regularization)) if damping else np.empty(0)
    )
    reg_shape = reg_current.shape
    nreg = reg_current.size

    def checked(value):
        value = np.asarray(value)
        if np.iscomplexobj(value) or not np.isfinite(value).all():
            raise FloatingPointError("nonfinite or nonreal augmented operator product")
        return value.astype(float, copy=False)

    checked(reg_current)

    def expand(free):
        image = np.zeros(current.shape)
        image[active] = np.asarray(free).reshape(-1)
        return image

    def matvec(free):
        image = expand(free)
        prediction = np.asarray(
            control.call("jacobian", jacobian.forward, image)
        ).reshape(precision.shape)
        # Index BEFORE arithmetic: excluded channels may legitimately be NaN.
        data = sqrt_w * prediction[selected]
        parts = [data.real, data.imag] if complex_data else [data]
        if damping:
            parts.append(
                sqrt_damping * np.asarray(regularizer(image, regularization)).ravel()
            )
        return checked(np.concatenate(parts))

    def rmatvec(vector):
        vector = checked(vector).ravel()
        data = vector[:ndata]
        if complex_data:
            data = data + 1j * vector[ndata:data_rows]
        full = np.zeros(precision.shape, dtype=complex if complex_data else float)
        full[selected] = sqrt_w * data
        image = checked(control.call("adjoint", jacobian.adjoint, full)).reshape(
            current.shape
        )
        if damping:
            reg_data = vector[data_rows:].reshape(reg_shape)
            reg_adj = (
                regularization.adjoint(reg_data)
                if hasattr(regularization, "adjoint")
                else regularizer(reg_data, regularization)
            )
            image = image + sqrt_damping * checked(reg_adj)
        return checked(image[active])

    target_data = residual[selected] / sqrt_w
    parts = [target_data.real, target_data.imag] if complex_data else [target_data]
    if damping:
        parts.append(-sqrt_damping * reg_current.ravel())
    target = checked(np.concatenate(parts))
    operator = LinearOperator(
        (data_rows + nreg, nmodel), matvec=matvec, rmatvec=rmatvec, dtype=float
    )
    started = time.perf_counter()
    before = dict(control.work.counts)
    record = {
        "method": f"augmented_{method}",
        "iteration_limit": int(iterations),
        "rtol": float(rtol),
        "atol": float(atol),
        "btol": float(btol),
        "condition_limit": float(conlim),
        "preconditioner": preconditioner,
        "preconditioner_seed": int(preconditioner_seed),
        "preconditioner_probes": (
            int(preconditioner_probes) if preconditioner != "none" else 0
        ),
        "free_parameters": nmodel,
        "augmented_rows": operator.shape[0],
        "converged": False,
        "certificate_scope": "unconstrained_linearized_subproblem_not_nonlinear_or_box_optimality",
        "nullspace_selection": (
            "minimum_scaled_step_norm"
            if preconditioner != "none"
            else "minimum_euclidean_step_norm"
        ),
    }
    if not hasattr(control, "inner_solver_history"):
        control.inner_solver_history = []
    try:
        # Exact binary RHS rescaling protects M* h before it can underflow.
        rhs_exponent = int(np.frexp(np.max(np.abs(target), initial=0))[1])
        scaled_target = np.ldexp(target, -rhs_exponent)
        initial_gradient = operator.rmatvec(scaled_target) if nmodel else np.empty(0)
        initial_norm = norm(initial_gradient)
        if initial_norm == 0:
            record.update(
                iterations=0,
                stop_reason="stationary_subproblem",
                true_relative_residual=0.0,
                converged=True,
                augmented_residual_norm=float(norm(target)),
                physical_normal_residual_norm=0.0,
            )
            return np.zeros_like(current)
        # Scale M from an operator product, not ||h||/||M*h||: a large
        # left-nullspace component of h must not amplify the entire system.
        probe_norm = norm(operator.matvec(initial_gradient / initial_norm))
        if not np.isfinite(probe_norm) or probe_norm == 0:
            raise FloatingPointError(
                "inconsistent adjoint or unresolved operator scale"
            )
        operator_exponent = int(np.frexp(probe_norm)[1])
        normalized = LinearOperator(
            operator.shape,
            matvec=lambda z: checked(np.ldexp(operator.matvec(z), -operator_exponent)),
            rmatvec=lambda v: checked(
                np.ldexp(operator.rmatvec(v), -operator_exponent)
            ),
            dtype=float,
        )
        initial_gradient = np.ldexp(initial_gradient, -operator_exponent)
        initial_norm = norm(initial_gradient)
        record.update(
            rhs_binary_exponent=rhs_exponent,
            operator_binary_exponent=operator_exponent,
            step_binary_exponent=rhs_exponent - operator_exponent,
        )
        scale = np.ones(nmodel)
        if preconditioner == "column_rms":
            rng = np.random.default_rng(preconditioner_seed)
            columns = np.zeros(nmodel)
            for _ in range(preconditioner_probes):
                probe = rng.choice([-1.0, 1.0], size=operator.shape[0])
                product = normalized.rmatvec(probe)
                with np.errstate(over="raise", invalid="raise"):
                    columns += product**2 / preconditioner_probes
            positive = columns[columns > 0]
            if positive.size:
                # A floor bounds scaling of poorly sampled/unseen columns.
                floor = max(float(np.median(positive)) * 1e-3, np.finfo(float).tiny)
                scale = 1 / np.sqrt(np.maximum(columns, floor))
                scale /= np.median(scale)
                scale = np.clip(scale, 1e-3, 1e3)
            record.update(
                right_scale_min=float(scale.min()), right_scale_max=float(scale.max())
            )
        scaled = LinearOperator(
            operator.shape,
            matvec=lambda z: normalized.matvec(scale * z),
            rmatvec=lambda v: scale * normalized.rmatvec(v),
            dtype=float,
        )
        options = dict(atol=atol, btol=btol, conlim=conlim)
        if method == "lsmr":
            result = lsmr(scaled, scaled_target, maxiter=iterations, **options)
            z, code, completed, estimated_r, estimated_normal, anorm, acond, _ = result
        else:
            result = lsqr(scaled, scaled_target, iter_lim=iterations, **options)
            z, code, completed, estimated_r, _, anorm, acond, estimated_normal = result[
                :8
            ]
        control.work.counts["inner_iterations"] = control.work.counts.get(
            "inner_iterations", 0
        ) + int(completed)
        normalized_free = checked(scale * z)
        with np.errstate(over="raise", invalid="raise"):
            free = checked(np.ldexp(normalized_free, rhs_exponent - operator_exponent))
        # Evaluate BOTH M*(Md-h) and M*Md-M*h. Strong cancellation can make
        # either expression spuriously small; never certify from that one alone.
        product = normalized.matvec(normalized_free)
        true_residual = product - scaled_target
        normal_direct = normalized.rmatvec(true_residual)
        normal_split = normalized.rmatvec(product) - initial_gradient
        normal_norm = max(norm(normal_direct), norm(normal_split))
        relative = float(normal_norm / initial_norm)
        normal_target_met = bool(relative <= rtol)
        scipy_converged = bool(code in {0, 1, 2, 4, 5})
        converged = normal_target_met and scipy_converged
        reasons = {
            0: "zero_solution",
            1: "compatible_backward_error",
            2: "least_squares_backward_error",
            3: "condition_limit",
            4: "machine_compatible_error",
            5: "machine_least_squares_error",
            6: "machine_condition_limit",
            7: "iteration_limit",
        }
        disagreement = float(norm(normal_direct - normal_split) / initial_norm)
        reason = "normal_residual_target" if converged else reasons[int(code)]
        if scipy_converged and not normal_target_met:
            reason = (
                "residual_cancellation"
                if disagreement > rtol
                else "normal_residual_not_met"
            )
        record.update(
            iterations=int(completed),
            scipy_stop_code=int(code),
            scipy_stop_reason=reasons[int(code)],
            stop_reason=reason,
            scipy_converged=scipy_converged,
            normal_residual_target_met=normal_target_met,
            converged=converged,
            true_relative_residual=float(relative),
            physical_normal_residual_norm=float(
                np.ldexp(normal_norm, rhs_exponent + operator_exponent)
            ),
            initial_normal_residual_norm=float(
                np.ldexp(initial_norm, rhs_exponent + operator_exponent)
            ),
            augmented_residual_norm=float(np.ldexp(norm(true_residual), rhs_exponent)),
            augmented_initial_residual_norm=float(norm(target)),
            estimated_augmented_residual_norm=float(
                np.ldexp(estimated_r, rhs_exponent)
            ),
            estimated_scaled_normal_residual_norm=float(
                np.ldexp(estimated_normal, rhs_exponent + operator_exponent)
            ),
            estimated_scaled_operator_norm=float(anorm),
            estimated_scaled_condition_number=float(acond),
            condition_estimate_is_rank_certificate=False,
            direct_normal_relative_residual=float(norm(normal_direct) / initial_norm),
            split_normal_relative_residual=float(norm(normal_split) / initial_norm),
            normal_evaluation_disagreement=disagreement,
            quadratic_objective_reduction=float(
                np.ldexp(
                    np.dot(initial_gradient, normalized_free)
                    - 0.5 * norm(product) ** 2,
                    2 * rhs_exponent,
                )
            ),
        )
        return expand(free)
    except BudgetExhausted as exc:
        record.update(stop_reason=exc.reason, certificate_unavailable=True)
        raise
    except FloatingPointError:
        record.update(stop_reason="numerical_failure", certificate_unavailable=True)
        raise
    finally:
        record["elapsed_s"] = time.perf_counter() - started
        record["work"] = {
            key: value - before.get(key, 0)
            for key, value in control.work.counts.items()
            if value != before.get(key, 0)
        }
        control.inner_solver_history.append(record)
