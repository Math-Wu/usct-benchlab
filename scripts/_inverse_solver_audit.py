"""GT-free objective/step diagnostics at a saved complete nonlinear iterate."""

import numpy as np
import time
from scipy.ndimage import gaussian_filter

from usctbench.solvers.least_squares import normal_step, normal_regularizer, regularizer


def channel_derivative_audit(value, plus, minus, tangent, observed, precision, h):
    """Attribute central loss-derivative error; never drop invalid training rays.

    A large per-ray error is a diagnostic, not proof of a peak switch. Entries
    outside the frozen training mask do not contribute, even if they are NaN.
    """
    arrays = [np.asarray(a) for a in (value, plus, minus, tangent, observed, precision)]
    value, plus, minus, tangent, observed, precision = arrays
    if any(a.shape != value.shape for a in arrays) or not np.isfinite(h) or h <= 0:
        raise ValueError("require equal shapes and positive finite difference step")
    if not np.isfinite(precision).all() or np.any(precision < 0):
        raise ValueError("require finite nonnegative precision")
    train = precision > 0
    finite = np.logical_and.reduce([np.isfinite(a) for a in arrays[:-1]])
    active = train & finite
    w = precision[active]
    r0, rp, rm = (
        value[active] - observed[active],
        plus[active] - observed[active],
        minus[active] - observed[active],
    )
    analytic = w * (r0.conj() * tangent[active]).real
    # Difference of squares in factored form avoids cancellation of large costs.
    numeric = w * ((rp - rm).conj() * (rp + rm)).real / (4 * h)
    error = numeric - analytic
    indices = np.argwhere(active)
    order = np.argsort(-np.abs(error), kind="stable")[:12]
    records = []
    for k in order:
        index = tuple(indices[k])
        records.append(
            {
                "index": list(map(int, index)),
                "analytic_data_derivative": float(analytic[k]),
                "central_data_derivative": float(numeric[k]),
                "derivative_error": float(error[k]),
                "base_delay_s": float(value[index]),
                "plus_delay_s": float(plus[index]),
                "minus_delay_s": float(minus[index]),
                "linear_delay_change_s": float(h * tangent[index]),
                "plus_linearization_error_s": float(
                    plus[index] - value[index] - h * tangent[index]
                ),
                "minus_linearization_error_s": float(
                    minus[index] - value[index] + h * tangent[index]
                ),
            }
        )
    return {
        "training_count": int(train.sum()),
        "invalid_training_count": int((train & ~finite).sum()),
        "complete_training_derivative": bool(np.all(finite[train])),
        "finite_training_data_derivative_error": float(error.sum()),
        "sum_absolute_channel_errors": float(np.abs(error).sum()),
        "top_channels": records,
        "interpretation": "large_error_is_not_by_itself_proof_of_peak_switch",
    }


def audit_iterate(
    forward,
    control,
    *,
    state,
    reference,
    bounds,
    damping,
    regularization,
    smooth_sigma=0,
    inner_iterations=12,
    compare_inner_solvers=False,
    inner_options=None,
    audit_methods=None,
    skip_gradient_fd=False,
    audit_caps=None,
):
    lin = control.call("forward", forward.linearize, state)
    residual = control.weighted_residual(lin.value)
    gradient = -control.call("adjoint", lin.jacobian.adjoint, residual)
    gradient += damping * normal_regularizer(state - reference, regularization)

    def objective(model, prediction):
        r = prediction[control.split.train] - control.observed[control.split.train]
        reg = regularizer(model - reference, regularization)
        return float(
            0.5 * np.sum(control.precision[control.split.train] * np.abs(r) ** 2)
            + 0.5 * damping * np.vdot(reg, reg).real
        )

    value = objective(state, lin.value)
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise FloatingPointError("invalid objective or gradient at audit checkpoint")
    steps = []
    plans = [
        ("normal_cg", "none", n)
        for n in sorted(set([inner_iterations, max(64, inner_iterations)]))
    ]
    if compare_inner_solvers:
        plans += [
            (method, preconditioner, inner_iterations)
            for method, preconditioner in [
                ("lsmr", "none"),
                ("lsqr", "none"),
                ("lsmr", "column_rms"),
            ]
        ]
    if audit_methods:
        plans = [
            (
                "lsmr" if name == "lsmr_column_rms" else name,
                "column_rms" if name == "lsmr_column_rms" else "none",
                inner_iterations,
            )
            for name in audit_methods
        ]
    if audit_caps:
        plans = [
            (method, preconditioner, cap)
            for method, preconditioner, _ in plans
            for cap in audit_caps
        ]
    for method, preconditioner, iterations in plans:
        started = time.perf_counter()
        counts = dict(control.work.counts)
        direction = normal_step(
            lin.jacobian,
            residual,
            state - reference,
            control,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
            method=method,
            preconditioner=preconditioner,
            **(inner_options or {}),
        )
        product = control.call("jacobian", lin.jacobian.forward, direction).reshape(
            control.observed.shape
        )
        model_residual = np.where(
            control.precision > 0, lin.value + product - control.observed, 0
        )
        reg = regularizer(state - reference + direction, regularization)
        quadratic = (
            0.5 * np.sum(control.precision * np.abs(model_residual) ** 2)
            + 0.5 * damping * np.vdot(reg, reg).real
        )
        smoothed = (
            gaussian_filter(direction, smooth_sigma, mode="nearest")
            if smooth_sigma
            else direction
        )
        candidate = np.clip(state + smoothed, 1 / bounds[1] ** 2, 1 / bounds[0] ** 2)
        speed = 1 / np.sqrt(state)
        candidate = 1 / np.clip(1 / np.sqrt(candidate), speed - 12, speed + 12) ** 2
        steps.append(
            {
                "inner": control.inner_solver_history[-1],
                "elapsed_s": time.perf_counter() - started,
                "work": {
                    k: v - counts.get(k, 0)
                    for k, v in control.work.counts.items()
                    if v != counts.get(k, 0)
                },
                "quadratic_objective": float(quadratic),
                "quadratic_reduction": float(value - quadratic),
                "relative_step_norm": float(
                    np.linalg.norm(direction) / np.linalg.norm(state)
                ),
                "newton_directional_derivative": float(
                    np.vdot(gradient, direction).real
                ),
                "smoothed_directional_derivative": float(
                    np.vdot(gradient, smoothed).real
                ),
                "old_projected_directional_derivative": float(
                    np.vdot(gradient, candidate - state).real
                ),
            }
        )

        print("completed inner audit", method, preconditioner, steps[-1], flush=True)

    # Difference the COMPLETE nonlinear penalized loss, not just J versus J*.
    # The direction scale is fixed from the current state, never from GT.
    direction = -gradient.copy()
    scale = float(np.max(np.abs(direction) / state))
    differences = []
    if scale > 0 and not skip_gradient_fd:
        direction *= 0.002 / scale
        analytic = float(np.vdot(gradient, direction).real)
        tangent = control.call("jacobian", lin.jacobian.forward, direction)
        for h in (1.0, 0.5, 0.25):
            plus = state + h * direction
            minus = state - h * direction
            p = control.call("gradient_fd", forward.linearize, plus).value
            m = control.call("gradient_fd", forward.linearize, minus).value
            finite_difference = (objective(plus, p) - objective(minus, m)) / (2 * h)
            differences.append(
                {
                    "step": h,
                    "analytic": analytic,
                    "finite_difference": finite_difference,
                    "relative_error": float(
                        abs(finite_difference - analytic)
                        / max(abs(analytic), np.finfo(float).tiny)
                    ),
                    "channels": channel_derivative_audit(
                        lin.value, p, m, tangent, control.observed, control.precision, h
                    ),
                }
            )
    return {
        "objective": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "normal_subproblems": steps,
        "objective_gradient_differences": differences,
        "gradient_fd_performed": not skip_gradient_fd,
        "ground_truth_used": False,
        "reconstruction_performed": False,
        "scope": "saved_checkpoint_not_global_optimality_or_image_quality_certificate",
    }
