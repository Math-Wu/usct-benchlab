"""GT-free objective/step diagnostics at a saved complete nonlinear iterate."""

import numpy as np
from scipy.ndimage import gaussian_filter

from usctbench.solvers.least_squares import normal_step, normal_regularizer, regularizer


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
    for iterations in sorted(set([inner_iterations, max(64, inner_iterations)])):
        direction = normal_step(
            lin.jacobian,
            residual,
            state - reference,
            control,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
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

    # Difference the COMPLETE nonlinear penalized loss, not just J versus J*.
    # The direction scale is fixed from the current state, never from GT.
    direction = -gradient.copy()
    scale = float(np.max(np.abs(direction) / state))
    differences = []
    if scale > 0:
        direction *= 0.002 / scale
        analytic = float(np.vdot(gradient, direction).real)
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
                }
            )
    return {
        "objective": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "normal_subproblems": steps,
        "objective_gradient_differences": differences,
        "ground_truth_used": False,
        "reconstruction_performed": False,
        "scope": "saved_checkpoint_not_global_optimality_or_image_quality_certificate",
    }
