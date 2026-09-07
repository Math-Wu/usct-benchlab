"""Projected, optionally preconditioned CGLS/IRLS for real linear inverse problems."""

from __future__ import annotations

import numpy as np

from usctbench.core.stopping import BudgetExhausted
from usctbench.solvers.least_squares import regularizer, normal_regularizer


def projected_cg(
    operator,
    control,
    *,
    initial,
    reference,
    project,
    to_image,
    damping=0.0,
    regularization="identity",
    roi=None,
    regularization_roi=None,
    preconditioner=None,
    huber_delta=None,
    irls_stages=1,
):
    """All CG steps share one global budget, including across IRLS restarts.

    IRLS freezes its training-only weights within each stage. Huber objective
    and validation residuals are always measured with the original precision.
    Backtracking handles physical bound projection; rejected trials never
    replace a valid checkpoint. Preconditioner is a physical-coordinate scale D.
    """
    x = np.array(initial, dtype=float, copy=True)
    start = x.copy()
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if huber_delta is not None and (not np.isfinite(huber_delta) or huber_delta <= 0):
        raise ValueError("huber_delta_s must be finite and positive")
    if (
        isinstance(irls_stages, bool)
        or int(irls_stages) != irls_stages
        or irls_stages <= 0
    ):
        raise ValueError("irls_iterations must be a positive integer")
    scale = (
        np.ones_like(x)
        if preconditioner is None
        else np.asarray(preconditioner, dtype=float)
    )
    if scale.shape != x.shape or not np.all(np.isfinite(scale)) or np.any(scale < 0):
        raise ValueError(
            "preconditioner must be finite, nonnegative and match the image"
        )
    scale = scale if roi is None else np.where(roi, scale, 0)
    regmask = (
        np.ones_like(x)
        if regularization_roi is None
        else np.asarray(regularization_roi, dtype=float)
    )

    def regularize(value):
        return regularizer(regmask * value, regularization)

    def normal(value):
        return regmask * normal_regularizer(regmask * value, regularization)

    def residual(prediction):
        return np.where(control.split.train, control.safe_observed - prediction, 0)

    def objective(value, prediction):
        a = np.abs(residual(prediction))
        loss = (
            0.5 * a**2
            if huber_delta is None
            else np.where(
                a <= huber_delta, 0.5 * a**2, huber_delta * (a - 0.5 * huber_delta)
            )
        )
        reg = regularize(value)
        return float(
            np.sum(control.precision * loss) + 0.5 * damping * np.vdot(reg, reg).real
        )

    def precision(prediction):
        if huber_delta is None:
            return control.precision
        return control.precision * np.minimum(
            1.0,
            huber_delta
            / np.maximum(np.abs(residual(prediction)), np.finfo(float).tiny),
        )

    try:
        prediction = control.call("forward", operator.forward, x).reshape(
            control.observed.shape
        )
        cost = objective(x, prediction)
        if control.observe(0, x, prediction, objective=cost, sound_speed=to_image(x)):
            return control.output(start)
        block = max(1, int(np.ceil(control.policy.max_iterations / irls_stages)))
        weights = precision(prediction)
        direction, gamma = None, None
        for iteration in range(1, control.policy.max_iterations + 1):
            restart = direction is None or (
                huber_delta is not None and (iteration - 1) % block == 0
            )
            if restart:
                weights = precision(prediction)
                control.work.counts["cg_restarts"] = (
                    control.work.counts.get("cg_restarts", 0) + 1
                )
            gradient = scale * (
                control.call(
                    "adjoint", operator.adjoint, weights * residual(prediction)
                )
                - damping * normal(x)
            )
            next_gamma = float(np.vdot(gradient, gradient).real)
            if next_gamma <= np.finfo(float).tiny:
                control.monitor.finish("stationary_gradient")
                break
            direction = (
                scale * gradient
                if restart
                else scale * gradient + (next_gamma / gamma) * direction
            )
            gamma = next_gamma
            q = control.call("jacobian", operator.forward, direction).reshape(
                control.observed.shape
            )
            reg = regularize(direction)
            denominator = float(
                np.sum(weights * np.abs(q) ** 2) + damping * np.vdot(reg, reg).real
            )
            if not np.isfinite(denominator) or denominator <= 0:
                control.monitor.finish("linear_solver_breakdown")
                break
            alpha = gamma / denominator
            accepted = False
            for trial in range(12):
                step = alpha * 0.5**trial
                raw = x + step * direction
                candidate = project(raw)
                projected = not np.array_equal(candidate, raw)
                next_prediction = (
                    control.call("line_search", operator.forward, candidate).reshape(
                        control.observed.shape
                    )
                    if projected
                    else prediction + step * q
                )
                next_cost = objective(candidate, next_prediction)
                if np.isfinite(next_cost) and next_cost <= cost + 1e-12 * max(
                    cost, np.finfo(float).tiny
                ):
                    accepted = True
                    break
            if not accepted:
                control.monitor.finish("line_search_failed")
                break
            relative_update = float(
                np.linalg.norm(candidate - x)
                / max(np.linalg.norm(reference + x), np.finfo(float).tiny)
            )
            x, prediction, cost = candidate, next_prediction, next_cost
            if control.observe(
                iteration,
                x,
                prediction,
                objective=cost,
                update_relative=relative_update,
                sound_speed=to_image(x),
            ):
                break
            if projected or trial:
                direction = None
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError:
        control.monitor.finish("numerical_failure")
    return control.output(start)
