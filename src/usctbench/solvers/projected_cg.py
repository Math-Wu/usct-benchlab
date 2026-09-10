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
    optimality_rtol=1e-10,
):
    """All CG steps share one global budget, including across IRLS restarts.

    IRLS freezes its training-only weights within each stage, but refreshes them
    early on inner convergence or loss of descent. The requested stage count
    sets the maximum stage length, not a cap on these corrective restarts.
    Huber objective and validation residuals use the original precision.
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

    def negative_gradient(weights):
        return control.call(
            "adjoint", operator.adjoint, weights * residual(prediction)
        ) - damping * normal(x)

    def free_variables(vector):
        # Probe one representable step: the callback owns the box/ROI bounds.
        # A clipped outward component must not enter either CG step-size term.
        probe = np.nextafter(x, np.where(vector >= 0, np.inf, -np.inf))
        free = (vector == 0) | (project(probe) != x)
        return free if roi is None else free & roi

    optimality = {}
    initial_gradient_norm = None
    if not np.isfinite(optimality_rtol) or optimality_rtol < 0:
        raise ValueError("gradient_rtol must be finite and nonnegative")

    def output():
        state, metrics = control.output(start)
        selected = metrics["stopping"]["selected_iteration"]
        # Never certify a restored validation checkpoint using another iterate's
        # gradient, or spend additional operator work after a stopping trigger.
        metrics["solver_optimality"] = {
            "objective": "huber" if huber_delta is not None else "least_squares",
            "includes_regularization": True,
            "criterion": "box_roi_active_set_KKT_gradient_inf_norm",
            "relative_tolerance": optimality_rtol,
            "iteration": selected,
            "verified": selected in optimality,
            "converged": False,
            **optimality.get(selected, {}),
        }
        return state, metrics

    try:
        prediction = control.call("forward", operator.forward, x).reshape(
            control.observed.shape
        )
        cost = objective(x, prediction)
        if control.observe(0, x, prediction, objective=cost, sound_speed=to_image(x)):
            return output()
        block = max(1, int(np.ceil(control.policy.max_iterations / irls_stages)))
        weights = precision(prediction)
        direction, gamma, previous_free = None, None, None
        stage_gamma = 0.0
        for iteration in range(1, control.policy.max_iterations + 1):
            current_weights = precision(prediction)
            true_gradient = negative_gradient(current_weights)
            if not np.all(np.isfinite(true_gradient)):
                raise FloatingPointError("nonfinite objective gradient")
            free = free_variables(true_gradient)
            gradient_norm = float(np.max(np.abs(np.where(free, true_gradient, 0))))
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient_norm
            tolerance = max(
                np.finfo(float).tiny, initial_gradient_norm * optimality_rtol
            )
            converged = gradient_norm <= tolerance
            optimality[iteration - 1] = {
                "projected_gradient_norm": gradient_norm,
                "initial_projected_gradient_norm": initial_gradient_norm,
                "projected_gradient_tolerance": tolerance,
                "converged": converged,
            }
            # Continuing CG after numerical stationarity can amplify roundoff,
            # including on unconstrained/ROI-only quadratics. The same relative
            # KKT check therefore applies to every objective, not only Huber.
            if converged:
                control.monitor.finish("stationary_gradient")
                break
            restart = (
                direction is None
                or (huber_delta is not None and (iteration - 1) % block == 0)
                or not np.array_equal(free, previous_free)
            )
            accepted = False
            # A stale IRLS/CG direction gets one fresh projected-gradient retry.
            # Retries consume operator budgets, not extra completed iterations.
            for _ in range(2):
                if restart:
                    weights = current_weights
                    control.work.counts["cg_restarts"] = (
                        control.work.counts.get("cg_restarts", 0) + 1
                    )
                    if huber_delta is not None:
                        control.work.counts["irls_reweights"] = (
                            control.work.counts.get("irls_reweights", 0) + 1
                        )
                stale_weights = not np.array_equal(weights, current_weights)
                raw_gradient = (
                    negative_gradient(weights) if stale_weights else true_gradient
                )
                gradient = scale * np.where(free, raw_gradient, 0)
                next_gamma = float(np.vdot(gradient, gradient).real)
                if not np.isfinite(next_gamma):
                    raise FloatingPointError("nonfinite scaled gradient")
                if stale_weights and next_gamma <= max(
                    np.finfo(float).tiny, stage_gamma * 1e-14
                ):
                    restart = True
                    continue
                if next_gamma <= np.finfo(float).tiny:
                    control.monitor.finish(
                        "stationary_gradient"
                        if converged
                        else "linear_solver_breakdown"
                    )
                    break
                search_direction = (
                    scale * gradient
                    if restart
                    else scale * gradient + (next_gamma / gamma) * direction
                )
                feasible = free_variables(search_direction)
                if (
                    np.any(search_direction[~feasible] != 0)
                    or np.vdot(true_gradient, search_direction).real <= 0
                ):
                    restart = True
                    continue
                q = control.call(
                    "jacobian", operator.forward, search_direction
                ).reshape(control.observed.shape)
                reg = regularize(search_direction)
                denominator = float(
                    np.sum(weights * np.abs(q) ** 2) + damping * np.vdot(reg, reg).real
                )
                if not np.isfinite(denominator) or denominator <= 0:
                    control.monitor.finish("linear_solver_breakdown")
                    break
                alpha = next_gamma / denominator
                for trial in range(12):
                    step = alpha * 0.5**trial
                    raw = x + step * search_direction
                    candidate = project(raw)
                    projected = not np.array_equal(candidate, raw)
                    next_prediction = (
                        control.call(
                            "line_search", operator.forward, candidate
                        ).reshape(control.observed.shape)
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
                    restart = True
                    continue
                relative_update = float(
                    np.linalg.norm(candidate - x)
                    / max(np.linalg.norm(reference + x), np.finfo(float).tiny)
                )
                small_update = (
                    control.policy.update_rtol is not None
                    and relative_update <= control.policy.update_rtol
                )
                small_improvement = (
                    control.policy.objective_rtol is not None
                    and (cost - next_cost) / max(abs(cost), np.finfo(float).tiny)
                    <= control.policy.objective_rtol
                )
                if stale_weights and (small_update or small_improvement):
                    accepted, restart = False, True
                    continue
                if restart:
                    stage_gamma = next_gamma
                direction, gamma, previous_free = search_direction, next_gamma, free
                break
            if not accepted:
                control.monitor.finish("line_search_failed")
                break
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
    return output()
