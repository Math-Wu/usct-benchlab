"""Relinearized pressure inversion with acceptance on the nonlinear model."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from usctbench.core.stopping import BudgetExhausted
from usctbench.solvers.least_squares import normal_step, normal_regularizer, regularizer


def nonlinear_least_squares(
    forward,
    control,
    *,
    initial,
    bounds,
    inner_iterations=12,
    damping=0.0,
    regularization="laplacian",
    roi=None,
    step_length=1.0,
    smooth_sigma=0.0,
    max_update_mps=12.0,
    max_backtracks=10,
    gradient_rtol=1e-8,
):
    """Inexact Gauss-Newton outer steps; no validation values enter updates.

    A Born step can fail to descend for a WKB model (caustics, low frequency or
    discretization error). Such a run stops with line_search_failed, not a claim
    of convergence. The returned state/prediction is an atomic complete iterate.
    """
    for name, value in (
        ("inner_iterations", inner_iterations),
        ("max_backtracks", max_backtracks),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("step_length", step_length),
        ("max_update_mps", max_update_mps),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not np.isfinite(smooth_sigma)
        or smooth_sigma < 0
        or not np.isfinite(damping)
        or damping < 0
    ):
        raise ValueError("smoothing and damping must be finite and nonnegative")
    if not np.isfinite(gradient_rtol) or gradient_rtol < 0:
        raise ValueError("gradient_rtol must be finite and nonnegative")
    state = np.asarray(initial, dtype=float).copy()
    reference = state.copy()
    attempts = []
    direction_checks = []
    gradient_checks = []
    initial_gradient_norm = None
    derivative_kind = None

    def objective(model, prediction):
        residual = np.where(control.split.train, control.safe_observed - prediction, 0)
        reg = regularizer(model - reference, regularization)
        return float(
            0.5 * np.sum(control.precision * np.abs(residual) ** 2)
            + 0.5 * damping * np.vdot(reg, reg).real
        )

    try:
        linearization = control.call("forward", forward.linearize, state)
        derivative_kind = linearization.derivative_kind
        cost = objective(state, linearization.value)
        control.observe(
            0,
            state,
            linearization.value,
            objective=cost,
            sound_speed=1 / np.sqrt(state),
        )
        for iteration in range(1, control.policy.max_iterations + 1):
            if control.monitor.reason is not None:
                break
            residual = control.weighted_residual(linearization.value)
            gradient = -control.call(
                "adjoint", linearization.jacobian.adjoint, residual
            )
            gradient += damping * normal_regularizer(state - reference, regularization)
            if roi is not None:
                gradient = np.where(roi, gradient, 0)
            if not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite nonlinear objective gradient")
            with np.errstate(over="raise", invalid="raise"):
                gradient_norm = float(np.linalg.norm(gradient))
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("nonfinite objective gradient norm")
            if initial_gradient_norm is None:
                initial_gradient_norm = gradient_norm
            gradient_relative = gradient_norm / max(
                initial_gradient_norm, np.finfo(float).tiny
            )
            gradient_checks.append(
                {
                    "iteration": control.monitor.history[-1]["iteration"],
                    "norm": gradient_norm,
                    "relative_to_initial": gradient_relative,
                }
            )
            if gradient_relative <= gradient_rtol:
                control.monitor.finish(
                    "stationary_surrogate_gradient"
                    if "approximation" in derivative_kind
                    else "stationary_gradient"
                )
                break
            feasible_gradient = np.where(
                ((state <= 1 / bounds[1] ** 2) & (gradient > 0))
                | ((state >= 1 / bounds[0] ** 2) & (gradient < 0)),
                0,
                gradient,
            )
            feasible_norm = float(np.linalg.norm(feasible_gradient))
            if feasible_norm == 0:
                control.monitor.finish(
                    "stationary_surrogate_projected_gradient"
                    if "approximation" in derivative_kind
                    else "stationary_projected_gradient"
                )
                break
            direction = normal_step(
                linearization.jacobian,
                residual,
                state - reference,
                control,
                iterations=inner_iterations,
                damping=damping,
                regularization=regularization,
                roi=roi,
            )
            raw_direction = direction.copy()
            raw_slope = float(np.vdot(gradient, direction).real)
            smoothed_slope = None
            if smooth_sigma:
                smoothed = gaussian_filter(direction, smooth_sigma, mode="nearest")
                if roi is not None:
                    smoothed = np.where(roi, smoothed, 0)
                smoothed_slope = float(np.vdot(gradient, smoothed).real)
                # Smoothing a Newton step can turn it uphill even for a convex
                # quadratic. It is only a proposal, not an accepted optimizer.
                if np.isfinite(smoothed_slope) and smoothed_slope < 0:
                    direction = smoothed
            if roi is not None:
                direction = np.where(roi, direction, 0)
            if not np.all(np.isfinite(direction)):
                raise FloatingPointError("nonfinite Born update")
            speed = 1 / np.sqrt(state)
            accepted = False
            proposals = [("newton_proposal", direction)]
            if not np.array_equal(direction, raw_direction):
                proposals.append(("unsmoothed_newton", raw_direction))
            # Projection/clipping can also destroy descent of a Newton step.
            # A projected negative gradient provides an independently checked
            # fallback, at the same model-step scale, without changing the loss.
            with np.errstate(over="raise", invalid="raise"):
                direction_scale = float(np.linalg.norm(raw_direction))
            if not np.isfinite(direction_scale):
                raise FloatingPointError("nonfinite Newton direction norm")
            if direction_scale == 0:
                direction_scale = 1e-3 * np.linalg.norm(state)
            gradient_step = -feasible_gradient * (direction_scale / feasible_norm)
            proposals.append(("projected_gradient", gradient_step))
            direction_checks.append(
                {
                    "iteration": iteration,
                    "raw_directional_derivative": raw_slope,
                    "smoothed_directional_derivative": smoothed_slope,
                    "smoothing_rejected": smoothed_slope is not None
                    and smoothed_slope >= 0,
                }
            )
            for proposal_name, proposal in proposals:
                for backtrack in range(max_backtracks):
                    alpha = step_length * 0.5**backtrack
                    candidate = np.clip(
                        state + alpha * proposal, 1 / bounds[1] ** 2, 1 / bounds[0] ** 2
                    )
                    candidate_speed = np.clip(
                        1 / np.sqrt(candidate),
                        speed - max_update_mps,
                        speed + max_update_mps,
                    )
                    candidate = 1 / candidate_speed**2
                    if roi is not None:
                        candidate = np.where(roi, candidate, reference)
                    slope = float(np.vdot(gradient, candidate - state).real)
                    row = {
                        "iteration": iteration,
                        "proposal": proposal_name,
                        "step_length": alpha,
                        "projected_directional_derivative": slope,
                        "objective": None,
                        "accepted": False,
                    }
                    if not np.isfinite(slope) or slope >= 0:
                        row["rejection"] = "non_descent_projected_step"
                        attempts.append(row)
                        continue
                    try:
                        trial = control.call(
                            "line_search", forward.linearize, candidate
                        )
                        trial_cost = objective(candidate, trial.value)
                    except FloatingPointError:
                        row["rejection"] = "nonfinite_trial_prediction"
                        attempts.append(row)
                        continue
                    accepted = bool(
                        np.isfinite(trial_cost) and trial_cost <= cost + 1e-4 * slope
                    )
                    row.update(
                        objective=trial_cost if np.isfinite(trial_cost) else None,
                        accepted=accepted,
                    )
                    attempts.append(row)
                    if accepted:
                        break
                if accepted:
                    break
            if not accepted:
                control.monitor.finish("line_search_failed")
                break
            update = float(np.linalg.norm(candidate - state) / np.linalg.norm(state))
            state, linearization, cost = candidate, trial, trial_cost
            control.work.counts["background_builds"] = forward.background_builds
            control.work.counts["eikonal_source_solves"] = forward.eikonal_solves
            control.observe(
                iteration,
                state,
                trial.value,
                objective=cost,
                update_relative=update,
                sound_speed=1 / np.sqrt(state),
            )
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError:
        control.monitor.finish("numerical_failure")
    control.work.counts["background_builds"] = forward.background_builds
    control.work.counts["eikonal_source_solves"] = forward.eikonal_solves
    control.work.counts["green_source_solves"] = getattr(forward, "green_solves", 0)
    control.work.counts["green_matvecs"] = getattr(forward, "green_matvecs", 0)
    selected, metrics = control.output(reference)
    metrics["line_search_history"] = attempts
    metrics["direction_checks"] = direction_checks
    metrics["gradient_checks"] = gradient_checks
    metrics["gradient_rtol"] = gradient_rtol
    metrics["gradient_scope"] = (
        "provided_jacobian_regularized_training_gradient_at_listed_iterates_not_necessarily_selected_checkpoint"
    )
    metrics["line_search_acceptance"] = "projected_step_armijo_1e-4"
    metrics["derivative_kind"] = derivative_kind
    metrics["nonlinear_background_updates"] = max(0, len(control.monitor.history) - 1)
    return selected, metrics
