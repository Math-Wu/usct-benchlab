"""Experimental SciPy TRF control of the same real band-delay objective.

Requires SciPy >= 1.16 (iteration callbacks); not a new registered algorithm.
There is no per-pixel update clipping. Reflective bounds and the trust region
control steps; the data, fine-grid penalty and train/validation split are fixed.
"""

import inspect

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator

from usctbench.core.stopping import BudgetExhausted
from usctbench.solvers.least_squares import regularizer


class SmoothTVLoss:
    """SciPy loss: quadratic data rows and soft-L1 spatial-gradient rows.

    With edge residual sqrt(lambda) * Dm, transition sqrt(lambda) * epsilon,
    the penalty is lambda * epsilon^2 * sum(sqrt(1 + (Dm/epsilon)^2) - 1).
    This is smooth anisotropic TV, not clipping or reweighting observed data.
    """

    def __init__(self, n_data, transition):
        if not np.isfinite(transition) or transition <= 0:
            raise ValueError("TV transition must be finite and positive")
        self.n_data, self.scale_squared = n_data, transition**2
        if not np.isfinite(self.scale_squared) or self.scale_squared == 0:
            raise ValueError("TV transition squared is not representable")

    def __call__(self, squared_residual):
        z = np.asarray(squared_residual)
        rho = np.vstack([z.copy(), np.ones_like(z), np.zeros_like(z)])
        edges = z[self.n_data :]
        root = np.sqrt(1 + edges / self.scale_squared)
        rho[0, self.n_data :] = 2 * edges / (root + 1)
        rho[1, self.n_data :] = 1 / root
        rho[2, self.n_data :] = -0.5 / self.scale_squared / root**3
        return rho


def solve_trust_region(
    forward,
    control,
    *,
    initial,
    bounds,
    prior_reference=None,
    damping,
    regularization,
    inner_iterations,
    tv_transition=None,
    gradient_rtol=None,
):
    if "callback" not in inspect.signature(least_squares).parameters:
        raise RuntimeError("experimental TRF requires SciPy >= 1.16 and Python >= 3.11")
    if (
        isinstance(inner_iterations, bool)
        or not isinstance(inner_iterations, int)
        or inner_iterations <= 0
    ):
        raise ValueError("inner_iterations must be a positive integer")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if gradient_rtol is not None and (
        not np.isfinite(gradient_rtol)
        or not 0 <= gradient_rtol < 1
        or 0 < gradient_rtol <= np.finfo(float).eps
    ):
        raise ValueError(
            "gradient_rtol must be zero or between machine epsilon and one; None retains legacy stopping"
        )
    if (
        len(bounds) != 2
        or not np.isfinite(bounds).all()
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError("bounds must be finite ordered positive speeds")
    initial_state = np.array(initial, float, copy=True)
    if (
        not np.isfinite(initial_state).all()
        or np.any(initial_state < 1 / bounds[1] ** 2)
        or np.any(initial_state > 1 / bounds[0] ** 2)
    ):
        raise ValueError(
            "initial squared slowness must be finite and within speed bounds"
        )
    reference = (
        initial_state.copy() if prior_reference is None else np.asarray(prior_reference)
    )
    if (
        reference.shape != initial_state.shape
        or np.iscomplexobj(reference)
        or not np.isfinite(reference).all()
        or np.any(reference <= 0)
    ):
        raise ValueError(
            "prior_reference must be a finite positive real model of the same shape"
        )
    reference = np.array(reference, dtype=float, copy=True)
    shape = initial_state.shape
    model_scale, residual_scale = 1 / 1500**2, 1e6
    train = control.split.train
    n_data = int(train.sum())
    # Remove a shared weight scale before square roots and Krylov products.
    # Reapplying it only for reporting avoids scale-dependent roundoff in IRLS.
    precision_reference = (
        float(np.max(control.precision[train]))
        if gradient_rtol is not None and n_data
        else 1.0
    )
    root_weight = np.sqrt(control.precision[train] / precision_reference)
    reg_shape = regularizer(reference * 0, regularization).shape
    n_residual = n_data + int(np.prod(reg_shape))
    reg_weight = float(np.sqrt(damping / precision_reference))
    loss = (
        "linear"
        if tv_transition is None
        else SmoothTVLoss(n_data, residual_scale * reg_weight * tv_transition)
    )

    def objective(residual):
        if isinstance(loss, str):
            return float(
                precision_reference
                * np.vdot(residual, residual).real
                / (2 * residual_scale**2)
            )
        return float(
            precision_reference * np.sum(loss(residual**2)[0]) / (2 * residual_scale**2)
        )

    cached_x = cached_lin = cached_residual = None
    last_accepted = initial_state.copy()
    result = None
    trial_failures = []
    initial_optimality = None
    gradient_reference = None
    optimality_reference = None
    previous_objective = None
    gtol = 1e-6 if gradient_rtol is None else (gradient_rtol or None)

    def evaluate(x):
        nonlocal cached_x, cached_lin, cached_residual
        if cached_x is None or not np.array_equal(x, cached_x):
            cached_x, cached_lin, cached_residual = None, None, None
            state = model_scale * (x.reshape(shape) + 1)
            lin = control.call("forward", forward.linearize, state)
            data_residual = (lin.value[train] - control.observed[train]) * root_weight
            penalty = reg_weight * regularizer(state - reference, regularization)
            residual = residual_scale * np.concatenate([data_residual, penalty.ravel()])
            if not np.isfinite(residual).all():
                raise FloatingPointError("invalid training prediction in TRF trial")
            cached_x, cached_lin, cached_residual = x.copy(), lin, residual
        return cached_lin, cached_residual

    def fun(x):
        try:
            return evaluate(x)[1]
        except FloatingPointError as exc:
            trial_failures.append(str(exc))
            # TRF rejects nonfinite trial residuals by reducing its trust radius.
            # An invalid initial point is handled separately before calling SciPy.
            return np.full(n_residual, np.inf)

    def jac(x):
        lin = evaluate(x)[0]

        def mv(v):
            dm = model_scale * np.asarray(v).reshape(shape)
            values = control.call("jacobian", lin.jacobian.forward, dm)
            penalty = reg_weight * regularizer(dm, regularization)
            return residual_scale * np.concatenate(
                [values[train] * root_weight, penalty.ravel()]
            )

        def rmv(v):
            v = np.asarray(v).ravel() * residual_scale
            sensitivity = np.zeros_like(control.observed, dtype=float)
            sensitivity[train] = v[:n_data] * root_weight
            gradient = control.call("adjoint", lin.jacobian.adjoint, sensitivity)
            reg_values = v[n_data:].reshape(reg_shape)
            transpose = (
                regularization.adjoint(reg_values)
                if hasattr(regularization, "adjoint")
                else regularizer(reg_values, regularization)
            )
            return model_scale * (gradient + reg_weight * transpose).ravel()

        return LinearOperator(
            (n_residual, reference.size), matvec=mv, rmatvec=rmv, dtype=float
        )

    def diagnostics(x, residual, state, *, initial=False):
        nonlocal gradient_reference, optimality_reference, previous_objective
        force = residual if isinstance(loss, str) else loss(residual**2)[1] * residual
        # Full penalized gradient in dimensionless x coordinates, with the
        # optimizer's arbitrary residual scaling removed. This adjoint is billed.
        gradient = jac(x).rmatvec(force) * precision_reference / residual_scale**2
        lower = 1 / bounds[1] ** 2 / model_scale - 1
        upper = 1 / bounds[0] ** 2 / model_scale - 1
        distance = np.where(
            gradient < 0, upper - x, np.where(gradient > 0, x - lower, 1.0)
        )
        gradient_norm = float(np.linalg.norm(gradient))
        optimality = float(np.max(np.abs(gradient * distance)))
        if initial:
            gradient_reference, optimality_reference = gradient_norm, optimality
        tiny = np.finfo(float).tiny
        # The projected-gradient mapping is zero at a box-constrained KKT point,
        # including an active bound with a nonzero unconstrained gradient.
        projected = x - np.clip(
            x - gradient / max(gradient_reference, tiny), lower, upper
        )
        current_objective = objective(residual)
        decrease = (
            None
            if previous_objective is None
            else previous_objective - current_objective
        )
        record = {
            "relative_gradient_l2": gradient_norm / max(gradient_reference, tiny),
            "relative_bound_optimality": optimality / max(optimality_reference, tiny),
            "projected_gradient_inf": float(np.max(np.abs(projected))),
            "true_objective_decrease": decrease,
            "true_objective_relative_decrease": (
                None
                if decrease is None
                else decrease / max(abs(previous_objective), tiny)
            ),
            "max_speed_update_mps": (
                0.0
                if initial
                else float(
                    np.max(np.abs(1 / np.sqrt(state) - 1 / np.sqrt(last_accepted)))
                )
            ),
        }
        previous_objective = current_objective
        return record

    def callback(intermediate_result):
        nonlocal last_accepted
        state = model_scale * (intermediate_result.x.reshape(shape) + 1)
        if not np.array_equal(state, last_accepted):
            lin, residual = evaluate(intermediate_result.x)
            reason = control.observe(
                len(control.monitor.history),
                state,
                lin.value,
                objective=objective(residual),
                update_relative=float(
                    np.linalg.norm(state - last_accepted)
                    / np.linalg.norm(last_accepted)
                ),
                sound_speed=1 / np.sqrt(state),
                optimizer_diagnostics=diagnostics(
                    intermediate_result.x, residual, state
                ),
            )
            last_accepted = state.copy()
            if reason is not None:
                raise StopIteration

    try:
        x0 = (initial_state / model_scale - 1).ravel()
        lin, residual = evaluate(x0)
        control.observe(
            0,
            initial_state,
            lin.value,
            objective=objective(residual),
            sound_speed=1 / np.sqrt(initial_state),
            optimizer_diagnostics=diagnostics(
                x0, residual, initial_state, initial=True
            ),
        )
        if control.monitor.reason is None:
            if gradient_rtol is not None:
                force = (
                    residual
                    if isinstance(loss, str)
                    else loss(residual**2)[1] * residual
                )
                gradient = jac(x0).rmatvec(force)
                lower = 1 / bounds[1] ** 2 / model_scale - 1
                upper = 1 / bounds[0] ** 2 / model_scale - 1
                # Same Coleman-Li bound distance used by SciPy's TRF gtol.
                distance = np.where(
                    gradient < 0, upper - x0, np.where(gradient > 0, x0 - lower, 1.0)
                )
                initial_optimality = float(np.max(np.abs(gradient * distance)))
                if not np.isfinite(initial_optimality):
                    raise FloatingPointError("nonfinite initial TRF gradient")
                if initial_optimality > 0:
                    factor = 1 / np.sqrt(initial_optimality)
                    residual_scale *= factor
                    cached_residual *= factor
                    if (
                        not np.isfinite(residual_scale)
                        or not np.isfinite(cached_residual).all()
                    ):
                        raise FloatingPointError("TRF gradient normalization overflow")
                    if tv_transition is not None:
                        loss = SmoothTVLoss(
                            n_data, residual_scale * reg_weight * tv_transition
                        )
                else:
                    control.monitor.finish("stationary_gradient")
        if control.monitor.reason is None:
            result = least_squares(
                fun,
                x0,
                jac=jac,
                method="trf",
                tr_solver="lsmr",
                loss=loss,
                bounds=(
                    1 / bounds[1] ** 2 / model_scale - 1,
                    1 / bounds[0] ** 2 / model_scale - 1,
                ),
                tr_options={"maxiter": inner_iterations, "atol": 1e-4, "btol": 1e-4},
                x_scale=0.02,
                ftol=1e-8,
                xtol=1e-8,
                gtol=gtol,
                max_nfev=1 + 8 * control.policy.max_iterations,
                callback=callback,
            )
        if control.monitor.reason is None:
            reasons = {
                0: "optimizer_evaluation_budget",
                1: "stationary_gradient",
                2: "objective_plateau",
                3: "update_stagnation",
                4: "objective_and_update_stagnation",
            }
            control.monitor.finish(reasons.get(result.status, "optimizer_stopped"))
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError:
        control.monitor.finish("numerical_failure")
    control.work.counts.update(
        {
            "background_builds": forward.background_builds,
            "green_source_solves": getattr(forward, "green_solves", 0),
            "green_matvecs": getattr(forward, "green_matvecs", 0),
        }
    )
    selected, metrics = control.output(initial_state)
    selected_iteration = metrics["stopping"]["selected_iteration"]
    metrics["selected_optimizer_diagnostics"] = next(
        (
            row.get("optimizer_diagnostics")
            for row in control.monitor.history
            if row["iteration"] == selected_iteration
        ),
        None,
    )
    metrics["gradient_reference_l2"] = gradient_reference
    metrics["bound_optimality_reference"] = optimality_reference
    metrics["prior_reference_source"] = (
        "initial" if prior_reference is None else "explicit"
    )
    metrics["optimizer"] = "scipy_trf_lsmr"
    metrics["optimizer_settings"] = {
        "inner_iteration_limit": inner_iterations,
        "lsmr_atol": 1e-4,
        "lsmr_btol": 1e-4,
        "x_scale": 0.02,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "gtol": gtol,
        "gradient_rtol": gradient_rtol,
        "gradient_normalization": (
            "initial_bound_scaled_full_penalized_gradient"
            if gradient_rtol is not None
            else "legacy_absolute_microsecond_residual"
        ),
        "initial_optimality_before_normalization": (
            None
            if initial_optimality is None
            else initial_optimality * precision_reference
        ),
        "objective_precision_reference": precision_reference,
        "residual_scale": residual_scale,
        "pixel_speed_step_clip": False,
        "direction_smoothing": False,
        "internal_stopping_may_precede_monitor": True,
        "regularization_loss": (
            "quadratic" if tv_transition is None else "smooth_anisotropic_tv"
        ),
        "tv_transition_squared_slowness": tv_transition,
        "data_loss": "quadratic",
    }
    metrics["trial_numerical_failures"] = trial_failures
    metrics["optimizer_terminal"] = (
        None
        if result is None
        else {
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "optimality": float(result.optimality),
            "optimality_scope": "terminal_iterate_not_validation_selected_checkpoint",
            "residual_scale": residual_scale,
            "model_scale": model_scale,
        }
    )
    return selected, metrics
