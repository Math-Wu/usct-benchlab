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
    damping,
    regularization,
    inner_iterations,
    tv_transition=None,
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
    if (
        len(bounds) != 2
        or not np.isfinite(bounds).all()
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError("bounds must be finite ordered positive speeds")
    reference = np.array(initial, float, copy=True)
    if (
        not np.isfinite(reference).all()
        or np.any(reference < 1 / bounds[1] ** 2)
        or np.any(reference > 1 / bounds[0] ** 2)
    ):
        raise ValueError(
            "initial squared slowness must be finite and within speed bounds"
        )
    shape = reference.shape
    model_scale, residual_scale = 1 / 1500**2, 1e6
    train = control.split.train
    root_weight = np.sqrt(control.precision[train])
    n_data = int(train.sum())
    reg_shape = regularizer(reference * 0, regularization).shape
    n_residual = n_data + int(np.prod(reg_shape))
    reg_weight = float(np.sqrt(damping))
    loss = (
        "linear"
        if tv_transition is None
        else SmoothTVLoss(n_data, residual_scale * reg_weight * tv_transition)
    )

    def objective(residual):
        if isinstance(loss, str):
            return float(np.vdot(residual, residual).real / (2 * residual_scale**2))
        return float(np.sum(loss(residual**2)[0]) / (2 * residual_scale**2))

    cached_x = cached_lin = cached_residual = None
    last_accepted = reference.copy()
    result = None
    trial_failures = []

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
            )
            last_accepted = state.copy()
            if reason is not None:
                raise StopIteration

    try:
        x0 = (reference / model_scale - 1).ravel()
        lin, residual = evaluate(x0)
        control.observe(
            0,
            reference,
            lin.value,
            objective=objective(residual),
            sound_speed=1 / np.sqrt(reference),
        )
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
                gtol=1e-6,
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
    selected, metrics = control.output(reference)
    metrics["optimizer"] = "scipy_trf_lsmr"
    metrics["optimizer_settings"] = {
        "inner_iteration_limit": inner_iterations,
        "lsmr_atol": 1e-4,
        "lsmr_btol": 1e-4,
        "x_scale": 0.02,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "gtol": 1e-6,
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
