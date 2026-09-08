"""Matrix-free real-parameter least squares with real or complex measurements."""

from __future__ import annotations

import numpy as np

from usctbench.core.stopping import BudgetExhausted


def regularizer(image, kind="identity"):
    if kind in {"identity", "l2"}:
        return np.asarray(image)
    if kind in {"laplacian", "roughness"}:
        a = np.pad(image, 1, mode="edge")
        return a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * image
    if isinstance(kind, tuple) and len(kind) == 2:
        # Dimensionless ell^2 * physical Laplacian; edge replication supplies
        # zero-normal-flux boundaries and preserves a symmetric discrete map.
        a = np.pad(image, 1, mode="edge")
        return kind[0] * (a[:-2, 1:-1] + a[2:, 1:-1] - 2 * image) + kind[1] * (
            a[1:-1, :-2] + a[1:-1, 2:] - 2 * image
        )
    raise ValueError("regularization must be identity or laplacian")


def normal_regularizer(image, kind="identity"):
    return (
        image
        if kind in {"identity", "l2"}
        else regularizer(regularizer(image, kind), kind)
    )


def normal_step(
    jacobian,
    residual,
    current,
    control,
    *,
    iterations,
    damping=0.0,
    regularization="identity",
    roi=None,
):
    """Truncated GN system (J* W J + damping L*L) step, exact discrete adjoints.

    `current` is the perturbation relative to the prior. Iterates of this inner
    linear solve are not complete nonlinear reconstructions or stopping units.
    """
    active = (
        np.ones(current.shape, dtype=bool)
        if roi is None
        else np.asarray(roi, dtype=bool)
    )
    rhs = control.call("adjoint", jacobian.adjoint, residual)
    rhs = np.where(
        active, rhs - damping * normal_regularizer(current, regularization), 0.0
    )
    solution = np.zeros_like(current)
    direction = rhs.copy()
    rr = float(np.vdot(rhs, rhs).real)
    initial_rr = rr
    for _ in range(iterations):
        if rr <= max(np.finfo(float).tiny, initial_rr * 1e-14):
            break
        projected = control.call("jacobian", jacobian.forward, direction).reshape(
            control.observed.shape
        )
        q = control.call("adjoint", jacobian.adjoint, control.precision * projected)
        q = np.where(
            active, q + damping * normal_regularizer(direction, regularization), 0.0
        )
        denom = float(np.vdot(direction, q).real)
        if not np.isfinite(denom) or denom <= 0:
            break
        alpha = rr / denom
        solution += alpha * direction
        rhs -= alpha * q
        next_rr = float(np.vdot(rhs, rhs).real)
        direction = rhs + (next_rr / rr) * direction
        rr = next_rr
        control.work.counts["inner_iterations"] = (
            control.work.counts.get("inner_iterations", 0) + 1
        )
    return solution


def linear_cgls(
    operator,
    control,
    *,
    initial,
    offset,
    to_speed,
    project,
    reference,
    damping=0.0,
    regularization="identity",
    roi=None,
):
    """CGLS with a real image / complex-data adjoint and budget-safe checkpoints.

    Projection onto physical bounds restarts conjugacy. Held-out data never
    enters the gradient, Hessian product, line search, or projection.
    """
    x = np.array(initial, dtype=float, copy=True)
    start = x.copy()
    try:
        pred = offset + control.call("forward", operator.forward, x).reshape(
            control.observed.shape
        )
        reg = regularizer(x, regularization)
        objective = (
            0.5
            * np.sum(
                control.precision
                * np.abs(np.where(control.split.train, control.safe_observed - pred, 0))
                ** 2
            )
            + 0.5 * damping * np.vdot(reg, reg).real
        )
        if control.observe(
            0, x, pred, objective=float(objective), sound_speed=to_speed(x)
        ):
            return control.output(start)
        gradient = control.call(
            "adjoint", operator.adjoint, control.weighted_residual(pred)
        ) - damping * normal_regularizer(x, regularization)
        if roi is not None:
            gradient = np.where(roi, gradient, 0)
        direction = gradient.copy()
        gamma = float(np.vdot(gradient, gradient).real)
        for iteration in range(1, control.policy.max_iterations + 1):
            if gamma <= np.finfo(float).tiny:
                control.monitor.finish("stationary_gradient")
                break
            q = control.call("jacobian", operator.forward, direction).reshape(
                control.observed.shape
            )
            reg_p = regularizer(direction, regularization)
            denom = float(
                np.sum(control.precision * np.abs(q) ** 2)
                + damping * np.vdot(reg_p, reg_p).real
            )
            if not np.isfinite(denom) or denom <= 0:
                control.monitor.finish("linear_solver_breakdown")
                break
            alpha = gamma / denom
            raw = x + alpha * direction
            candidate = project(raw)
            projected = not np.array_equal(raw, candidate)
            next_pred = (
                offset
                + control.call("forward", operator.forward, candidate).reshape(
                    control.observed.shape
                )
                if projected
                else pred + alpha * q
            )
            relative_update = float(
                np.linalg.norm(candidate - x)
                / max(np.linalg.norm(reference + x), np.finfo(float).tiny)
            )
            reg = regularizer(candidate, regularization)
            objective = (
                0.5
                * np.sum(
                    control.precision
                    * np.abs(
                        np.where(
                            control.split.train, control.safe_observed - next_pred, 0
                        )
                    )
                    ** 2
                )
                + 0.5 * damping * np.vdot(reg, reg).real
            )
            x, pred = candidate, next_pred
            if control.observe(
                iteration,
                x,
                pred,
                objective=float(objective),
                update_relative=relative_update,
                sound_speed=to_speed(x),
            ):
                break
            gradient = control.call(
                "adjoint", operator.adjoint, control.weighted_residual(pred)
            ) - damping * normal_regularizer(x, regularization)
            if roi is not None:
                gradient = np.where(roi, gradient, 0)
            next_gamma = float(np.vdot(gradient, gradient).real)
            direction = (
                gradient.copy()
                if projected
                else gradient + (next_gamma / gamma) * direction
            )
            gamma = next_gamma
        if control.monitor.reason is None:
            control.monitor.finish("max_iterations")
    except BudgetExhausted as exc:
        control.monitor.finish(exc.reason)
    except FloatingPointError:
        control.monitor.finish("numerical_failure")
    return control.output(start)
