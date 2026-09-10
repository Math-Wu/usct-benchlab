"""Matrix-free real-parameter least squares with real or complex measurements."""

from __future__ import annotations

import numpy as np

from usctbench.core.stopping import BudgetExhausted


def regularizer(image, kind="identity"):
    if hasattr(kind, "forward") and hasattr(kind, "adjoint"):
        return kind.forward(image)
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
    if hasattr(kind, "forward") and hasattr(kind, "adjoint"):
        return kind.adjoint(kind.forward(image))
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
    method="normal_cg",
    rtol=1e-7,
    atol=1e-10,
    btol=1e-10,
    conlim=1e8,
    preconditioner="none",
    preconditioner_probes=8,
    preconditioner_seed=0,
):
    """Truncated GN system (J* W J + damping L*L) step, exact discrete adjoints.

    `current` is the perturbation relative to the prior. Iterates of this inner
    linear solve are not complete nonlinear reconstructions or stopping units.
    """
    if method in {"lsmr", "lsqr"}:
        from usctbench.solvers.augmented import augmented_step

        return augmented_step(
            jacobian,
            residual,
            current,
            control,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
            roi=roi,
            method=method,
            rtol=rtol,
            atol=atol,
            btol=btol,
            conlim=conlim,
            preconditioner=preconditioner,
            preconditioner_probes=preconditioner_probes,
            preconditioner_seed=preconditioner_seed,
        )
    if method != "normal_cg":
        raise ValueError("inner solver must be normal_cg, lsmr or lsqr")
    if preconditioner != "none":
        raise ValueError("normal_cg does not support column_rms preconditioning")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    active = (
        np.ones(current.shape, dtype=bool)
        if roi is None
        else np.asarray(roi, dtype=bool)
    )
    rhs = control.call("adjoint", jacobian.adjoint, residual)
    rhs = np.where(
        active, rhs - damping * normal_regularizer(current, regularization), 0.0
    )
    if not np.isfinite(rhs).all():
        raise FloatingPointError("nonfinite normal-system right-hand side")
    solution = np.zeros_like(current)
    direction = rhs.copy()
    rr = float(np.vdot(rhs, rhs).real)
    initial_rr = rr

    def normal_product(image):
        projected = control.call("jacobian", jacobian.forward, image).reshape(
            control.observed.shape
        )
        weighted = np.where(control.precision > 0, projected, 0) * control.precision
        if not np.isfinite(weighted).all():
            raise FloatingPointError("nonfinite training Jacobian product")
        q = control.call("adjoint", jacobian.adjoint, weighted)
        return np.where(
            active, q + damping * normal_regularizer(image, regularization), 0.0
        )

    initial_rhs = rhs.copy()
    completed = 0
    for _ in range(iterations):
        if rr <= max(np.finfo(float).tiny, initial_rr * rtol**2):
            break
        q = normal_product(direction)
        denom = float(np.vdot(direction, q).real)
        if not np.isfinite(denom) or denom <= 0:
            raise FloatingPointError("nonpositive or nonfinite normal-system curvature")
        alpha = rr / denom
        solution += alpha * direction
        rhs -= alpha * q
        next_rr = float(np.vdot(rhs, rhs).real)
        if not np.isfinite(next_rr):
            raise FloatingPointError("nonfinite normal-system residual")
        direction = rhs + (next_rr / rr) * direction
        rr = next_rr
        completed += 1
        control.work.counts["inner_iterations"] = (
            control.work.counts.get("inner_iterations", 0) + 1
        )
    # Recompute rather than reporting only the recursively updated CG residual.
    # An iteration cap is not a certificate that the GN subproblem was solved.
    true_residual = initial_rhs - normal_product(solution) if completed else initial_rhs
    true_norm = float(np.linalg.norm(true_residual))
    if not np.isfinite(true_norm):
        raise FloatingPointError("nonfinite recomputed normal-system residual")
    denominator = max(np.sqrt(initial_rr), np.finfo(float).tiny)
    record = {
        "method": "real_parameter_normal_cg",
        "iterations": completed,
        "iteration_limit": iterations,
        "recursive_relative_residual": float(np.sqrt(rr) / denominator),
        "true_relative_residual": true_norm / denominator,
        "converged": bool(true_norm <= denominator * rtol),
        "rtol": rtol,
        "certificate_scope": "normal_residual_only_not_parameter_accuracy_or_backward_error",
        "stop_reason": (
            "normal_residual_target"
            if true_norm <= denominator * rtol
            else "iteration_limit"
        ),
    }
    if not hasattr(control, "inner_solver_history"):
        control.inner_solver_history = []
    control.inner_solver_history.append(record)
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
