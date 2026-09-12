"""Low-level algebraic solvers retained for compatibility.

Registry algorithms use the budgeted solvers in least_squares and row_action.
These array-only helpers retain their historical fixed-iteration API.
"""

from __future__ import annotations

import numpy as np

from usctbench.operators.straight_ray import StraightRayProjector


def apply_mask(
    values: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    sqrt_weights: bool = True,
) -> np.ndarray:
    out = np.asarray(values, dtype=float).reshape(-1).copy()
    out[~mask] = 0.0
    if weights is not None:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if weight_values.size != out.size:
            raise ValueError("weights shape must match values")
        multiplier = (
            np.sqrt(np.clip(weight_values, 0.0, 1.0))
            if sqrt_weights
            else np.clip(weight_values, 0.0, 1.0)
        )
        out *= multiplier
    return out


def masked_norm(
    values: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None
) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite_mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if weights is None:
        return float(np.linalg.norm(values[finite_mask]))
    weight_values = np.asarray(weights, dtype=float).reshape(-1)
    weighted = values[finite_mask] * np.sqrt(
        np.clip(weight_values[finite_mask], 0.0, 1.0)
    )
    return float(np.linalg.norm(weighted))


def cgls_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    damping: float = 0.0,
    regularization: str = "identity",
    weights: np.ndarray | None = None,
    preconditioner: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    initial_image: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Solve a masked least-squares system with CGLS."""

    d = (
        np.ones(projector.grid.shape, dtype=float)
        if preconditioner is None
        else np.asarray(preconditioner, dtype=float)
    )
    if d.shape != projector.grid.shape:
        raise ValueError("preconditioner must match projector grid shape")
    if roi_mask is not None:
        d = np.where(np.asarray(roi_mask, dtype=bool), d, 0.0)
    if initial_image is None:
        z = np.zeros(projector.grid.shape, dtype=float)
    else:
        initial = np.asarray(initial_image, dtype=float)
        if initial.shape != projector.grid.shape:
            raise ValueError("initial_image must match projector grid shape")
        z = np.divide(
            initial, d, out=np.zeros_like(initial, dtype=float), where=d > 0.0
        )
    x = d * z
    residual = apply_mask(target - projector.forward(x), mask, weights)
    s = d * projector.adjoint(apply_mask(residual, mask, weights))
    if damping > 0:
        s = s - d * damping * _regularization_normal(
            x, regularization, roi_mask=roi_mask
        )
    p = s.copy()
    gamma = float(np.vdot(s, s))
    residual_norms = [float(np.linalg.norm(residual[mask]))]

    for _ in range(max(0, int(iterations))):
        if gamma <= 0:
            break
        q = apply_mask(projector.forward(d * p), mask, weights)
        denom = float(np.vdot(q, q))
        if damping > 0:
            reg_p = _regularization_forward(d * p, regularization, roi_mask=roi_mask)
            denom += damping * float(np.vdot(reg_p, reg_p))
        if denom <= 0:
            break
        alpha = gamma / denom
        z = z + alpha * p
        x = d * z
        residual = residual - alpha * q
        s_next = d * projector.adjoint(apply_mask(residual, mask, weights))
        if damping > 0:
            s_next = s_next - d * damping * _regularization_normal(
                x, regularization, roi_mask=roi_mask
            )
        gamma_next = float(np.vdot(s_next, s_next))
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        if gamma_next <= 1.0e-30:
            break
        beta = gamma_next / gamma
        p = s_next + beta * p
        s = s_next
        gamma = gamma_next
    return x, residual_norms


def huber_irls_cgls_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    damping: float = 0.0,
    regularization: str = "identity",
    weights: np.ndarray | None = None,
    preconditioner: np.ndarray | None = None,
    roi_mask: np.ndarray | None = None,
    huber_delta: float = 5.0e-7,
    irls_iterations: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Solve with Huber-style IRLS residual down-weighting."""

    base_weights = (
        np.ones(projector.n_rays, dtype=float)
        if weights is None
        else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    )
    robust_weights = np.ones(projector.n_rays, dtype=float)
    x = np.zeros(projector.grid.shape, dtype=float)
    residual_curve = [
        masked_norm(np.asarray(target, dtype=float).reshape(-1), mask, base_weights)
    ]
    delta = max(float(huber_delta), 1.0e-15)
    for _ in range(max(1, int(irls_iterations))):
        effective_weights = base_weights * robust_weights
        x, inner_curve = cgls_solve(
            projector,
            target,
            mask,
            iterations=iterations,
            damping=damping,
            regularization=regularization,
            weights=effective_weights,
            preconditioner=preconditioner,
            roi_mask=roi_mask,
            initial_image=x,
        )
        raw_residual = np.asarray(target, dtype=float).reshape(-1) - projector.forward(
            x
        )
        abs_residual = np.abs(raw_residual)
        robust_weights = np.where(
            abs_residual <= delta, 1.0, delta / np.maximum(abs_residual, delta)
        )
        robust_weights = np.where(np.asarray(mask, dtype=bool), robust_weights, 0.0)
        if inner_curve:
            residual_curve.extend(inner_curve[1:])
        residual_curve.append(
            masked_norm(raw_residual, mask, base_weights * robust_weights)
        )
    return x, residual_curve


def _regularization_forward(
    image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None
) -> np.ndarray:
    kind_normalized = str(kind).lower()
    array = _roi_regularization_image(image, roi_mask)
    if kind_normalized in ("identity", "l2"):
        return array
    if kind_normalized in ("laplacian", "roughness"):
        return _laplacian(array)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _regularization_normal(
    image: np.ndarray, kind: str, *, roi_mask: np.ndarray | None = None
) -> np.ndarray:
    kind_normalized = str(kind).lower()
    roi = None if roi_mask is None else np.asarray(roi_mask, dtype=bool)
    array = _roi_regularization_image(image, roi)
    if kind_normalized in ("identity", "l2"):
        normal = array
        return normal if roi is None else np.where(roi, normal, 0.0)
    if kind_normalized in ("laplacian", "roughness"):
        normal = _laplacian(_laplacian(array))
        return normal if roi is None else np.where(roi, normal, 0.0)
    raise ValueError("regularization must be 'identity' or 'laplacian'")


def _roi_regularization_image(
    image: np.ndarray, roi_mask: np.ndarray | None
) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if roi_mask is None:
        return array
    return np.where(np.asarray(roi_mask, dtype=bool), array, 0.0)


def _laplacian(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    padded = np.pad(array, ((1, 1), (1, 1)), mode="edge")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )


def sirt_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    nonnegative: bool = False,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run a normalized simultaneous iterative reconstruction update."""

    x = np.zeros(projector.grid.shape, dtype=float)
    row_norm = projector.row_norms(power=1)
    row_norm[row_norm <= 0.0] = 1.0
    col_norm = projector.col_norms(power=1)
    if weights is not None:
        col_norm = projector.adjoint(
            np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
            * np.asarray(mask, dtype=float).reshape(-1)
        )
    col_norm[col_norm <= 0.0] = 1.0
    residual_norms: list[float] = []
    for _ in range(max(0, int(iterations))):
        residual_raw = np.asarray(target, dtype=float).reshape(-1) - projector.forward(
            x
        )
        residual = apply_mask(residual_raw, mask, weights)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
        update_values = apply_mask(
            residual_raw / row_norm, mask, weights, sqrt_weights=False
        )
        update = projector.adjoint(update_values) / col_norm
        x = x + float(relaxation) * update
        x = _post_update(
            x,
            iteration_index=_ + 1,
            nonnegative=nonnegative,
            smooth_sigma=smooth_sigma,
            smooth_every=smooth_every,
            roi_mask=roi_mask,
        )
    residual = apply_mask(target - projector.forward(x), mask, weights)
    residual_norms.append(float(np.linalg.norm(residual[mask])))
    return x, residual_norms


def sart_solve(
    projector: StraightRayProjector,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int,
    relaxation: float,
    subsets: int = 8,
    smooth_sigma: float = 0.0,
    smooth_every: int = 0,
    roi_mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Run subset-normalized SART updates over masked rays."""

    x = np.zeros(projector.grid.shape, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    weight_values = (
        np.ones(projector.n_rays, dtype=float)
        if weights is None
        else np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    )
    row_norm = projector.row_norms(power=1)
    row_norm[row_norm <= 0.0] = 1.0
    ray_ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    subset_count = max(1, min(int(subsets), int(ray_ids.size) if ray_ids.size else 1))
    residual_norms: list[float] = []

    for _ in range(max(0, int(iterations))):
        for subset in np.array_split(ray_ids, subset_count):
            if subset.size == 0:
                continue
            subset_mask = np.zeros(projector.n_rays, dtype=bool)
            subset_mask[subset] = True
            residual_raw = target - projector.forward(x)
            residual = apply_mask(residual_raw, subset_mask, weight_values)
            col_norm = projector.adjoint(subset_mask.astype(float) * weight_values)
            col_norm[col_norm <= 0.0] = 1.0
            update_values = apply_mask(
                residual_raw / row_norm, subset_mask, weight_values, sqrt_weights=False
            )
            update = projector.adjoint(update_values) / col_norm
            x = x + float(relaxation) * update
            x = _post_update(
                x,
                iteration_index=_ + 1,
                nonnegative=False,
                smooth_sigma=smooth_sigma,
                smooth_every=smooth_every,
                roi_mask=roi_mask,
            )
        residual = apply_mask(target - projector.forward(x), mask, weight_values)
        residual_norms.append(float(np.linalg.norm(residual[mask])))
    return x, residual_norms


def _post_update(
    image: np.ndarray,
    *,
    iteration_index: int,
    nonnegative: bool,
    smooth_sigma: float,
    smooth_every: int,
    roi_mask: np.ndarray | None,
) -> np.ndarray:
    updated = np.asarray(image, dtype=float)
    if roi_mask is not None:
        updated = np.where(np.asarray(roi_mask, dtype=bool), updated, 0.0)
    if (
        smooth_sigma > 0.0
        and smooth_every > 0
        and iteration_index % int(smooth_every) == 0
    ):
        updated = _gaussian_smooth(updated, float(smooth_sigma))
        if roi_mask is not None:
            updated = np.where(np.asarray(roi_mask, dtype=bool), updated, 0.0)
    if nonnegative:
        updated = np.maximum(updated, 0.0)
    return updated


def _gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ModuleNotFoundError:
        return image
    return np.asarray(
        gaussian_filter(image, sigma=float(sigma), mode="nearest"), dtype=float
    )
