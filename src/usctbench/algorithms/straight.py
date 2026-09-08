"""Application wiring for straight-ray solvers; physics lives in operators/."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from usctbench.algorithms._control import InversionControl, add_image_metrics
from usctbench.algorithms.ray import (
    configured_ray_weights,
    coverage_preconditioner,
    image_diagnostic_metrics,
    ray_weight_metrics,
    reference_sound_speed,
    speed_bounds,
    target_delta_tof,
)
from usctbench.core.config import coerce_bool
from usctbench.core.schema import ReconstructionResult
from usctbench.operators.forward.straight_ray import StraightRayProjector
from usctbench.solvers.projected_cg import projected_cg
from usctbench.solvers.row_action import row_action


def reconstruct(case, config, algorithm):
    projector = StraightRayProjector.from_case(
        case, backend=config.parameters.get("projector_backend", "csr")
    )
    target, mask = target_delta_tof(case, projector)
    weights = configured_ray_weights(case, projector, mask, config)
    p = config.parameters
    c0 = reference_sound_speed(case, config)
    low, high = speed_bounds(config)
    if not low <= c0 <= high:
        raise ValueError("reference sound speed must lie within sound_speed_bounds_mps")
    is_cg = algorithm == "straight_cgls"
    lambda_value = float(p.get("lambda", p.get("regularization_lambda", 0)))
    control = InversionControl(
        case,
        config,
        target.reshape(projector.ray_shape),
        default_iterations=(
            30 if is_cg else (10 if algorithm == "straight_sart" else 50)
        ),
        weights=weights.reshape(projector.ray_shape),
        valid_mask=mask.reshape(projector.ray_shape),
        iteration_unit="cgls_step" if is_cg else "full_sweep",
    )
    roi_only = coerce_bool(p.get("roi_update_only", False))
    roi_laplacian = coerce_bool(
        p.get("roi_laplacian", p.get("roi_aware_laplacian", False))
    )
    roi = case.grid.roi_mask if roi_only else None
    initial = np.zeros(case.grid.shape)
    s0 = np.full(case.grid.shape, 1.0 / c0)

    def project(x):
        bounded = np.clip(x, 1.0 / high - 1.0 / c0, 1.0 / low - 1.0 / c0)
        return bounded if roi is None else np.where(roi, bounded, 0)

    def to_speed(x):
        return 1.0 / (s0 + x)

    use_preconditioning = coerce_bool(p.get("coverage_preconditioning", False))
    preconditioner, coverage = None, None
    # Coverage construction is a real adjoint application and is budgeted.
    # It cannot consume held-out receiver values or exceed the adjoint budget.
    if use_preconditioning and not is_cg:
        raise ValueError("coverage_preconditioning is supported by straight_cgls only")
    if use_preconditioning:
        from usctbench.core.stopping import BudgetExhausted

        try:
            preconditioner, coverage = control.call(
                "adjoint",
                coverage_preconditioner,
                projector,
                control.split.train.ravel(),
                control.split.weights.ravel(),
                roi_mask=roi,
                eps=float(p.get("coverage_preconditioner_eps", 1e-12)),
                max_scale=float(p.get("coverage_preconditioner_max_scale", 10)),
                normalize=coerce_bool(p.get("coverage_preconditioner_normalize", True)),
            )
        except BudgetExhausted as exc:
            control.monitor.finish(exc.reason)
    robust_loss = str(p.get("robust_loss", "none")).lower()
    if robust_loss not in {"none", "huber", "irls", "huber_irls"}:
        raise ValueError("robust_loss must be none or huber/irls/huber_irls")
    if not is_cg and robust_loss != "none":
        raise ValueError("robust_loss is supported by straight_cgls only")
    if control.monitor.reason is not None:
        state, metrics = control.output(initial)
    elif is_cg:
        state, metrics = projected_cg(
            projector,
            control,
            initial=initial,
            reference=s0,
            project=project,
            to_image=to_speed,
            damping=float(p.get("damping", lambda_value**2)),
            regularization=str(p.get("regularization", "identity")),
            roi=roi,
            regularization_roi=case.grid.roi_mask if roi_laplacian else None,
            preconditioner=preconditioner,
            huber_delta=(
                float(p.get("huber_delta_s", 5e-7)) if robust_loss != "none" else None
            ),
            irls_stages=p.get("irls_iterations", 3) if robust_loss != "none" else 1,
        )
    else:
        sigma = float(p.get("smooth_sigma", 0))
        every = int(p.get("smooth_every", 0))
        if not np.isfinite(sigma) or sigma < 0 or every < 0:
            raise ValueError("smoothing parameters must be nonnegative and finite")

        def smooth(x, iteration):
            return (
                gaussian_filter(x, sigma, mode="nearest")
                if sigma > 0 and every > 0 and iteration % every == 0
                else x
            )

        state, metrics = row_action(
            projector,
            control,
            initial=initial,
            reference=s0,
            project=project,
            to_image=to_speed,
            relaxation=float(
                p.get("relaxation", 0.2 if algorithm == "straight_sart" else 0.3)
            ),
            subsets=p.get("subsets", 8) if algorithm == "straight_sart" else 1,
            postprocess=smooth,
        )
    sound_speed = to_speed(state)
    metrics.update(
        {
            "backend": "native_siddon_straight_ray",
            "projector_backend": projector.backend,
            "projector_csr_storage_bytes": projector.storage_bytes,
            "roi_update_only": roi_only,
            "roi_laplacian": roi_laplacian,
            "coverage_preconditioning": use_preconditioning,
            "robust_loss": robust_loss,
            "objective_name": (
                "weighted_huber" if robust_loss != "none" else "weighted_least_squares"
            ),
            "iterations_scope": "global across all subsets/IRLS stages",
            **ray_weight_metrics(weights, mask, config),
        }
    )
    if is_cg:
        metrics.update(
            {
                "regularization": str(p.get("regularization", "identity")),
                "regularization_lambda": float(
                    p.get("lambda", p.get("regularization_lambda", 0))
                ),
                "regularization_lambda_squared": float(
                    p.get("damping", lambda_value**2)
                ),
            }
        )
    else:
        metrics.update(
            {
                "relaxation": float(
                    p.get("relaxation", 0.2 if algorithm == "straight_sart" else 0.3)
                ),
                "subsets": p.get("subsets", 8) if algorithm == "straight_sart" else 1,
                "smooth_sigma": float(p.get("smooth_sigma", 0)),
                "smooth_every": int(p.get("smooth_every", 0)),
            }
        )
    add_image_metrics(metrics, sound_speed, case, c0)
    if case.ground_truth.sound_speed_mps is not None:
        metrics.update(
            image_diagnostic_metrics(
                sound_speed,
                case.ground_truth.sound_speed_mps,
                roi_mask=case.grid.roi_mask,
                coverage=coverage,
                boundary_band_pixels=int(p.get("boundary_band_pixels", 4)),
            )
        )
    return ReconstructionResult(
        algorithm=algorithm,
        case_id=case.case_id,
        sound_speed_mps=sound_speed,
        metrics=metrics,
    )
