"""Ray-based USCT reconstruction algorithms and projector utilities."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from usctbench.core.registry import register_algorithm
from usctbench.core.schema import (
    AlgorithmConfig,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)

# Historical imports remain valid after separating physics and numerical solvers.
from usctbench.operators.forward.straight_ray import StraightRayProjector
from usctbench.solvers.algebraic import (  # noqa: F401
    apply_mask,
    masked_norm,
    cgls_solve,
    huber_irls_cgls_solve,
    sirt_solve,
    sart_solve,
    _gaussian_smooth,
    _post_update,
    _regularization_forward,
    _regularization_normal,
    _roi_regularization_image,
    _laplacian,
)


def parameter(config: AlgorithmConfig, key: str, default: Any) -> Any:
    return config.parameters.get(key, default)


def reference_sound_speed(case: USCTCase, config: AlgorithmConfig) -> float:
    value = parameter(
        config,
        "reference_sound_speed_mps",
        case.metadata.get("reference_sound_speed_mps", 1500.0),
    )
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("reference_sound_speed_mps must be positive")
    return value


def speed_bounds(config: AlgorithmConfig) -> tuple[float, float]:
    bounds = parameter(config, "sound_speed_bounds_mps", (1300.0, 1700.0))
    low, high = float(bounds[0]), float(bounds[1])
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high <= low:
        raise ValueError("sound_speed_bounds_mps must be [positive_low, higher_high]")
    return low, high


def valid_ray_mask(case: USCTCase, projector: StraightRayProjector) -> np.ndarray:
    if case.measurement.valid_mask is None:
        return np.ones(projector.n_rays, dtype=bool)
    mask = np.asarray(case.measurement.valid_mask, dtype=bool).reshape(-1)
    if mask.size != projector.n_rays:
        raise ValueError(
            "measurement.valid_mask shape must match transmitter/receiver ray shape"
        )
    return mask


def configured_ray_weights(
    case: USCTCase,
    projector: StraightRayProjector,
    mask: np.ndarray | None,
    config: AlgorithmConfig,
) -> np.ndarray:
    """Return per-ray confidence weights after optional config shaping."""

    threshold = float(
        config.parameters.get(
            "min_ray_weight",
            config.parameters.get(
                "ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)
            ),
        )
    )
    power = float(config.parameters.get("ray_weight_power", 1.0))
    return ray_weights(case, projector, mask, min_weight=threshold, weight_power=power)


def ray_weights(
    case: USCTCase,
    projector: StraightRayProjector,
    mask: np.ndarray | None = None,
    *,
    min_weight: float = 0.0,
    weight_power: float = 1.0,
) -> np.ndarray:
    """Return per-ray confidence weights clipped to [0, 1].

    ``min_weight`` and ``weight_power`` let algorithms sharpen feature-derived
    confidence without editing the feature case. They default to the historical
    behavior: no threshold and no exponentiation.
    """

    if min_weight < 0.0 or min_weight > 1.0:
        raise ValueError("min_ray_weight/ray_weight_threshold must be in [0, 1]")
    if weight_power <= 0.0:
        raise ValueError("ray_weight_power must be positive")

    source = case.measurement.ray_weights
    if source is None:
        source = case.measurement.feature_quality
    if source is None:
        weights = np.ones(projector.n_rays, dtype=float)
    else:
        weights = np.asarray(source, dtype=float).reshape(-1)
        if weights.size != projector.n_rays:
            raise ValueError(
                "measurement.ray_weights/feature_quality shape must match transmitter/receiver ray shape"
            )
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
    if min_weight > 0.0:
        weights = np.where(weights >= float(min_weight), weights, 0.0)
    if weight_power != 1.0:
        weights = np.power(weights, float(weight_power))
    if mask is not None:
        weights = np.where(np.asarray(mask, dtype=bool), weights, 0.0)
    return weights


def ray_weight_metrics(
    weights: np.ndarray, mask: np.ndarray, config: AlgorithmConfig
) -> dict[str, float]:
    """Summarize transformed ray weights used by a reconstruction."""

    valid = np.asarray(mask, dtype=bool).reshape(-1)
    values = np.asarray(weights, dtype=float).reshape(-1)
    if not np.any(valid):
        return {
            "ray_weight_mean": 0.0,
            "ray_weight_nonzero_fraction": 0.0,
            "ray_weight_p10": 0.0,
            "ray_weight_p50": 0.0,
            "ray_weight_p90": 0.0,
            "min_ray_weight": float(
                config.parameters.get(
                    "min_ray_weight",
                    config.parameters.get(
                        "ray_weight_min",
                        config.parameters.get("ray_weight_threshold", 0.0),
                    ),
                )
            ),
            "ray_weight_power": float(config.parameters.get("ray_weight_power", 1.0)),
        }
    used = values[valid]
    return {
        "ray_weight_mean": float(np.mean(used)),
        "ray_weight_nonzero_fraction": float(np.mean(used > 0.0)),
        "ray_weight_p10": float(np.percentile(used, 10.0)),
        "ray_weight_p50": float(np.percentile(used, 50.0)),
        "ray_weight_p90": float(np.percentile(used, 90.0)),
        "min_ray_weight": float(
            config.parameters.get(
                "min_ray_weight",
                config.parameters.get(
                    "ray_weight_min", config.parameters.get("ray_weight_threshold", 0.0)
                ),
            )
        ),
        "ray_weight_power": float(config.parameters.get("ray_weight_power", 1.0)),
    }


def coverage_preconditioner(
    projector: StraightRayProjector,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    roi_mask: np.ndarray | None = None,
    eps: float = 1.0e-12,
    max_scale: float = 10.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return D ~= 1/sqrt(A.T @ ray_weights + eps) and the coverage map."""

    ray_values = np.asarray(mask, dtype=float).reshape(-1)
    if weights is not None:
        ray_values = ray_values * np.clip(
            np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0
        )
    coverage = projector.adjoint(ray_values)
    preconditioner = 1.0 / np.sqrt(np.maximum(coverage, 0.0) + max(float(eps), 0.0))
    roi = (
        np.ones(projector.grid.shape, dtype=bool)
        if roi_mask is None
        else np.asarray(roi_mask, dtype=bool)
    )
    finite_roi = roi & np.isfinite(preconditioner) & (coverage > 0.0)
    if normalize and np.any(finite_roi):
        median = float(np.median(preconditioner[finite_roi]))
        if median > 0.0:
            preconditioner = preconditioner / median
    preconditioner = np.clip(
        np.nan_to_num(preconditioner, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        float(max_scale),
    )
    if roi_mask is not None:
        preconditioner = np.where(roi, preconditioner, 0.0)
    return preconditioner, coverage


def image_diagnostic_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    roi_mask: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
    boundary_band_pixels: int = 4,
) -> dict[str, float]:
    """Return coverage/error and boundary-band image diagnostics."""

    pred = np.asarray(prediction, dtype=float)
    target = np.asarray(truth, dtype=float)
    roi = (
        np.ones(pred.shape, dtype=bool)
        if roi_mask is None
        else np.asarray(roi_mask, dtype=bool)
    )
    finite = roi & np.isfinite(pred) & np.isfinite(target)
    error = np.abs(pred - target)
    metrics: dict[str, float] = {}
    if coverage is not None:
        cov = np.asarray(coverage, dtype=float)
        metrics["coverage_abs_error_corr"] = _corr(cov[finite], error[finite])
        metrics["coverage_mean"] = _masked_mean(cov, finite)
        metrics["coverage_p10"] = _masked_percentile(cov, finite, 10.0)
    boundary = boundary_band_mask(roi, pixels=boundary_band_pixels)
    boundary_finite = boundary & np.isfinite(pred) & np.isfinite(target)
    metrics["boundary_band_pixels"] = float(boundary_band_pixels)
    metrics["boundary_band_fraction"] = float(
        np.sum(boundary_finite) / max(1, int(np.sum(finite)))
    )
    if np.any(boundary_finite):
        metrics["boundary_band_rmse"] = float(
            np.sqrt(np.mean((pred[boundary_finite] - target[boundary_finite]) ** 2))
        )
        metrics["boundary_band_mae"] = float(np.mean(error[boundary_finite]))
    else:
        metrics["boundary_band_rmse"] = float("nan")
        metrics["boundary_band_mae"] = float("nan")
    return metrics


def boundary_band_mask(roi_mask: np.ndarray, *, pixels: int) -> np.ndarray:
    roi = np.asarray(roi_mask, dtype=bool)
    if pixels <= 0 or not np.any(roi):
        return np.zeros_like(roi, dtype=bool)
    eroded = roi.copy()
    for _ in range(int(pixels)):
        padded = np.pad(
            eroded, ((1, 1), (1, 1)), mode="constant", constant_values=False
        )
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return roi & ~eroded


def target_delta_tof(
    case: USCTCase, projector: StraightRayProjector
) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.delta_tof_s is None:
        raise ValueError(
            "straight-ray sound-speed reconstruction requires measurement.delta_tof_s"
        )
    target = np.asarray(case.measurement.delta_tof_s, dtype=float).reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError(
            "measurement.delta_tof_s shape must match transmitter/receiver ray shape"
        )
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask


def target_attenuation_integral(
    case: USCTCase, projector: StraightRayProjector
) -> tuple[np.ndarray, np.ndarray]:
    if case.measurement.log_amp is None:
        raise ValueError("attenuation reconstruction requires measurement.log_amp")
    log_amp = np.asarray(case.measurement.log_amp, dtype=float)
    if log_amp.ndim == 3:
        log_amp = np.nanmean(log_amp, axis=0)
    target = -log_amp.reshape(-1)
    if target.size != projector.n_rays:
        raise ValueError(
            "measurement.log_amp shape must match transmitter/receiver ray shape"
        )
    mask = valid_ray_mask(case, projector) & np.isfinite(target)
    target = np.where(mask, target, 0.0)
    return target, mask


def residual_metrics(initial_norm: float, final_norm: float) -> dict[str, float]:
    if initial_norm > 0.0:
        relative = final_norm / initial_norm
        reduction = 1.0 - relative
    else:
        relative = 0.0 if final_norm == 0.0 else float("inf")
        reduction = 0.0
    return {
        "data_residual_norm": float(final_norm),
        "initial_data_residual_norm": float(initial_norm),
        "data_relative_residual": float(relative),
        "data_residual_reduction": float(reduction),
    }


def slowness_to_sound_speed(
    delta_slowness: np.ndarray, c0: float, bounds_mps: tuple[float, float]
) -> np.ndarray:
    low, high = bounds_mps
    min_slowness = 1.0 / high
    max_slowness = 1.0 / low
    slowness = np.clip((1.0 / c0) + delta_slowness, min_slowness, max_slowness)
    return 1.0 / slowness


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 2:
        return float("nan")
    x = x[finite]
    y = y[finite]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=float)[finite]))


def _masked_percentile(
    values: np.ndarray, mask: np.ndarray, percentile: float
) -> float:
    finite = np.asarray(mask, dtype=bool) & np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(
        np.percentile(np.asarray(values, dtype=float)[finite], float(percentile))
    )


def run_with_failure_capture(
    algorithm: str,
    case: USCTCase,
    func: Any,
) -> ReconstructionResult:
    started = time.perf_counter()
    try:
        result = func()
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised through CLI/integration failures.
        return ReconstructionResult(
            algorithm=algorithm,
            case_id=case.case_id,
            runtime_s=time.perf_counter() - started,
            status=ResultStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    result.runtime_s = time.perf_counter() - started
    reason = result.metrics.get("stop_reason")
    if reason in {"numerical_failure", "linear_solver_breakdown"}:
        # A finite last checkpoint is useful for diagnosis, not a successful solve.
        result.status = ResultStatus.FAILED
        result.failure_reason = f"Native inversion terminated with {reason}"
        result.metrics["recovered_checkpoint_only"] = True
    return result


class StraightRayCGLSAlgorithm:
    name = "straight_cgls"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        from usctbench.algorithms.straight import reconstruct

        return run_with_failure_capture(
            self.name, case, lambda: reconstruct(case, config, self.name)
        )


class StraightRaySIRTAlgorithm:
    name = "straight_sirt"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        from usctbench.algorithms.straight import reconstruct

        return run_with_failure_capture(
            self.name, case, lambda: reconstruct(case, config, self.name)
        )


class StraightRaySARTAlgorithm:
    name = "straight_sart"

    def run(self, case: USCTCase, config: AlgorithmConfig) -> ReconstructionResult:
        from usctbench.algorithms.straight import reconstruct

        return run_with_failure_capture(
            self.name, case, lambda: reconstruct(case, config, self.name)
        )


def register_ray_algorithms(*, replace: bool = False) -> None:
    """Register built-in straight-ray sound-speed algorithms."""

    register_algorithm(
        "straight_sart",
        StraightRaySARTAlgorithm,
        description="Straight-ray SART sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_sirt",
        StraightRaySIRTAlgorithm,
        description="Straight-ray SIRT sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )
    register_algorithm(
        "straight_cgls",
        StraightRayCGLSAlgorithm,
        description="Straight-ray CGLS sound-speed reconstruction.",
        tags=("ray", "sound-speed"),
        replace=replace,
    )


__all__ = [
    "StraightRayProjector",
    "StraightRayCGLSAlgorithm",
    "StraightRaySARTAlgorithm",
    "StraightRaySIRTAlgorithm",
    "register_ray_algorithms",
]
