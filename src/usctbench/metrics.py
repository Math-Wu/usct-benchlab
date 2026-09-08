"""Metric helpers for USCT reconstructions."""

from __future__ import annotations

import math
import hashlib
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity


def _masked_values(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target must have the same shape")
    finite = np.isfinite(truth)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("no finite pixels available for metric computation")
    if not np.all(np.isfinite(pred[finite])):
        raise FloatingPointError("nonfinite reconstruction inside the evaluation ROI")
    return pred[finite], truth[finite]


def compute_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "",
    data_range: float | None = None,
) -> dict[str, float]:
    """Compute scalar image metrics over finite ROI pixels."""

    pred_image = np.asarray(prediction, dtype=float)
    truth_image = np.asarray(target, dtype=float)
    finite_mask = np.isfinite(truth_image)
    if mask is not None:
        finite_mask &= np.asarray(mask, dtype=bool)
    pred, truth = _masked_values(pred_image, truth_image, finite_mask)
    error = pred - truth
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    target_range = float(np.max(truth) - np.min(truth))
    nrmse = rmse / target_range if target_range > 0 else 0.0 if rmse == 0 else math.inf
    if data_range is None:
        data_range = (
            target_range if target_range > 0 else max(float(np.max(np.abs(truth))), 1.0)
        )
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    psnr = math.inf if mse == 0 else 20.0 * math.log10(data_range / math.sqrt(mse))
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}nrmse": nrmse,
        f"{prefix}psnr": psnr,
        f"{prefix}ssim": _image_ssim(
            pred_image,
            truth_image,
            valid_mask=finite_mask,
            data_range=data_range,
        ),
        f"{prefix}global_ssim": _global_ssim(pred, truth, data_range=data_range),
    }


def non_water_tissue_mask(
    target: np.ndarray,
    *,
    water_speed_mps: float = 1500.0,
    water_tolerance_mps: float = 0.1,
) -> np.ndarray:
    """Evaluation-only mask: remove edge-connected water, retain enclosed tissue.

    An enclosed tissue pixel equal to water speed is retained. No reconstructed
    values or acquisition/inversion ROI enter this deterministic GT mask.
    """
    from scipy.ndimage import binary_propagation

    truth = np.asarray(target, dtype=float)
    if truth.ndim != 2 or min(truth.shape) < 1:
        raise ValueError("tissue evaluation requires a 2-D ground-truth image")
    if (
        not np.isfinite(water_speed_mps)
        or water_speed_mps <= 0
        or not np.isfinite(water_tolerance_mps)
        or water_tolerance_mps < 0
    ):
        raise ValueError(
            "water speed must be positive and tolerance finite/nonnegative"
        )
    finite = np.isfinite(truth)
    water = finite & (np.abs(truth - water_speed_mps) <= water_tolerance_mps)
    seeds = np.zeros(truth.shape, dtype=bool)
    seeds[[0, -1], :] = water[[0, -1], :]
    seeds[:, [0, -1]] = water[:, [0, -1]]
    exterior = binary_propagation(seeds, mask=water)
    return finite & ~exterior


def compute_regional_image_metrics(
    prediction: np.ndarray,
    target: np.ndarray | None,
    *,
    primary_region: str = "tissue",
    water_speed_mps: float = 1500.0,
    water_tolerance_mps: float = 0.1,
    data_range_mps: float | None = None,
) -> dict[str, Any]:
    """Post-reconstruction benchmark scores; never used for stopping or updates.

    Preserve full-image and tissue scores under explicit prefixes. Unprefixed
    RMSE/PSNR/SSIM follow the declared primary region. Both regions use one fixed
    range from the finite full GT (or an explicit protocol range), not a range
    chosen separately for each reconstruction. Missing/empty GT gives no score.
    """
    if primary_region not in {"tissue", "full_image"}:
        raise ValueError("primary_region must be tissue or full_image")
    names = ("rmse", "mae", "nrmse", "psnr", "ssim", "global_ssim")
    info = {
        "version": "tissue_external_water_v1",
        "primary_region": primary_region,
        "mask_source": "ground_truth_boundary_connected_water_evaluation_only",
        "used_for_reconstruction_or_stopping": False,
        "water_speed_mps": float(water_speed_mps),
        "water_tolerance_mps": float(water_tolerance_mps),
    }
    result = {key: None for key in names}
    # Clear legacy baseline scores when the requested region is unavailable.
    result.update(
        {
            "water_" + key: None
            for key in (
                "baseline_rmse",
                "reconstruction_rmse",
                "absolute_rmse_improvement",
                "relative_rmse_improvement",
                "improved",
            )
        }
    )
    result["image_evaluation"] = info
    result["primary_image_region"] = primary_region
    if target is None:
        info["status"] = "ground_truth_unavailable"
        return result
    truth = np.asarray(target, dtype=float)
    tissue = non_water_tissue_mask(
        truth, water_speed_mps=water_speed_mps, water_tolerance_mps=water_tolerance_mps
    )
    finite = np.isfinite(truth)
    if not finite.any():
        info["status"] = "ground_truth_unavailable"
        return result
    if data_range_mps is None:
        data_range_mps = float(np.ptp(truth[finite]))
        if data_range_mps == 0:
            data_range_mps = max(float(np.max(np.abs(truth[finite]))), 1.0)
    full = compute_image_metrics(prediction, truth, data_range=data_range_mps)
    result.update({"full_image_" + k: v for k, v in full.items()})
    info.update(
        {
            "data_range_mps": float(data_range_mps),
            "tissue_pixels": int(tissue.sum()),
            "full_image_pixels": int(finite.sum()),
            "mask_sha256": hashlib.sha256(
                np.array(tissue.shape, dtype="<i8").tobytes() + tissue.tobytes()
            ).hexdigest(),
            "ssim_window": "complete valid windows, up to 7 pixels; global fallback for tiny masks",
        }
    )
    if tissue.any():
        scores = compute_image_metrics(
            prediction, truth, mask=tissue, data_range=data_range_mps
        )
        result.update({"tissue_" + k: v for k, v in scores.items()})
        if primary_region == "tissue":
            result.update(scores)
    if primary_region == "full_image":
        result.update(full)
    if tissue.any() or primary_region == "full_image":
        result.update(
            compute_baseline_improvement_metrics(
                prediction,
                truth,
                water_speed_mps,
                mask=tissue if primary_region == "tissue" else finite,
            )
        )
        info["water_initialization_comparison_region"] = primary_region
    water = finite & ~tissue
    info["water_background_pixels"] = int(water.sum())
    if water.any():
        error = np.asarray(prediction)[water] - truth[water]
        result["water_background_rmse"] = float(np.sqrt(np.mean(error**2)))
        result["water_background_bias"] = float(np.mean(error))
    info["status"] = (
        "ok" if tissue.any() or primary_region == "full_image" else "empty_tissue"
    )
    return result


def compute_baseline_improvement_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline: float | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "water_",
) -> dict[str, float | bool]:
    """Compare reconstruction RMSE against a baseline image or scalar."""

    pred, truth = _masked_values(prediction, target, mask)
    baseline_array = np.asarray(baseline, dtype=float)
    if baseline_array.ndim == 0:
        base = np.full_like(np.asarray(target, dtype=float), float(baseline_array))
    else:
        base = baseline_array
    base_values, _ = _masked_values(base, target, mask)
    reconstruction_rmse = math.sqrt(float(np.mean((pred - truth) ** 2)))
    baseline_rmse = math.sqrt(float(np.mean((base_values - truth) ** 2)))
    if baseline_rmse == 0.0:
        relative = math.inf if reconstruction_rmse < baseline_rmse else 0.0
    else:
        relative = (baseline_rmse - reconstruction_rmse) / baseline_rmse
    return {
        f"{prefix}baseline_rmse": baseline_rmse,
        f"{prefix}reconstruction_rmse": reconstruction_rmse,
        f"{prefix}absolute_rmse_improvement": baseline_rmse - reconstruction_rmse,
        f"{prefix}relative_rmse_improvement": relative,
        f"{prefix}improved": reconstruction_rmse < baseline_rmse,
    }


def _global_ssim(
    prediction: np.ndarray, target: np.ndarray, *, data_range: float
) -> float:
    """Compute a deterministic global SSIM approximation without SciPy."""

    pred = prediction.astype(float)
    truth = target.astype(float)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(np.mean(pred))
    mu_y = float(np.mean(truth))
    var_x = float(np.mean((pred - mu_x) ** 2))
    var_y = float(np.mean((truth - mu_y) ** 2))
    cov_xy = float(np.mean((pred - mu_x) * (truth - mu_y)))
    denom = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    if denom == 0:
        return 1.0
    return float(((2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)) / denom)


def _image_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray,
    data_range: float,
) -> float:
    """Compute standard 2-D SSIM with invalid pixels neutralized."""

    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if pred.ndim != 2 or truth.ndim != 2:
        return _global_ssim(pred[valid], truth[valid], data_range=data_range)
    if not np.any(valid):
        raise ValueError("no finite pixels available for metric computation")

    rows, cols = np.where(valid)
    row_slice = slice(int(rows.min()), int(rows.max()) + 1)
    col_slice = slice(int(cols.min()), int(cols.max()) + 1)
    pred_crop = pred[row_slice, col_slice].copy()
    truth_crop = truth[row_slice, col_slice].copy()
    valid_crop = valid[row_slice, col_slice]
    if min(pred_crop.shape) < 3:
        return _global_ssim(pred[valid], truth[valid], data_range=data_range)

    fill_value = float(np.median(truth[valid]))
    pred_crop[~valid_crop] = fill_value
    truth_crop[~valid_crop] = fill_value
    pred_crop[~np.isfinite(pred_crop)] = fill_value
    truth_crop[~np.isfinite(truth_crop)] = fill_value

    win_size = min(7, pred_crop.shape[0], pred_crop.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        return _global_ssim(pred[valid], truth[valid], data_range=data_range)
    _, similarity_map = structural_similarity(
        truth_crop,
        pred_crop,
        data_range=float(data_range) if data_range > 0 else 1.0,
        win_size=win_size,
        full=True,
    )
    # Exclude padded background and incomplete windows from the ROI score.
    from scipy.ndimage import binary_erosion

    interior = binary_erosion(valid_crop, structure=np.ones((win_size, win_size)))
    if not np.any(interior):
        return _global_ssim(pred[valid], truth[valid], data_range=data_range)
    return float(np.mean(similarity_map[interior]))


def residual_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    prefix: str = "data_",
) -> dict[str, float]:
    """Compute finite residual norms between predicted and observed data."""

    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed data must have the same shape")
    finite = np.isfinite(pred) & np.isfinite(obs)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        raise ValueError("no finite data samples available for residual metrics")
    residual = pred[finite] - obs[finite]
    obs_values = obs[finite]
    norm = float(np.linalg.norm(residual))
    observed_norm = float(np.linalg.norm(obs_values))
    return {
        f"{prefix}residual_norm": norm,
        f"{prefix}observed_norm": observed_norm,
        f"{prefix}relative_residual": (
            norm / observed_norm if observed_norm > 0 else math.inf if norm > 0 else 0.0
        ),
        f"{prefix}rmse": float(np.sqrt(np.mean(residual**2))),
        f"{prefix}mae": float(np.mean(np.abs(residual))),
        f"{prefix}num_samples": float(residual.size),
    }


def baseline_improvement(
    baseline_error: float,
    reconstruction_error: float,
    *,
    prefix: str = "baseline_",
) -> dict[str, float | bool]:
    """Report absolute and relative improvement over a baseline error."""

    baseline = float(baseline_error)
    reconstruction = float(reconstruction_error)
    if baseline < 0 or reconstruction < 0:
        raise ValueError("errors must be non-negative")
    if baseline == 0:
        relative = math.inf if reconstruction < baseline else 0.0
    else:
        relative = (baseline - reconstruction) / baseline
    return {
        f"{prefix}error": baseline,
        f"{prefix}reconstruction_error": reconstruction,
        f"{prefix}absolute_improvement": baseline - reconstruction,
        f"{prefix}relative_improvement": relative,
        f"{prefix}improved": reconstruction < baseline,
    }
