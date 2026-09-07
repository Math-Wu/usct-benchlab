"""Leakage-aware validation masks and real/complex measurement residuals."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def _broadcast(value, shape, *, dtype, name):
    try:
        return np.broadcast_to(np.asarray(value, dtype=dtype), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must broadcast to data shape {shape}") from exc


def residual_statistics(predicted, observed, *, mask=None, weights=None) -> dict:
    """Compute norms without discarding imaginary parts or invalid predictions.

    Non-finite observations are missing data. A non-finite prediction at an
    otherwise usable observation is a numerical failure, not a removable sample.
    Zero observed norm makes a relative residual undefined (JSON null).
    Weights are nonnegative precision weights, i.e. norms use sqrt(weights).
    """
    pred, obs = np.asarray(predicted), np.asarray(observed)
    if pred.shape != obs.shape:
        raise ValueError("predicted and observed data must have the same shape")
    active = np.isfinite(obs)
    if mask is not None:
        active &= _broadcast(mask, obs.shape, dtype=bool, name="mask")
    weight = _broadcast(
        1.0 if weights is None else weights, obs.shape, dtype=float, name="weights"
    )
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("weights must be finite and nonnegative")
    active &= weight > 0
    count = int(active.sum())
    result = {"num_samples": count, "status": "ok" if count else "empty"}
    if not count:
        for key in (
            "residual_norm",
            "observed_norm",
            "relative_residual",
            "rmse",
            "mae",
            "weighted_residual_norm",
            "weighted_observed_norm",
            "weighted_relative_residual",
        ):
            result[key] = None
        return result
    if not np.all(np.isfinite(pred[active])):
        raise FloatingPointError("non-finite prediction on an active observation")
    residual = pred[active] - obs[active]
    norm = float(np.linalg.norm(residual))
    observed_norm = float(np.linalg.norm(obs[active]))
    weighted_norm = float(np.linalg.norm(np.sqrt(weight[active]) * residual))
    weighted_observed = float(np.linalg.norm(np.sqrt(weight[active]) * obs[active]))
    if not np.all(np.isfinite([norm, observed_norm, weighted_norm, weighted_observed])):
        raise FloatingPointError("overflow in measurement residual norm")
    result.update(
        residual_norm=norm,
        observed_norm=observed_norm,
        relative_residual=norm / observed_norm if observed_norm > 0 else None,
        rmse=float(norm / np.sqrt(count)),
        mae=float(np.mean(np.abs(residual))),
        weighted_residual_norm=weighted_norm,
        weighted_observed_norm=weighted_observed,
        weighted_relative_residual=(
            weighted_norm / weighted_observed if weighted_observed > 0 else None
        ),
    )
    return result


@dataclass(frozen=True)
class DataSplit:
    """Read-only disjoint masks; validation data never contributes to training."""

    train: np.ndarray
    receiver: np.ndarray
    frequency: np.ndarray
    joint: np.ndarray
    unused: np.ndarray
    weights: np.ndarray
    metadata: dict

    @property
    def validation(self):
        return self.receiver | self.frequency | self.joint

    def evaluate(self, predicted, observed) -> dict:
        return {
            name: residual_statistics(
                predicted, observed, mask=getattr(self, name), weights=self.weights
            )
            for name in ("train", "receiver", "frequency", "joint")
        }


def _indices(explicit, fraction, candidates, rng, name):
    fraction = float(fraction)
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError(f"{name}_fraction must be in [0, 1)")
    if explicit is not None:
        raw = np.asarray(explicit)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
            raise ValueError(f"{name} indices must be a one-dimensional integer list")
        ids = raw.astype(int)
        if len(np.unique(ids)) != len(ids) or not np.all(np.isin(ids, candidates)):
            raise ValueError(
                f"{name} indices must be unique and have valid observations"
            )
        if fraction:
            raise ValueError(f"use explicit {name} indices or a fraction, not both")
        return np.sort(ids)
    count = max(1, int(np.floor(len(candidates) * fraction))) if fraction else 0
    if count >= len(candidates) and count:
        raise ValueError(
            f"not enough groups for a nonempty {name} holdout and training"
        )
    return np.sort(rng.choice(candidates, count, replace=False))


def make_data_split(
    observed,
    *,
    valid_mask=None,
    weights=None,
    receiver_indices=None,
    frequency_indices=None,
    receiver_fraction=0.0,
    frequency_fraction=0.0,
    seed=0,
    exclude_reciprocal=True,
    tx_positions=None,
    rx_positions=None,
) -> DataSplit:
    """Hold out complete receivers/frequencies, not random scalar samples.

    Pair data uses (tx, rx); frequency data uses (frequency, tx, rx). On reciprocal
    acquisitions, transmitters colocated with held-out receivers are also removed
    from training, preventing the reverse channel leaking the same information.
    Receiver/frequency/joint validation masks are disjoint. Reverse-only channels
    are deliberately unused. A holdout used for stopping is validation, not an
    untouched final test set.
    """
    obs = np.asarray(observed)
    if obs.ndim not in (2, 3):
        raise ValueError("data must have shape (tx, rx) or (frequency, tx, rx)")
    active = np.isfinite(obs)
    if valid_mask is not None:
        active &= _broadcast(valid_mask, obs.shape, dtype=bool, name="valid_mask")
    weight = _broadcast(
        1.0 if weights is None else weights, obs.shape, dtype=float, name="weights"
    )
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("weights must be finite and nonnegative")
    active &= weight > 0
    if not np.any(active):
        raise ValueError("no valid, positive-weight observations")
    if not isinstance(exclude_reciprocal, (bool, np.bool_)):
        raise ValueError("exclude_reciprocal must be a boolean")
    rng = np.random.default_rng(seed)
    receiver_candidates = np.flatnonzero(
        np.any(active.reshape(-1, obs.shape[-1]), axis=0)
    )
    receivers = _indices(
        receiver_indices, receiver_fraction, receiver_candidates, rng, "receiver"
    )
    frequency_candidates = (
        np.flatnonzero(np.any(active.reshape(obs.shape[0], -1), axis=1))
        if obs.ndim == 3
        else np.array([], dtype=int)
    )
    frequencies = _indices(
        frequency_indices, frequency_fraction, frequency_candidates, rng, "frequency"
    )
    rx_flag = _broadcast(
        np.isin(np.arange(obs.shape[-1]), receivers),
        obs.shape,
        dtype=bool,
        name="receivers",
    )
    freq_flag = np.zeros(obs.shape, dtype=bool)
    if obs.ndim == 3:
        freq_flag[frequencies] = True
    reverse_flag = np.zeros(obs.shape, dtype=bool)
    excluded_tx = np.array([], dtype=int)
    if exclude_reciprocal and receivers.size:
        if tx_positions is None or rx_positions is None:
            raise ValueError(
                "receiver holdout requires positions, or explicit exclude_reciprocal=False"
            )
        tx, rx = np.asarray(tx_positions), np.asarray(rx_positions)
        if tx.shape != (obs.shape[-2], 2) or rx.shape != (obs.shape[-1], 2):
            raise ValueError("transducer positions must match the measurement shape")
        if not np.all(np.isfinite(tx)) or not np.all(np.isfinite(rx)):
            raise ValueError("transducer positions must be finite")
        excluded_tx = np.flatnonzero(
            np.any(
                np.all(
                    np.isclose(tx[:, None], rx[receivers][None], rtol=0, atol=1e-12),
                    axis=-1,
                ),
                axis=-1,
            )
        )
        reverse_flag[..., excluded_tx, :] = True
    masks = {
        "train": active & ~rx_flag & ~freq_flag & ~reverse_flag,
        "receiver": active & rx_flag & ~freq_flag,
        "frequency": active & freq_flag & ~rx_flag & ~reverse_flag,
        "joint": active & rx_flag & freq_flag,
        "unused": active & reverse_flag & ~rx_flag,
    }
    if not np.any(masks["train"]):
        raise ValueError("holdout leaves no training observations")
    digest = hashlib.sha256(str(obs.shape).encode())
    for value in masks.values():
        digest.update(np.packbits(value).tobytes())
        value.setflags(write=False)
    weight.setflags(write=False)
    metadata = {
        "seed": int(seed),
        "receiver_indices": receivers.tolist(),
        "frequency_indices": frequencies.tolist(),
        "reciprocal_tx_excluded": excluded_tx.tolist(),
        "mask_sha256": digest.hexdigest(),
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "invalid_or_zero_weight_count": int(obs.size - active.sum()),
        "holdout_role": "validation_not_final_test",
    }
    return DataSplit(**masks, weights=weight, metadata=metadata)
