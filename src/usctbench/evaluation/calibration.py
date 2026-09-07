"""Independent-water source calibration using training observations only.

The fitted source spectrum is shared across receivers for each frequency and
transmitter. Unobserved source/frequency groups require external calibration;
validation pressure is never used to fill missing calibration groups.
"""

from __future__ import annotations

import numpy as np


def fit_source_spectrum(
    unit_source_prediction: np.ndarray,
    water_pressure: np.ndarray,
    train_mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Fit q[f,tx] minimizing sum_rx w |q G - p_water|**2 on training data.

    Arrays use (frequency, transmitter, receiver) order. Weights are nonnegative
    precisions, not their square roots. The returned spectrum must be used with
    the same pressure units, source convention, geometry and Fourier transform
    as the supplied unit-source prediction.
    """
    green = np.asarray(unit_source_prediction, dtype=complex)
    observed = np.asarray(water_pressure, dtype=complex)
    train = np.asarray(train_mask)
    if green.ndim != 3 or observed.shape != green.shape or train.shape != green.shape:
        raise ValueError("prediction, water pressure and train_mask must share (f,tx,rx) shape")
    if train.dtype.kind != "b":
        raise ValueError("train_mask must be boolean")
    precision = (
        np.ones(green.shape, dtype=float)
        if weights is None
        else np.broadcast_to(np.asarray(weights, dtype=float), green.shape)
    )
    if not np.all(np.isfinite(precision)) or np.any(precision < 0):
        raise ValueError("weights must be finite nonnegative precisions")
    usable = train & np.isfinite(green) & np.isfinite(observed) & (precision > 0)
    safe_green = np.where(usable, green, 0.0)
    safe_observed = np.where(usable, observed, 0.0)
    safe_weights = np.where(usable, precision, 0.0)
    numerator = np.sum(safe_weights * np.conj(safe_green) * safe_observed, axis=-1)
    denominator = np.sum(safe_weights * np.abs(safe_green) ** 2, axis=-1)
    if np.any(denominator <= np.finfo(float).tiny):
        raise ValueError(
            "a frequency/transmitter group has no usable training water observations; "
            "provide independent source calibration, do not fit from held-out data"
        )
    source = numerator / denominator
    if not np.all(np.isfinite(source)) or np.any(np.abs(source) <= np.finfo(float).tiny):
        raise ValueError("source calibration is nonfinite or has insufficient excitation")
    return source
