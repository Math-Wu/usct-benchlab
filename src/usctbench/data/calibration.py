"""Instrument source calibration from an independent water-only acquisition."""

from __future__ import annotations

import numpy as np

from usctbench.evaluation.data import residual_statistics


def fit_water_source(unit_water_prediction, water_reference, *, valid_mask=None):
    """Fit one complex source coefficient per (frequency, transmitter).

    Minimize sum_rx |q[f,tx] G_water[f,tx,rx] - p_water[f,tx,rx]|^2.
    This is a *multiplicative* calibration of both the forward field and its
    Jacobian, not an additive replacement of the background prediction.
    Only independently acquired reference data may be passed here. No specimen
    measurements or ground-truth labels are accepted. A source-only model does
    not identify arbitrary receiver responses or finite-aperture directivity.
    """
    unit = np.asarray(unit_water_prediction, dtype=complex)
    water = np.asarray(water_reference, dtype=complex)
    if unit.ndim != 3 or water.shape != unit.shape:
        raise ValueError(
            "water and unit predictions must have matching (frequency, tx, rx) shape"
        )
    active = np.isfinite(water)
    if valid_mask is not None:
        active &= np.broadcast_to(np.asarray(valid_mask, dtype=bool), water.shape)
    if not np.all(np.isfinite(unit[active])):
        raise ValueError("nonfinite unit-source prediction on calibration channels")
    u = np.where(active, unit, 0)
    w = np.where(active, water, 0)
    denominator = np.sum(np.abs(u) ** 2, axis=-1)
    if np.any(denominator <= np.finfo(float).tiny) or not np.all(
        np.isfinite(denominator)
    ):
        raise ValueError(
            "each frequency/transmitter needs a nonzero water calibration channel"
        )
    source = np.sum(u.conj() * w, axis=-1) / denominator
    if not np.all(np.isfinite(source)) or np.any(
        np.abs(source) <= np.finfo(float).tiny
    ):
        raise ValueError("water source calibration overflow or no excitation")
    statistics = residual_statistics(unit * source[..., None], water, mask=active)
    return source, {
        "method": "independent_water_complex_least_squares_per_frequency_transmitter",
        "measurement_scope": "independent_water_acquisition_not_specimen_holdout",
        "calibration_samples": int(active.sum()),
        "minimum_channels_per_source": int(active.sum(axis=-1).min()),
        "relative_residual": statistics["relative_residual"],
        "specimen_data_used": False,
        "heldout_frequencies_have_independent_instrument_calibration": True,
        "receiver_response_model": "not_separately_estimated",
    }
