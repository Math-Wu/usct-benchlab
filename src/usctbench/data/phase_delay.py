"""Reference-relative multi-frequency delay for training-only initialization."""

import numpy as np


def phase_slope_delays(
    observed,
    reference,
    frequencies,
    distances,
    train_mask,
    *,
    c0=1500.0,
    bounds=(1300.0, 1700.0),
    max_phase_rms=0.2,
    min_amplitude_ratio=0.05,
):
    """Fit phase intercept + omega*delay using at least three TRAIN frequencies.

    This estimates a group delay, not an exact first arrival. A broadband ToF
    picker must not initialize a pressure inversion with held-out frequencies.
    Discard weak reference/object signals, nonlinear phase and out-of-bound fits.
    The fit is unchanged if held-out pressure values are modified arbitrarily.
    """
    p, water = np.asarray(observed), np.asarray(reference)
    frequencies = np.asarray(frequencies, dtype=float)
    train = np.broadcast_to(np.asarray(train_mask, bool), p.shape)
    if p.ndim != 3 or p.shape != water.shape or frequencies.shape != (p.shape[0],):
        raise ValueError("phase delays require matching (frequency,tx,rx) arrays")
    if (
        not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or not 0 < bounds[0] <= c0 <= bounds[1]
        or not np.isfinite(max_phase_rms)
        or max_phase_rms <= 0
        or not np.isfinite(min_amplitude_ratio)
        or min_amplitude_ratio <= 0
    ):
        raise ValueError("invalid frequencies, speed bounds or phase fit tolerance")
    indices = np.flatnonzero(np.any(train, axis=(1, 2)))
    indices = indices[np.argsort(frequencies[indices])]
    if len(indices) < 3 or np.any(np.diff(frequencies[indices]) <= 0):
        raise ValueError(
            "phase initialization requires at least three distinct training frequencies"
        )
    # Index before arithmetic: held-out NaNs and phases cannot affect fitting.
    p, water, train = p[indices], water[indices], train[indices]
    active = np.all(train & np.isfinite(p) & np.isfinite(water), axis=0)
    water_amplitude = np.abs(water)
    active &= np.all(water_amplitude > np.finfo(float).tiny, axis=0)
    active &= np.all(np.abs(p) > min_amplitude_ratio * water_amplitude, axis=0)
    x = 2 * np.pi * frequencies[indices]
    x -= x.mean()
    delay = np.full(p.shape[1:], np.nan)
    confidence = np.zeros(p.shape[1:])
    rms = np.full(p.shape[1:], np.nan)
    if np.any(active):
        phase = np.unwrap(np.angle(p[:, active] * water[:, active].conj()), axis=0)
        slope = np.sum(x[:, None] * phase, axis=0) / np.sum(x * x)
        residual = phase - phase.mean(axis=0) - x[:, None] * slope
        delay[active] = slope
        rms[active] = np.sqrt(np.mean(residual**2, axis=0))
    distance = np.broadcast_to(np.asarray(distances, float), delay.shape)
    active &= (distance > 0) & (rms <= max_phase_rms)
    active &= delay >= distance * (1 / bounds[1] - 1 / c0)
    active &= delay <= distance * (1 / bounds[0] - 1 / c0)
    confidence[active] = np.exp(-((rms[active] / max_phase_rms) ** 2))
    delay[~active] = np.nan
    return (
        delay,
        confidence,
        {
            "method": "training_only_reference_relative_phase_slope",
            "frequencies_hz": frequencies[indices].tolist(),
            "valid_fraction": float(active.sum() / max(np.all(train, axis=0).sum(), 1)),
            "phase_rms_median": float(np.median(rms[active])) if active.any() else None,
            "heldout_values_used": False,
            "max_phase_rms": float(max_phase_rms),
            "min_amplitude_ratio": float(min_amplitude_ratio),
            "limitation": "group-delay initialization; phase-unwrapping aliases and multipath remain possible",
        },
    )
