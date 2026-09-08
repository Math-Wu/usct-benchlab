"""Bounded direct-arrival delay from pressure and independently measured water."""

import numpy as np
from scipy.signal import correlate, correlation_lags


def water_relative_delays(
    pressure,
    water,
    time_s,
    distances_m,
    *,
    reference_speed_mps=1500.0,
    speed_bounds=(1300.0, 1700.0),
    pulse_duration_s=15e-6,
    minimum_correlation=0.7,
):
    """Return delay/valid/precision and picker QC; arrays use (time,tx,rx).

    Absolute windows use source onset at time zero and the specified pulse
    duration. This is a band-limited direct-pulse lag, not an exact first-arrival
    time. No GT or image-derived mask is read. Invalid delays remain NaN.
    """
    p, w, t = np.asarray(pressure), np.asarray(water), np.asarray(time_s)
    distance = np.asarray(distances_m)
    if (
        p.ndim != 3
        or p.shape != w.shape
        or p.shape[1:] != distance.shape
        or t.shape != (p.shape[0],)
    ):
        raise ValueError(
            "pressure/water require matching (time,tx,rx) and explicit time"
        )
    if not np.all(np.isfinite(t)) or t.size < 3:
        raise ValueError("time axis must be finite with at least 3 samples")
    dt = float(np.mean(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-5, atol=0):
        raise ValueError("sampling must be uniform and increasing")
    lo, hi = speed_bounds
    if (
        not 0 < lo <= reference_speed_mps <= hi
        or pulse_duration_s <= 0
        or not 0 < minimum_correlation <= 1
    ):
        raise ValueError("invalid speed bounds, pulse duration or minimum correlation")
    delay = np.full(distance.shape, np.nan)
    quality = np.zeros(distance.shape)
    for pair in np.ndindex(distance.shape):
        d = distance[pair]
        if not np.isfinite(d) or d <= 0:
            continue
        window = (t >= d / hi - pulse_duration_s) & (t <= d / lo + 2 * pulse_duration_s)
        x, y = p[(window,) + pair], w[(window,) + pair]
        if x.size < 5 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue
        x, y = x - x.mean(), y - y.mean()
        numerator = correlate(x, y, mode="full", method="fft")
        xx = correlate(x * x, np.ones_like(y), method="fft")
        yy = correlate(np.ones_like(x), y * y, method="fft")
        denominator = np.sqrt(np.maximum(xx, 0) * np.maximum(yy, 0))
        corr = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.max(denominator) * 1e-8,
        )
        lags = correlation_lags(len(x), len(y)) * dt
        minimum, maximum = d * (1 / hi - 1 / reference_speed_mps), d * (
            1 / lo - 1 / reference_speed_mps
        )
        physical = (lags >= minimum) & (lags <= maximum)
        if not np.any(physical):
            continue
        peak = int(np.argmax(np.where(physical, corr, -np.inf)))
        if corr[peak] < minimum_correlation:
            continue
        shift = 0.0
        if 0 < peak < len(corr) - 1:
            curvature = corr[peak - 1] - 2 * corr[peak] + corr[peak + 1]
            if curvature < 0:
                shift = float(
                    np.clip(
                        0.5 * (corr[peak - 1] - corr[peak + 1]) / curvature, -0.5, 0.5
                    )
                )
        value = lags[peak] + shift * dt
        if minimum <= value <= maximum:
            delay[pair] = value
            quality[pair] = np.clip(corr[peak], 0, 1)
    valid = np.isfinite(delay)
    nonself = np.isfinite(distance) & (distance > 0)
    return (
        delay,
        valid,
        quality**2,
        {
            "method": "bounded_normalized_water_direct_pulse_xcorr",
            "valid_fraction": float(valid.sum() / max(nonself.sum(), 1)),
            "minimum_correlation": minimum_correlation,
            "speed_bounds_mps": list(speed_bounds),
            "pulse_duration_s": pulse_duration_s,
            "mean_valid_correlation": (
                float(quality[valid].mean()) if np.any(valid) else None
            ),
            "ground_truth_used": False,
        },
    )
