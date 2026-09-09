"""GT-free test of the translated-water-pulse assumption behind a delay.

A small residual supports a waveform translation approximation; it does not
certify a geometric first arrival. Diagnostics never change observations/masks.
"""

import numpy as np


def translated_reference_residual(
    pressure,
    water,
    time_s,
    delay_s,
    distances_m,
    *,
    speed_bounds=(1300.0, 1700.0),
    pulse_duration_s=15e-6,
    source_onset_s=0.0,
):
    """Fit a nonnegative scalar gain to delayed water on each direct window.

    All arrays are pair-local: (time, tx, rx), (tx, rx). No GT, other receiver
    or held-out frequency is consulted. The fitted gain is only a diagnostic,
    not a source calibration or an inverse-noise variance. Invalid/truncated
    windows return NaN, never an artificial zero residual.
    """
    p, w = np.asarray(pressure), np.asarray(water)
    t = np.asarray(time_s, dtype=float)
    delay, distance = np.asarray(delay_s), np.asarray(distances_m)
    if (
        p.ndim != 3
        or p.shape != w.shape
        or p.shape[1:] != delay.shape
        or distance.shape != delay.shape
        or t.shape != (p.shape[0],)
        or np.iscomplexobj(p)
        or np.iscomplexobj(w)
    ):
        raise ValueError(
            "require real (time,tx,rx) pressure and matching delay/geometry"
        )
    lo, hi = speed_bounds
    if (
        t.size < 3
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
        or not np.isfinite([lo, hi, pulse_duration_s, source_onset_s]).all()
        or not 0 < lo <= hi
        or pulse_duration_s <= 0
    ):
        raise ValueError("invalid time, speed bounds or pulse duration")
    residual = np.full(delay.shape, np.nan)
    gains = np.full(delay.shape, np.nan)
    for pair in np.ndindex(delay.shape):
        d, tau = distance[pair], delay[pair]
        if not np.isfinite([d, tau]).all() or d <= 0:
            continue
        lower = source_onset_s + d / hi - pulse_duration_s
        upper = source_onset_s + d / lo + 2 * pulse_duration_s
        if lower < t[0] or upper > t[-1]:
            continue
        mask = (t >= lower) & (t <= upper)
        times = t[mask] - tau
        x, y = p[(mask,) + pair].astype(float), w[(slice(None),) + pair].astype(float)
        if (
            x.size < 5
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or times[0] < t[0]
            or times[-1] > t[-1]
        ):
            continue
        ref = np.interp(times, t, y)
        # Normalize before products to make units irrelevant and avoid overflow.
        xs, ys = np.max(np.abs(x)), np.max(np.abs(ref))
        if xs <= 0 or ys <= 0:
            continue
        x, ref = x / xs, ref / ys
        gain = max(0.0, float(np.dot(x, ref) / np.dot(ref, ref)))
        gains[pair] = gain * (xs / ys)
        residual[pair] = np.linalg.norm(x - gain * ref) / np.linalg.norm(x)
    return residual, gains
