"""Bounded direct-arrival delay from pressure and independently measured water."""

import numpy as np
from scipy.signal import correlate, correlation_lags, hilbert


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
    picker="xcorr",
    envelope_fraction=0.1,
    minimum_peak_snr=5.0,
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
        or not np.all(np.isfinite([lo, hi, reference_speed_mps, pulse_duration_s]))
        or pulse_duration_s <= 0
        or not 0 < minimum_correlation <= 1
    ):
        raise ValueError("invalid speed bounds, pulse duration or minimum correlation")
    if picker not in {"xcorr", "envelope"}:
        raise ValueError("picker must be xcorr or envelope")
    if (
        not 0 < envelope_fraction < 1
        or not np.isfinite(minimum_peak_snr)
        or minimum_peak_snr <= 0
    ):
        raise ValueError("envelope fraction must be in (0,1) and peak SNR positive")
    delay = np.full(distance.shape, np.nan)
    quality = np.zeros(distance.shape)
    for pair in np.ndindex(distance.shape):
        d = distance[pair]
        if not np.isfinite(d) or d <= 0:
            continue
        if picker == "envelope":
            bounds = (d / hi - 0.25 * pulse_duration_s, d / lo + pulse_duration_s)
            water_bounds = (
                d / reference_speed_mps - 0.25 * pulse_duration_s,
                d / reference_speed_mps + pulse_duration_s,
            )
            object_pick = _envelope_onset(
                p[:, pair[0], pair[1]], t, bounds, envelope_fraction, minimum_peak_snr
            )
            water_pick = _envelope_onset(
                w[:, pair[0], pair[1]],
                t,
                water_bounds,
                envelope_fraction,
                minimum_peak_snr,
            )
            if object_pick is not None and water_pick is not None:
                value = object_pick[0] - water_pick[0]
                if (
                    d * (1 / hi - 1 / reference_speed_mps)
                    <= value
                    <= d * (1 / lo - 1 / reference_speed_mps)
                ):
                    delay[pair] = value
                    quality[pair] = min(object_pick[1], water_pick[1])
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
            "method": (
                "bounded_normalized_water_direct_pulse_xcorr"
                if picker == "xcorr"
                else "water_relative_hilbert_envelope_onset"
            ),
            "valid_fraction": float(valid.sum() / max(nonself.sum(), 1)),
            "minimum_correlation": minimum_correlation if picker == "xcorr" else None,
            "speed_bounds_mps": list(speed_bounds),
            "pulse_duration_s": pulse_duration_s,
            "mean_valid_confidence": (
                float(quality[valid].mean()) if np.any(valid) else None
            ),
            "mean_valid_correlation": (
                float(quality[valid].mean())
                if picker == "xcorr" and np.any(valid)
                else None
            ),
            "envelope_fraction": envelope_fraction if picker == "envelope" else None,
            "minimum_peak_snr": minimum_peak_snr if picker == "envelope" else None,
            "limitation": "finite-band pulse onset/group delay, not a certified causal first arrival",
            "confidence_definition": (
                "normalized pulse correlation"
                if picker == "xcorr"
                else "one minus pre-window envelope RMS / peak; signal quality, not arrival accuracy"
            ),
            "ground_truth_used": False,
        },
    )


def _envelope_onset(trace, time, bounds, fraction, minimum_snr):
    """Subsample first threshold crossing before the direct-window peak.

    Hilbert envelopes can pre-ring. A non-water waveform distortion can change
    this threshold onset too; the output is a candidate feature, not an oracle.
    """
    if not np.all(np.isfinite(trace)):
        return None
    envelope = np.abs(hilbert(trace))
    window = np.flatnonzero((time >= bounds[0]) & (time <= bounds[1]))
    if window.size < 5:
        return None
    peak_index = window[np.argmax(envelope[window])]
    peak = envelope[peak_index]
    before = envelope[time < bounds[0]]
    noise = float(np.sqrt(np.mean(before**2))) if before.size else 0.0
    if not peak > max(np.finfo(float).tiny, minimum_snr * noise):
        return None
    threshold = fraction * peak
    candidates = window[(window <= peak_index) & (envelope[window] >= threshold)]
    if not candidates.size or candidates[0] == window[0]:
        return None
    index = candidates[0]
    a, b = envelope[index - 1], envelope[index]
    arrival = time[index - 1] + (threshold - a) / (b - a) * (
        time[index] - time[index - 1]
    )
    confidence = float(np.clip(1 - noise / peak, 0, 1))
    return float(arrival), confidence
