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
    source_onset_s=0.0,
    aic_envelope_fraction=0.25,
    aic_window_s=None,
):
    """Return delay/valid/precision and picker QC; arrays use (time,tx,rx).

    Absolute windows use the explicit source onset and the specified pulse
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
        or not np.isfinite(source_onset_s)
        or not 0 < minimum_correlation <= 1
    ):
        raise ValueError("invalid speed bounds, pulse duration or minimum correlation")
    if picker not in {"xcorr", "envelope", "aic"}:
        raise ValueError("picker must be xcorr, envelope or aic")
    aic_window_s = pulse_duration_s if aic_window_s is None else float(aic_window_s)
    if (
        not 0 < envelope_fraction < 1
        or not np.isfinite(minimum_peak_snr)
        or minimum_peak_snr <= 0
        or not 0 < aic_envelope_fraction < 1
        or not np.isfinite(aic_window_s)
        or aic_window_s <= 0
    ):
        raise ValueError("envelope fraction must be in (0,1) and peak SNR positive")
    delay = np.full(distance.shape, np.nan)
    quality = np.zeros(distance.shape)
    for pair in np.ndindex(distance.shape):
        d = distance[pair]
        if not np.isfinite(d) or d <= 0:
            continue
        if picker in {"envelope", "aic"}:
            bounds = (
                source_onset_s + d / hi - 0.25 * pulse_duration_s,
                source_onset_s + d / lo + pulse_duration_s,
            )
            water_bounds = (
                source_onset_s + d / reference_speed_mps - 0.25 * pulse_duration_s,
                source_onset_s + d / reference_speed_mps + pulse_duration_s,
            )
            if picker == "aic":
                pick = _aic_onset
                options = (aic_envelope_fraction, minimum_peak_snr, aic_window_s)
            else:
                pick = _envelope_onset
                options = (envelope_fraction, minimum_peak_snr)
            object_pick = pick(p[:, pair[0], pair[1]], t, bounds, *options)
            water_pick = pick(w[:, pair[0], pair[1]], t, water_bounds, *options)
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
        window = (t >= source_onset_s + d / hi - pulse_duration_s) & (
            t <= source_onset_s + d / lo + 2 * pulse_duration_s
        )
        # float32 pressure is preserved on disk; local correlation accumulation
        # needs float64 to avoid a spurious water-vs-itself subsample lag.
        x, y = p[(window,) + pair].astype(float), w[(window,) + pair].astype(float)
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
            "method": {
                "xcorr": "bounded_normalized_water_direct_pulse_xcorr",
                "envelope": "water_relative_hilbert_envelope_onset",
                "aic": "water_relative_modified_aic_onset",
            }[picker],
            "valid_fraction": float(valid.sum() / max(nonself.sum(), 1)),
            "minimum_correlation": minimum_correlation if picker == "xcorr" else None,
            "speed_bounds_mps": list(speed_bounds),
            "pulse_duration_s": pulse_duration_s,
            "source_onset_s": float(source_onset_s),
            "aic_window_s": aic_window_s if picker == "aic" else None,
            "aic_envelope_fraction": aic_envelope_fraction if picker == "aic" else None,
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


def arrival_observation_metadata(qc):
    """Explicit adapter contract; correlation quality is not timing uncertainty."""
    methods = {
        "bounded_normalized_water_direct_pulse_xcorr": "finite_band_correlation_delay",
        "water_relative_hilbert_envelope_onset": "finite_band_envelope_onset_difference",
        "water_relative_modified_aic_onset": "finite_band_aic_onset_difference",
    }
    if qc.get("method") not in methods:
        raise ValueError("unsupported arrival method for observation contract")
    return {
        "feature_source": qc["method"],
        "picker_method": qc["method"],
        "observable_definition": methods[qc["method"]],
        "feature_channel": "delta_tof_s",
        "tof_observation_contract": {
            "version": 1,
            "observable": methods[qc["method"]],
            "units": "s",
            "sign": "object_minus_water",
            "reference": "independent_water_pressure_same_acquisition",
            "source_onset_s": float(qc.get("source_onset_s", 0.0)),
            "pulse_duration_s": qc["pulse_duration_s"],
            "speed_bounds_mps": qc["speed_bounds_mps"],
            "weight_meaning": "squared_picker_confidence_not_inverse_noise_variance",
            "geometric_first_arrival_certified": False,
            "model_discrepancy_assessed": False,
            "reference_model": "model(object)-model(water), same numerical geometry",
            "ground_truth_used_in_extraction": False,
        },
    }


def _aic_onset(trace, time, bounds, fraction, minimum_snr, window_s):
    """Envelope-guided variance change point, inspired by Javaherian 2020 App. A.

    A small window ends at the first envelope crossing. Two Gaussian segments
    use population variances with a relative numerical floor; splits need three
    samples on each side. Average times in a quarter-window neighbourhood of
    the minimum using normalized exp(-delta_AIC/2) weights. Boundary minima are
    invalid. This explicit discretization is not an upstream code reproduction.
    Posterior concentration is NOT measurement uncertainty; finite-band tails
    and waveform distortion can move the inferred change point.
    """
    onset = _envelope_onset(trace, time, bounds, fraction, minimum_snr)
    if onset is None or onset[0] - window_s < time[0]:
        return None
    indices = np.flatnonzero((time >= onset[0] - window_s) & (time <= onset[0]))
    if indices.size < 12:
        return None
    signal = np.asarray(trace[indices], dtype=float)
    scale = np.max(np.abs(signal))
    if not scale > 0:
        return None
    signal = signal / scale
    signal -= signal.mean()
    n = signal.size
    splits = np.arange(3, n - 2)
    sums = np.r_[0.0, np.cumsum(signal)]
    squares = np.r_[0.0, np.cumsum(signal**2)]
    left = squares[splits] / splits - (sums[splits] / splits) ** 2
    right_n = n - splits
    right = (squares[-1] - squares[splits]) / right_n - (
        (sums[-1] - sums[splits]) / right_n
    ) ** 2
    floor = max(np.var(signal) * 1e-14, np.finfo(float).tiny)
    aic = splits * np.log(np.maximum(left, floor)) + right_n * np.log(
        np.maximum(right, floor)
    )
    best = int(np.argmin(aic))
    if best in (0, len(aic) - 1):
        return None
    half = max(1, int(round(n * 0.25)) // 2)
    near = np.arange(max(0, best - half), min(len(aic), best + half + 1))
    weights = np.exp(-0.5 * (aic[near] - aic[best]))
    weights /= weights.sum()
    # A split lies between the final noise sample and first signal sample.
    times = (time[indices[splits[near] - 1]] + time[indices[splits[near]]]) / 2
    return float(np.dot(weights, times)), onset[1]
