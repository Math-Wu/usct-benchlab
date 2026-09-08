"""Pair-local reference-relative phase travel times, not a wave inversion.

An existing independently picked pulse delay resolves the integer cycle only.
The returned delay is the phase divided by omega, not d(phase)/d(omega). These
observables can differ in a finite-band, scattering medium. Fitting either with
Siddon/Eikonal remains a geometric approximation, not finite-frequency physics.
"""

from __future__ import annotations

import numpy as np


def water_relative_phase_delays(
    pressure,
    water,
    frequency_hz,
    anchor_delay_s,
    distances_m,
    *,
    valid_mask=None,
    reference_speed_mps=1500.0,
    speed_bounds=(1300.0, 1700.0),
    min_amplitude_ratio=0.05,
):
    """Return (delay, valid, QC) from one explicit frequency's (tx,rx) data.

    For exp(-i omega t), delay = (arg(P/Pwater) + 2*pi*k)/(2*pi*f),
    where k is the integer nearest to f*anchor - arg(P/Pwater)/(2*pi).
    Correct unwrapping assumes the anchor is within half a period of the
    desired phase delay; proximity to a branch boundary is reported, not a
    certification that this assumption holds. No cross-channel smoothing,
    reciprocal averaging, source fitting, GT or validation-driven tuning occurs.
    Invalid observations remain NaN and never become a zero-delay observation.
    """
    p, w = np.asarray(pressure), np.asarray(water)
    anchor = np.asarray(anchor_delay_s, dtype=float)
    distance = np.asarray(distances_m, dtype=float)
    if p.ndim != 2 or any(a.shape != p.shape for a in (w, anchor, distance)):
        raise ValueError(
            "pressure/water/anchor/distances must have matching (tx,rx) shapes"
        )
    f, c0 = float(frequency_hz), float(reference_speed_mps)
    lo, hi = speed_bounds
    if (
        not np.isfinite([f, c0, lo, hi, min_amplitude_ratio]).all()
        or f <= 0
        or not 0 < lo <= c0 <= hi
        or min_amplitude_ratio < 0
    ):
        raise ValueError("invalid frequency, speed bounds or amplitude threshold")
    if valid_mask is None:
        requested = np.ones(p.shape, dtype=bool)
    else:
        requested = np.asarray(valid_mask)
        if requested.dtype.kind != "b" or requested.shape != p.shape:
            raise ValueError("valid_mask must be a matching boolean array")
    valid = requested & np.isfinite(p) & np.isfinite(w)
    valid &= np.isfinite(anchor) & np.isfinite(distance) & (distance > 0)
    amplitude_p, amplitude_w = np.abs(p), np.abs(w)
    valid &= np.isfinite(amplitude_p) & np.isfinite(amplitude_w)
    valid &= (amplitude_w > np.finfo(float).tiny) & (amplitude_p > np.finfo(float).tiny)
    valid &= amplitude_p >= min_amplitude_ratio * amplitude_w
    lower, upper = distance * (1 / hi - 1 / c0), distance * (1 / lo - 1 / c0)
    valid &= (anchor >= lower) & (anchor <= upper)
    delay = np.full(p.shape, np.nan)
    margin = np.full(p.shape, np.nan)
    cycles = np.zeros(p.shape, dtype=np.int64)
    if valid.any():
        # Subtract angles instead of multiplying complex values: avoid overflow
        # or underflow for pressure expressed in different amplitude units.
        phase = np.angle(np.exp(1j * (np.angle(p[valid]) - np.angle(w[valid]))))
        position = f * anchor[valid] - phase / (2 * np.pi)
        if not np.isfinite(position).all() or np.any(np.abs(position) >= 2**52):
            raise ValueError("frequency times anchor exceeds resolvable cycle range")
        nearest = np.rint(position)
        delay[valid] = (phase / (2 * np.pi) + nearest) / f
        margin[valid] = 0.5 - np.abs(position - nearest)
        cycles[valid] = nearest.astype(np.int64)
    valid &= (delay >= lower) & (delay <= upper) & (margin > 1e-12)
    delay[~valid] = np.nan
    return (
        delay,
        valid,
        {
            "method": "water_relative_single_frequency_phase",
            "frequency_hz": f,
            "anchor_role": "integer_cycle_only",
            "requested_channels": int(requested.sum()),
            "valid_channels": int(valid.sum()),
            "dropped_channels": int(np.count_nonzero(requested & ~valid)),
            "nonzero_cycle_channels": int(np.count_nonzero(cycles[valid])),
            "minimum_branch_margin_cycles": (
                float(margin[valid].min()) if valid.any() else None
            ),
            "min_amplitude_ratio": float(min_amplitude_ratio),
            "ground_truth_used": False,
            "cross_channel_fitting": False,
            "limitation": "phase delay remains finite-frequency; anchor cycle errors and multipath are not removed",
        },
    )
