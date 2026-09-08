"""Diagnostic error attribution, never a correction for measured travel times.

Only reciprocity_floor is truth-free. The other helpers operate on explicit
reference predictions/errors supplied by the caller. Oracle-generated controls
must not be labelled as physical acquisitions or deployed as GT-free algorithms.
"""

from __future__ import annotations

import numpy as np


def _active_arrays(*arrays, mask, weights):
    values = [np.asarray(a) for a in arrays]
    m = np.asarray(mask)
    w = np.asarray(weights, dtype=float)
    if m.dtype != bool or not values or any(a.shape != m.shape for a in values):
        raise ValueError(
            "a boolean mask and identically shaped real arrays are required"
        )
    if w.shape != m.shape or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be finite, nonnegative and match the data")
    active = m & (w > 0)
    if not active.any():
        raise ValueError("at least one positive-weight active observation is required")
    for a in values:
        if np.iscomplexobj(a) or not np.isfinite(a[active]).all():
            raise ValueError("active travel times must be finite real values")
    return values, active, w


def error_decomposition(straight, eikonal, observed, *, mask, weights):
    """Exact vector identity, with cross term; RMS differences are not causal shares.

    refraction_discrete = E_h(c) - A(c); observable_gap = P(p) - E_h(c).
    The latter includes feature, wave/ray and discretization mismatch together.
    It is NOT an estimate of picker error alone or continuum model error.
    """
    (a, e, b), active, w = _active_arrays(
        straight, eikonal, observed, mask=mask, weights=weights
    )
    w = w[active]
    r, g = (e - a)[active], (b - e)[active]

    def energy(x):
        return float(np.dot(w, x * x) / w.sum())

    er, eg, total = energy(r), energy(g), energy(r + g)
    cross = float(2 * np.dot(w, r * g) / w.sum())
    return {
        "count": int(active.sum()),
        "refraction_discrete_rms_s": np.sqrt(er),
        "observable_gap_rms_s": np.sqrt(eg),
        "total_straight_gap_rms_s": np.sqrt(total),
        "twice_cross_mean_s2": cross,
        "component_cosine": cross / (2 * np.sqrt(er * eg)) if er * eg > 0 else None,
        "identity_absolute_error_s2": abs(total - er - eg - cross),
        "oracle_diagnostic_only": True,
    }


def reciprocity_floor(observed, *, mask, weights):
    """Weighted lower bound for ANY exactly reciprocal ToF prediction.

    Requires identical TX/RX position ordering, verified by the caller. Missing
    reverse channels contribute zero to this lower bound, not a fake observation.
    The bound is conditional on reciprocal source/receiver timing/calibration;
    it says nothing about the physically correct speed or attainable image RMSE.
    """
    (b,), active, w = _active_arrays(observed, mask=mask, weights=weights)
    if b.ndim != 2 or b.shape[0] != b.shape[1]:
        raise ValueError("reciprocity requires a square, aligned TX/RX acquisition")
    i, j = np.where(np.triu(active & active.T, 1))
    wi, wj = w[i, j], w[j, i]
    difference = b[i, j] - b[j, i]
    sse = float(np.sum(wi * wj / (wi + wj) * difference**2))
    signal = float(np.sum(w[active] * b[active] ** 2))
    return {
        "reciprocal_pairs": len(i),
        "active_directed_channels": int(active.sum()),
        "minimum_weighted_sse_s2": sse,
        "minimum_weighted_rms_s": np.sqrt(sse / w[active].sum()),
        "minimum_relative_residual": np.sqrt(sse / signal) if signal > 0 else None,
        "uses_ground_truth": False,
    }


def shuffled_pair_error(error, *, mask, weights, groups, seed, shuffle_group_ids=None):
    """Remove spatial structure while preserving partition energy, mean and reciprocity.

    Only pairs whose two directions are in the SAME partition are shuffled.
    Cross-partition and one-sided observations stay exactly unchanged. Callers
    may restrict shuffle_group_ids (e.g. training only). Inside
    each partition, project a seeded permutation of whitened pair means off the
    constant mode, renormalize its energy, and restore the original weighted mean.
    The original pairwise antisymmetric error is unchanged. Thus this is a
    norm/mean-preserving decorrelation control, NOT iid Gaussian noise nor an
    exact histogram-preserving shuffle. No validation value enters training.
    """
    (err,), active, w = _active_arrays(error, mask=mask, weights=weights)
    g = np.asarray(groups)
    if err.ndim != 2 or err.shape[0] != err.shape[1] or g.shape != err.shape:
        raise ValueError("error and partition labels must match a square acquisition")
    if not np.issubdtype(g.dtype, np.integer):
        raise ValueError("partition labels must be integers")
    rng = np.random.default_rng(seed)
    out = err.astype(float, copy=True)
    matched = np.triu(active & active.T & (g == g.T), 1)
    changed = 0
    selected = (
        np.unique(g[active]) if shuffle_group_ids is None else tuple(shuffle_group_ids)
    )
    for group in selected:
        i, j = np.where(matched & (g == group))
        if len(i) < 3:
            continue
        wi, wj = w[i, j], w[j, i]
        q = np.sqrt(wi + wj)
        mu = (wi * err[i, j] + wj * err[j, i]) / (wi + wj)
        u = q * mu
        parallel = q * (q @ u) / (q @ q)
        orthogonal = u - parallel
        v = rng.permutation(orthogonal)
        v -= q * (q @ v) / (q @ q)
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        v *= np.linalg.norm(orthogonal) / norm
        replacement = (parallel + v) / q
        delta = replacement - mu
        out[i, j] += delta
        out[j, i] += delta
        changed += len(i)
    return out, {"shuffled_pairs": changed, "seed": int(seed)}
