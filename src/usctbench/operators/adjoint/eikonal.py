"""Reverse accumulation of the causal fast-marching linearization."""

from __future__ import annotations

import numpy as np
from usctbench.operators._marching import reverse_accumulate


def eikonal_adjoint(jacobian, data: np.ndarray) -> np.ndarray:
    model = jacobian.model
    values = np.asarray(data, dtype=float)
    if values.size != model.n_rays:
        raise ValueError("data size must match (n_tx, n_rx)")
    if not np.all(np.isfinite(values)):
        raise ValueError("adjoint data must be finite")
    values = values.reshape(model.ray_shape)
    result = np.zeros(model.shape)
    for tape, receiver_values in zip(jacobian.tapes, values):
        sensitivity = np.zeros(tape.times.size)
        np.add.at(
            sensitivity,
            model.receiver_indices.ravel(),
            (receiver_values[:, None] * model.receiver_weights).ravel(),
        )
        gradient = reverse_accumulate(
            tape.order, tape.parents, tape.weights, tape.local_derivative, sensitivity
        )
        result += gradient.reshape(model.shape)
    return result[model.crop].copy()
