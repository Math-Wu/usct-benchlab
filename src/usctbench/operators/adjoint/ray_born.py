"""Real-model adjoint of the complex-pressure ray-Born perturbation map."""

from __future__ import annotations

import numpy as np


def ray_born_adjoint(operator, data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=complex)
    if values.shape != operator.data_shape or not np.all(np.isfinite(values)):
        raise ValueError("adjoint data must be finite with shape (frequency, tx, rx)")
    result = np.zeros(operator.n_pixels)
    for index, frequency in enumerate(operator.frequencies_hz):
        green = operator.green_fields(index)
        weighted = values[index] * operator.source_spectrum[index, :, None].conj()
        receiver_field = weighted @ green[operator.rx_ids].conj()
        result += (
            np.sum(green[operator.tx_ids].conj() * receiver_field, axis=0).real
            * (2 * np.pi * frequency) ** 2
            * operator.area
        )
    return result.reshape(operator.grid.shape)
