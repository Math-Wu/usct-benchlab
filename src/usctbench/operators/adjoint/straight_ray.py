"""Exact transpose of the straight-ray cell-intersection projector."""

from __future__ import annotations

import numpy as np


def backproject(projector, ray_values: np.ndarray) -> np.ndarray:
    """Use the identical intersection lengths; never retrace an approximate ray."""
    values = np.asarray(ray_values, dtype=float).reshape(-1)
    if values.size != projector.n_rays:
        raise ValueError(
            f"ray_values has {values.size} entries, expected {projector.n_rays}"
        )
    flat = np.zeros(projector.n_pixels, dtype=float)
    for value, indices, lengths in zip(
        values, projector.indices_by_ray, projector.lengths_by_ray_m, strict=True
    ):
        if indices.size:
            np.add.at(flat, indices, value * lengths)
    return flat.reshape(projector.grid.shape)
