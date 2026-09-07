"""Small array-based contracts shared by inverse solvers and physics models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from usctbench.core.schema import GridSpec


class LinearOperator(Protocol):
    """Image-to-data linear map with its Euclidean discrete adjoint.

    Real parameters and complex measurements use Re(<J x, y>) = <x, J* y>.
    Nonlinear forward models expose a linearization, not an 'inverse adjoint'.
    """

    grid: GridSpec

    def forward(self, image: np.ndarray) -> np.ndarray: ...

    def adjoint(self, data: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Linearization:
    value: np.ndarray
    jacobian: LinearOperator


def adjoint_error(
    operator: LinearOperator, image: np.ndarray, data: np.ndarray
) -> float:
    """Relative real-inner-product error, with no dimensional absolute floor."""
    left = float(np.vdot(operator.forward(image), data).real)
    right = float(np.vdot(image, operator.adjoint(data)).real)
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
