"""Exact adjoint of the frozen-stencil external Helmholtz Jacobian."""

import numpy as np


def helmholtz_adjoint(jacobian, values):
    data = np.asarray(values, complex)
    if data.shape != jacobian.model.data_shape:
        raise ValueError("pressure cotangent must match (frequency,tx,rx)")
    result = np.zeros(np.prod(jacobian.grid.shape))
    for f, (factor, derivative, fields) in enumerate(jacobian.blocks):
        source = jacobian.model.rx.T @ data[f].conj().T
        adjoint_field = factor.solve(source, trans="H")
        result -= np.real(
            np.sum(fields.conj() * (derivative.conj().T @ adjoint_field), axis=1)
        )
    return result.reshape(jacobian.grid.shape)
