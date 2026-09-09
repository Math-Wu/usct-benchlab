"""Optional CuPy acceleration of the SAME complex128 volume-integral model.

No CUDA dependency is imported on the default CPU path. CPU/GPU forward,
directional-derivative and adjoint agreement are required acceptance gates.
"""

from collections import OrderedDict
import inspect

import numpy as np


def solve_fields(solver, incident, device):
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator, gmres

    with cp.cuda.Device(device):
        potential = cp.asarray(solver.potential)
        kernel = cp.asarray(solver.kernel_fft)

        def matvec(values):
            if solver.budget_check is not None:
                solver.budget_check()
            solver.matvecs += 1
            field = values.reshape(solver.shape)
            transform = cp.fft.fft2(potential * field, s=solver.fft_shape)
            convolution = cp.fft.ifft2(kernel * transform)[
                : solver.shape[0], : solver.shape[1]
            ]
            return (field - convolution).ravel()

        n = int(np.prod(solver.shape))
        operator = LinearOperator((n, n), matvec=matvec, dtype=cp.complex128)
        result = np.empty_like(incident, dtype=complex)
        tolerance = {
            (
                "rtol" if "rtol" in inspect.signature(gmres).parameters else "tol"
            ): solver.rtol
        }
        restart = min(40, n)
        for index, rhs in enumerate(incident):
            b = cp.asarray(rhs, dtype=cp.complex128)
            value, info = gmres(
                operator,
                b,
                x0=b,
                atol=0,
                restart=restart,
                maxiter=restart * solver.maxiter,
                **tolerance,
            )
            residual = float(cp.linalg.norm(matvec(value) - b) / cp.linalg.norm(b))
            solver.solves += 1
            solver.maximum_relative_residual = max(
                solver.maximum_relative_residual, residual
            )
            if info != 0 or not np.isfinite(residual) or residual > solver.rtol * 1.01:
                raise FloatingPointError(
                    f"CUDA Green solve failed: info={info}, residual={residual}"
                )
            result[index] = cp.asnumpy(value)
        return result


class CudaBornJacobian:
    """GPU matrix products with bounded field cache; model/data API stays NumPy."""

    def __init__(self, operator, device):
        import cupy as cp

        self.operator, self.device, self.cp = operator, device, cp
        self.grid, self.data_shape = operator.grid, operator.data_shape
        self._cache = OrderedDict()

    def fields(self, k):
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        field = self.cp.asarray(self.operator.green_fields(k))
        if field.nbytes <= self.operator.max_cache_bytes:
            while (
                self._cache
                and sum(f.nbytes for f in self._cache.values()) + field.nbytes
                > self.operator.max_cache_bytes
            ):
                self._cache.popitem(last=False)
            self._cache[k] = field
        return field

    def forward(self, perturbation):
        dm = np.asarray(perturbation)
        if (
            dm.shape != self.grid.shape
            or np.iscomplexobj(dm)
            or not np.isfinite(dm).all()
        ):
            raise ValueError("require finite real model perturbation")
        cp, op = self.cp, self.operator
        with cp.cuda.Device(self.device):
            x = cp.asarray(dm.ravel())
            result = cp.empty(self.data_shape, dtype=cp.complex128)
            for k, frequency in enumerate(op.frequencies_hz):
                field = self.fields(k)
                result[k] = (
                    ((field[op.tx_ids] * x) @ field[op.rx_ids].T)
                    * ((2 * np.pi * frequency) ** 2 * op.area)
                    * cp.asarray(op.source_spectrum[k, :, None])
                )
            return cp.asnumpy(result)

    def adjoint(self, values):
        values = np.asarray(values, complex)
        if values.shape != self.data_shape or not np.isfinite(values).all():
            raise ValueError("require finite complex pressure sensitivity")
        cp, op = self.cp, self.operator
        with cp.cuda.Device(self.device):
            result = cp.zeros(op.n_pixels, dtype=cp.float64)
            for k, frequency in enumerate(op.frequencies_hz):
                field = self.fields(k)
                weighted = cp.asarray(values[k] * op.source_spectrum[k, :, None].conj())
                receiver = weighted @ field[op.rx_ids].conj()
                result += cp.sum(field[op.tx_ids].conj() * receiver, axis=0).real * (
                    (2 * np.pi * frequency) ** 2 * op.area
                )
            return cp.asnumpy(result).reshape(self.grid.shape)
