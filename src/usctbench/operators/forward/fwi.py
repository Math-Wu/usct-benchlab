"""Live full-wave operators using the external WaveformInversionUST matrix.

No full-wave PDE is replaced by a travel-time projector. MATLAB constructs the
upstream optimized Helmholtz stencil; SciPy solves that exported sparse system.
This portable CPU reference bridge is intended for derivative validation and
agent integration, not a replacement for the production CUDA block-LU driver.
"""

from pathlib import Path
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu

from usctbench.operators.base import Linearization
from usctbench.operators.forward.eikonal import interpolation


class MatlabHelmholtzForward:
    """Squared slowness -> complex pressure; explicit fixed stencil bounds.

    Images/geometry use [y,x]. This operator requires transducers inside the
    supplied grid and outside its PML; it never silently resizes an acquisition.
    MATLAB's exp(-ikr) fields are conjugated at the public exp(-i omega t)
    boundary. `source_spectrum` is in discrete RHS units for this matrix, so
    analytic Green-function or other instruments' source factors are not reused.
    """

    def __init__(
        self,
        grid,
        geometry,
        frequencies_hz,
        *,
        functions_path,
        matlab="matlab",
        pml_m=0.009,
        pml_strength=10.0,
        stencil_speed_bounds=(1300.0, 1700.0),
        source_spectrum=None,
        timeout_s=180,
    ):
        self.grid, self.geometry = grid, geometry
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        if (
            self.frequencies_hz.ndim != 1
            or not np.all(np.isfinite(self.frequencies_hz))
            or np.any(self.frequencies_hz <= 0)
            or self.frequencies_hz.size == 0
        ):
            raise ValueError("frequencies must be a finite positive vector")
        self.functions_path = Path(functions_path).resolve()
        if not (self.functions_path / "HelmholtzSolver.m").exists():
            raise FileNotFoundError(
                "functions_path must contain upstream HelmholtzSolver.m"
            )
        self.matlab, self.timeout_s = matlab, float(timeout_s)
        self.pml_m, self.pml_strength = float(pml_m), float(pml_strength)
        self.stencil_speed_bounds = tuple(stencil_speed_bounds)
        if (
            not 0 < self.stencil_speed_bounds[0] <= self.stencil_speed_bounds[1]
            or pml_m <= 0
            or pml_strength <= 0
        ):
            raise ValueError("invalid PML or fixed stencil velocity bounds")
        positions = np.vstack([geometry.tx_pos_m, geometry.rx_pos_m])
        coordinates = (positions - grid.origin_m) / grid.spacing_m - 0.5
        lower = np.array(grid.origin_m) + pml_m
        upper = np.array(grid.origin_m) + np.array(grid.shape) * grid.spacing_m - pml_m
        if np.any(positions <= lower) or np.any(positions >= upper):
            raise ValueError(
                "transducers must lie inside the supplied grid, outside PML"
            )
        ids, weights = interpolation(grid.shape, coordinates)
        count = len(positions)
        self.sampling = coo_matrix(
            (weights.ravel(), (np.repeat(np.arange(count), 4), ids.ravel())),
            shape=(count, np.prod(grid.shape)),
        ).tocsr()
        self.n_tx = len(geometry.tx_pos_m)
        self.tx = self.sampling[: self.n_tx]
        self.rx = self.sampling[self.n_tx :]
        self.data_shape = (len(self.frequencies_hz), self.n_tx, len(geometry.rx_pos_m))
        self.source_spectrum = (
            np.ones(self.data_shape[:2], complex)
            if source_spectrum is None
            else np.broadcast_to(
                np.asarray(source_spectrum, complex), self.data_shape[:2]
            ).copy()
        )
        if not np.all(np.isfinite(self.source_spectrum)):
            raise ValueError("source spectrum must be finite")
        self.background_builds = self.eikonal_solves = 0

    def linearize(self, squared_slowness):
        model = np.asarray(squared_slowness, dtype=float)
        if (
            model.shape != self.grid.shape
            or np.any(model <= 0)
            or not np.all(np.isfinite(model))
        ):
            raise ValueError("model must be a finite positive squared-slowness image")
        with tempfile.TemporaryDirectory(prefix="usct_helmholtz_") as folder:
            root = Path(folder)
            savemat(
                root / "request.mat",
                {
                    "speed_yx": 1 / np.sqrt(model),
                    "x": self.grid.origin_m[1]
                    + (np.arange(self.grid.shape[1]) + 0.5) * self.grid.spacing_m[1],
                    "y": self.grid.origin_m[0]
                    + (np.arange(self.grid.shape[0]) + 0.5) * self.grid.spacing_m[0],
                    "frequencies": self.frequencies_hz,
                    "pml_m": self.pml_m,
                    "pml_strength": self.pml_strength,
                    "stencil_bounds": self.stencil_speed_bounds,
                },
            )

            def quote(x):
                return str(x).replace("'", "''")

            bridge = Path(__file__).with_name("matlab")
            expression = f"addpath('{quote(bridge)}'); export_wust_operator('{quote(root)}','{quote(self.functions_path)}');"
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in {"LD_LIBRARY_PATH", "LD_PRELOAD"}
            }
            # The CPU sparse export avoids triggering the production GPU LU
            # factorization a second time merely to export its matrix.
            env["CUDA_VISIBLE_DEVICES"] = ""
            result = subprocess.run(
                [self.matlab, "-batch", expression],
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_s,
            )
            if result.returncode:
                raise RuntimeError(
                    "external WUST matrix export failed: "
                    + (result.stderr + result.stdout)[-4000:]
                )
            blocks = []
            predictions = []
            for f, frequency in enumerate(self.frequencies_hz):
                data = loadmat(root / f"operator_{f+1}.mat")
                # MATLAB linearizes [y,x] in column-major order; public NumPy
                # operators flatten row-major. Explicitly permute both axes.
                order = (
                    np.arange(model.size).reshape(self.grid.shape, order="F").ravel()
                )
                matrix = data["H"].tocsr()[order][:, order].tocsc()
                derivative = data["B"].tocsr()[order][:, order] @ diags(
                    ((2 * np.pi * frequency) ** 2 * data["PML"].ravel())
                )
                factor = splu(matrix)
                sources = self.tx.T.toarray() * self.source_spectrum[f].conj()[None, :]
                fields = factor.solve(sources.astype(complex))
                predictions.append((self.rx @ fields).T.conj())
                blocks.append((factor, derivative, fields))
        self.background_builds += 1
        jacobian = HelmholtzJacobian(self, blocks)
        return Linearization(np.asarray(predictions), jacobian)

    def forward(self, squared_slowness):
        return self.linearize(squared_slowness).value


class HelmholtzJacobian:
    def __init__(self, forward, blocks):
        self.model, self.grid, self.blocks = forward, forward.grid, blocks

    def forward(self, perturbation):
        dm = np.asarray(perturbation, dtype=float)
        if dm.shape != self.grid.shape:
            raise ValueError("perturbation must match grid")
        outputs = []
        for factor, derivative, fields in self.blocks:
            field = -factor.solve(derivative @ (fields * dm.ravel()[:, None]))
            outputs.append((self.model.rx @ field).T.conj())
        return np.asarray(outputs)

    def adjoint(self, values):
        from usctbench.operators.adjoint.fwi import helmholtz_adjoint

        return helmholtz_adjoint(self, values)
