"""Deterministic operator checks, not an image-quality or clinical certificate.

Run with PYTHONPATH=src python scripts/verify_numerical_contracts.py --out FILE.
The finite differences keep the Eikonal active parents fixed. The Born remainder
uses an independently assembled dense multiple-scattering system on the same
cell discretization; it does not validate the continuum approximation in tissue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import hankel1

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators import adjoint_error
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.ray_born import RayBornForward, RayBornOperator


def measure_contracts():
    rng = np.random.default_rng(83)
    report = {}
    grid = GridSpec(shape=(3, 3), spacing_m=(1, 1))
    geom = GeometrySpec(tx_pos_m=[[0.5, -1], [2.5, -1]], rx_pos_m=[[0.5, 4], [2.5, 4]])
    x = rng.normal(size=grid.shape)
    y = rng.normal(size=4)
    projectors = {
        name: StraightRayProjector.from_grid_geometry(grid, geom, backend=name)
        for name in ("reference", "csr")
    }
    report["straight"] = {
        name: {
            "adjoint_error": adjoint_error(op, x, y),
            "explicit_forward_max_abs": float(
                np.max(np.abs(op.forward(x) - op.matrix @ x.ravel()))
            ),
            "explicit_adjoint_max_abs": float(
                np.max(np.abs(op.adjoint(y).ravel() - op.matrix.T @ y))
            ),
            "horizontal_integral_error": float(
                abs(op.forward(np.ones(grid.shape))[0] - 3)
            ),
        }
        for name, op in projectors.items()
    }
    grid = GridSpec(shape=(11, 13), spacing_m=(0.001, 0.0008))
    geom = GeometrySpec(
        tx_pos_m=[[-0.001, 0.0012], [0.0032, -0.002]],
        rx_pos_m=[[0.0092, 0.01], [0.011, 0.0082]],
    )
    m = (1 + 0.02 * rng.random(grid.shape)) / 1500
    dm = rng.normal(size=grid.shape) / 1500 * 0.005
    data = rng.normal(size=4)
    epsilons = [0.01, 0.003, 0.001, 0.0003, 0.0001]
    report["eikonal"] = {}
    for order in (1, 2):
        f = EikonalForward(grid, geom, spatial_order=order)
        lin = f.linearize(m)
        jv = lin.jacobian.forward(dm)
        rows = []
        for eps in epsilons:
            perturbed = f.linearize(m + eps * dm)
            rows.append(
                {
                    "epsilon": eps,
                    "relative_forward_difference_error": float(
                        np.linalg.norm((perturbed.value - lin.value) / eps - jv)
                        / np.linalg.norm(jv)
                    ),
                    "same_active_parents": all(
                        np.array_equal(a.parents, b.parents)
                        for a, b in zip(lin.jacobian.tapes, perturbed.jacobian.tapes)
                    ),
                }
            )
        report["eikonal"][str(order)] = {
            "adjoint_error": adjoint_error(lin.jacobian, dm, data),
            "directional_difference": rows,
        }
    grid = GridSpec(shape=(6, 7), spacing_m=(0.001, 0.001), origin_m=(-0.003, -0.0035))
    geom = GeometrySpec(
        tx_pos_m=[[-0.009, -0.008], [0.009, -0.003]],
        rx_pos_m=[[0.005, 0.009], [-0.006, 0.007]],
    )
    source = np.array([[1 + 0.2j, 0.7j], [0.8, -0.4j]])
    op = RayBornOperator(grid, geom, [180e3, 240e3], source_spectrum=source)
    dm = rng.normal(size=grid.shape) * 1e-9
    data = rng.normal(size=op.data_shape) + 1j * rng.normal(size=op.data_shape)
    report["born"] = {
        "parameter": "squared_slowness_s2_per_m2",
        "inner_product": "Re(vdot(J dm, r)) = vdot(dm, J* r)",
        "adjoint_error": adjoint_error(op, dm, data),
    }
    omega = 2 * np.pi * op.frequencies_hz[0]
    points = np.indices(grid.shape).reshape(2, -1).T
    points = np.array(grid.origin_m) + (points + 0.5) * grid.spacing_m
    radius = np.maximum(
        np.linalg.norm(points[:, None] - points[None], axis=-1), op.source_radius_m
    )
    gpp = 0.25j * hankel1(0, omega / 1500 * radius)
    gs = (
        0.25j
        * hankel1(0, omega / 1500 * np.linalg.norm(points - geom.tx_pos_m[0], axis=-1))
        * source[0, 0]
    )
    gr = 0.25j * hankel1(
        0, omega / 1500 * np.linalg.norm(geom.rx_pos_m[:, None] - points[None], axis=-1)
    )
    errors = []
    for scale in [1, 0.5, 0.25, 0.125]:
        delta = np.full(grid.shape, 2e-10 * scale)
        potential = omega**2 * op.area * delta.ravel()
        total = np.linalg.solve(np.eye(op.n_pixels) - gpp * potential, gs)
        scattered = gr @ (potential * total)
        errors.append(float(np.linalg.norm(scattered - op.forward(delta)[0, 0])))
    report["born"].update(
        remainder_norms=errors,
        remainder_orders=np.log2(np.array(errors[:-1]) / errors[1:]).tolist(),
    )
    f = RayBornForward(
        grid,
        geom,
        op.frequencies_hz,
        source_spectrum=source,
        green_backend="volume_integral",
        green_solver_rtol=1e-12,
    )
    m = np.full(grid.shape, 1 / 1500**2)
    lin = f.linearize(m)
    jv = lin.jacobian.forward(dm)
    rows = []
    for eps in [0.1, 0.05, 0.025]:
        numeric = (f.forward(m + eps * dm) - f.forward(m - eps * dm)) / (2 * eps)
        rows.append(
            {
                "epsilon": eps,
                "relative_central_difference_error": float(
                    np.linalg.norm(numeric - jv) / np.linalg.norm(jv)
                ),
            }
        )
    report["volume_integral"] = {
        "derivative_kind": lin.derivative_kind,
        "adjoint_error": adjoint_error(lin.jacobian, dm, data),
        "directional_difference": rows,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(measure_contracts(), indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
