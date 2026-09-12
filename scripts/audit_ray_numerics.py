"""Reproducible ray-operator numerics and optional post-hoc data consistency.

--case reads GT only for a diagnostic forward prediction AFTER reconstruction.
It never supplies an initialization, fitted offset, mask, or corrected data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from usctbench.core.io import read_case_hdf5
from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.data.waveforms import pressure_spectrum
from usctbench.operators.eikonal import EikonalForward, fast_march
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.model_space import BilinearBasis, ReducedLinearOperator


def dot_error(op, image, data):
    lhs, rhs = np.vdot(op.forward(image), data), np.vdot(image, op.adjoint(data))
    return float(abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny))


def numerical_audit():
    rng = np.random.default_rng(20260908)
    grid = GridSpec(shape=(21, 23), spacing_m=(0.001, 0.0013))
    geometry = GeometrySpec(
        tx_pos_m=[[-0.001, 0.003], [0.005, -0.001]],
        rx_pos_m=[[0.023, 0.019], [0.010, 0.032]],
    )
    straight = StraightRayProjector.from_grid_geometry(grid, geometry)
    eikonal = EikonalForward(grid, geometry)
    basis = BilinearBasis(grid, (6, 7))
    model = (1 + 0.02 * rng.random(grid.shape)) / 1500
    linearization = eikonal.linearize(model)
    v = rng.normal(size=grid.shape) / 1500
    data = rng.normal(size=4)
    eps = 1e-6
    fd = (eikonal.forward(model + eps * v) - eikonal.forward(model - eps * v)) / (
        2 * eps
    )
    jv = linearization.jacobian.forward(v)
    coarse = rng.normal(size=basis.shape) / 1500
    result = {
        "straight_adjoint_relative_error": dot_error(straight, v, data),
        "eikonal_adjoint_relative_error": dot_error(linearization.jacobian, v, data),
        "eikonal_directional_derivative_relative_error": float(
            np.linalg.norm(fd - jv) / np.linalg.norm(jv)
        ),
        "reduced_straight_adjoint_relative_error": dot_error(
            ReducedLinearOperator(straight, basis), coarse, data
        ),
        "reduced_eikonal_adjoint_relative_error": dot_error(
            ReducedLinearOperator(linearization.jacobian, basis), coarse, data
        ),
    }
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    ).ravel()
    homogeneous = eikonal.forward(np.full(grid.shape, 1 / 1500))
    result["calibrated_water_max_error_s"] = float(
        np.max(np.abs(homogeneous - distance / 1500))
    )
    result["water_scope"] = (
        "calibrated homogeneous consistency, not independent validation of heterogeneous accuracy"
    )
    exact = minimize_scalar(
        lambda x: np.hypot(0.025, x) / 1450 + np.hypot(0.025, 0.07 - x) / 1750,
        bounds=(0, 0.07),
        method="bounded",
        options={"xatol": 1e-14},
    ).fun
    rows = []
    for n in (41, 81, 161):
        h = 0.1 / (n - 1)
        y = np.arange(n)[:, None] * h
        s = np.broadcast_to(np.where(y < 0.05, 1 / 1450, 1 / 1750), (n, n)).copy()
        tape = fast_march(s, (h, h), np.array([0.025, 0.015]) / h)
        receiver = tuple(np.rint(np.array([0.075, 0.085]) / h).astype(int))
        value = float(tape.times.reshape(n, n)[receiver])
        rows.append(
            {
                "grid_n": n,
                "spacing_m": h,
                "time_s": value,
                "absolute_error_s": abs(value - exact),
                "relative_error": abs(value - exact) / exact,
            }
        )
    result["snell_fermat_exact_s"] = exact
    result["snell_refinement"] = rows
    assert (
        rows[2]["absolute_error_s"]
        < rows[1]["absolute_error_s"]
        < rows[0]["absolute_error_s"]
    )
    for key in (
        "straight_adjoint_relative_error",
        "eikonal_adjoint_relative_error",
        "reduced_straight_adjoint_relative_error",
        "reduced_eikonal_adjoint_relative_error",
    ):
        assert result[key] < 1e-11, (key, result[key])
    assert result["eikonal_directional_derivative_relative_error"] < 3e-5
    return result


def case_diagnostic(path):
    case = read_case_hdf5(path)
    m = case.measurement
    result = {
        "case_id": case.case_id,
        "purpose": "post-hoc GT forward consistency only; no offset fitting or data correction",
    }
    if m.time_data is not None and m.water_reference_time is not None:
        # Chunk transmitters to keep complex conversion buffers bounded.
        pressure_errors = {}
        for name, raw, stored in [
            ("object", m.time_data, m.freq_data),
            ("water", m.water_reference_time, m.water_reference),
        ]:
            error2, norm2 = 0.0, 0.0
            for start in range(0, raw.shape[1], 4):
                actual = pressure_spectrum(
                    raw[:, start : start + 4, :], m.time_axis_s, m.frequencies_hz
                )
                expected = stored[:, start : start + 4, :]
                error2 += float(np.linalg.norm(actual - expected) ** 2)
                norm2 += float(np.linalg.norm(expected) ** 2)
            pressure_errors[name + "_dtft_relative_error"] = float(
                np.sqrt(error2 / norm2)
            )
        result["raw_pressure_check"] = pressure_errors
    else:
        result["raw_pressure_check"] = (
            "not available; no time-domain repicking performed"
        )
    m.time_data = m.water_reference_time = None
    truth = case.ground_truth.sound_speed_mps
    if truth is None:
        result["gt_forward"] = "not available"
        return result
    c0 = 1500.0
    valid = np.asarray(m.valid_mask) & np.isfinite(m.delta_tof_s)
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    straight = StraightRayProjector.from_case(case)
    eikonal = EikonalForward(case.grid, case.geometry)
    predictions = {
        "straight": straight.forward(1 / truth - 1 / c0).reshape(distance.shape),
        "eikonal": eikonal.forward(1 / truth).reshape(distance.shape) - distance / c0,
    }
    result["gt_forward"] = {}
    for name, prediction in predictions.items():
        residual = prediction[valid] - m.delta_tof_s[valid]
        result["gt_forward"][name] = {
            "unweighted_rms_error_us": float(np.sqrt(np.mean(residual**2)) * 1e6),
            "unweighted_bias_us": float(np.mean(residual) * 1e6),
            "relative_residual": float(
                np.linalg.norm(residual) / np.linalg.norm(m.delta_tof_s[valid])
            ),
            "valid_observations": int(valid.sum()),
        }
    result["interpretation_limit"] = (
        "misfit combines feature, geometric-model and discretization errors; it does not isolate one cause"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", type=Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError("use a fresh output path")
    result = numerical_audit()
    args.out.write_text(json.dumps(result, indent=2))
    if args.case is not None:
        result["posthoc_case_diagnostic"] = case_diagnostic(args.case)
        args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
