#!/usr/bin/env python3
"""Independent homogeneous k-Wave calibration; explicit GPU job per acquisition.

Reuse sampling, source and geometry, never breast GT. Calibrate on 1460/1540 m/s,
evaluate the frozen choice on 1480/1520 m/s. No per-breast delay correction.
Run --help. Outputs must live outside the repository and in a fresh directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.io import loadmat, savemat

from usctbench.data.arrival import water_relative_delays
from usctbench.core.schema import GridSpec, GeometrySpec
from usctbench.data.waveforms import _read_channels


def evaluate(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if (
        hashlib.sha256((out / "input.mat").read_bytes()).hexdigest()
        != manifest["input_sha256"]
    ):
        raise ValueError("calibration input changed")
    inputs = loadmat(out / "input.mat", simplify_cells=True)
    with h5py.File(out / "pressure.mat") as f:
        pressure = np.asarray(f["pressure"]).transpose(3, 2, 1, 0)
        time = np.asarray(f["time"]).ravel()
        speeds = np.asarray(f["speeds"]).ravel()
    if not np.isfinite(pressure).all():
        raise ValueError("nonfinite calibration pressure")
    positions = inputs["positions_yx"]
    tx = np.asarray(inputs["tx_indices"], dtype=int)
    distance = np.linalg.norm(positions[tx, None] - positions[None], axis=-1)
    active = distance > distance.max() * 0.5
    water = pressure[..., np.flatnonzero(speeds == 1500)[0]]
    with h5py.File(Path(manifest["acquisition"]) / "pressure.mat") as f:
        original_t = np.asarray(f["time"]).ravel()
        if not np.allclose(time, original_t, rtol=1e-12, atol=1e-15):
            raise ValueError("calibration changed sampling")
        original_water = _read_channels(
            f["water_dataset"],
            ("tx", "rx", "time"),
            np.asarray(manifest["tx_parent_indices"]),
            np.asarray(manifest["rx_parent_indices"]),
            len(time),
            2**30,
        )
    replay = float(
        np.linalg.norm(water.astype(float) - original_water)
        / np.linalg.norm(original_water)
    )
    if not np.isfinite(replay) or replay > 1e-5:
        raise ValueError("independent water replay does not match parent acquisition")
    candidates = {
        "xcorr": {"picker": "xcorr"},
        "aic": {"picker": "aic"},
        **{
            f"envelope_{f:g}": {"picker": "envelope", "envelope_fraction": f}
            for f in (0.02, 0.05, 0.1, 0.2)
        },
    }
    arrays = {"distance_m": distance, "active": active}
    expected_delays = {}
    if "bump_yx" in inputs:
        from audit_tof_adaptation import numerical_delays

        grid = GridSpec(
            shape=inputs["bump_yx"].shape,
            spacing_m=(float(inputs["dx"]),) * 2,
            origin_m=(-int(inputs["N"]) // 2 * inputs["dx"] - inputs["dx"] / 2,) * 2,
        )
        geometry = GeometrySpec(tx_pos_m=positions[tx], rx_pos_m=positions)
        for speed in speeds:
            field = 1500 + (speed - 1500) * inputs["bump_yx"]
            expected_delays[speed], _ = numerical_delays(field, grid, geometry, 2)
            arrays[f"eikonal_{int(speed)}_delay_s"] = expected_delays[speed]
    results = {}
    for name, options in candidates.items():
        results[name] = {}
        for k, speed in enumerate(speeds):
            delay, valid, _, _ = water_relative_delays(
                pressure[..., k],
                water,
                time,
                distance,
                pulse_duration_s=manifest["pulse_duration_s"],
                **options,
            )
            expected = (
                expected_delays[speed]
                if expected_delays
                else distance * (1 / speed - 1 / 1500)
            )
            use = active & valid
            error = (delay - expected)[use] * 1e9
            signal = expected[use] * 1e9
            results[name][str(int(speed))] = {
                "valid_fraction": float(use.sum() / active.sum()),
                "rms_ns": float(np.sqrt(np.mean(error**2))) if error.size else None,
                "bias_ns": float(error.mean()) if error.size else None,
                "reference_rms_ns": (
                    float(np.sqrt(np.mean(signal**2))) if signal.size else None
                ),
                "relative_model_error": (
                    float(np.linalg.norm(error) / np.linalg.norm(signal))
                    if np.linalg.norm(signal) > 0
                    else None
                ),
                "p95_abs_ns": (
                    float(np.quantile(np.abs(error), 0.95)) if error.size else None
                ),
            }
            arrays[f"{name}_{int(speed)}_delay_s"] = delay
    # Predeclared training-only selection; no access to breast errors or test speeds.
    scores = {}
    for name, values in results.items():
        training = [values[str(c)] for c in (1460, 1540)]
        if all(
            v["valid_fraction"] >= 0.95 and v["rms_ns"] is not None for v in training
        ):
            scores[name] = float(np.mean([v["rms_ns"] for v in training]))
    selected = min(scores, key=scores.get) if scores and not expected_delays else None
    summary = {
        "manifest": manifest,
        "results": results,
        "training_scores_ns": scores,
        "selected_on_training_only": selected,
        "validation_speeds_mps": [1480, 1520],
        "selection_scope": "homogeneous-medium timing fidelity, not anatomy/model validation",
        "reference_model": (
            "second_order_eikonal_not_exact"
            if expected_delays
            else "analytic_distance_over_speed"
        ),
        "water_replay_relative_residual": replay,
        "delay_correction_fitted": False,
        "pressure_sha256": hashlib.sha256(
            (out / "pressure.mat").read_bytes()
        ).hexdigest(),
    }
    np.savez_compressed(out / "diagnostics.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--kwave-path", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--medium-shape", choices=("uniform", "gaussian"), default="uniform"
    )
    parser.add_argument(
        "--gaussian-width-m",
        type=float,
        help="fixed independent phantom sigma; default 0.24 times ring radius",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if out.is_relative_to(repo):
        parser.error("outputs must be outside the repository")
    if args.evaluate_only:
        evaluate(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    original = loadmat(args.acquisition / "simulation_input.mat", simplify_cells=True)
    with h5py.File(args.acquisition / "pressure.mat") as f:
        time = np.asarray(f["time"]).ravel()
        signal = np.asarray(f["source_signal"]).ravel().astype(float)
    indices = np.arange(0, len(original["elements_yx"]), 2)
    elements = original["elements_yx"][indices]
    positions = original["positions_yx"][indices]
    values = {
        "N": original["sim_speed_yx"].shape[0],
        "dx": original["dx"],
        "pml": original["pml"],
        "Nt": time.size,
        "dt": np.diff(time).mean(),
        "elements_yx": elements,
        "positions_yx": positions,
        "tx_indices": np.array([0, len(indices) // 4]),
        "source_signal": signal,
        "speeds": np.array([1500, 1460, 1540, 1480, 1520]),
    }
    if args.medium_shape == "gaussian":
        n = values["N"]
        yy, xx = (np.indices((n, n)) - n // 2) * values["dx"]
        width = (
            args.gaussian_width_m
            if args.gaussian_width_m is not None
            else 0.24 * np.max(np.linalg.norm(positions, axis=1))
        )
        if not np.isfinite(width) or width < 4 * values["dx"]:
            raise ValueError(
                "Gaussian width must be finite and at least four simulation cells"
            )
        values["bump_yx"] = np.exp(-(yy**2 + xx**2) / (2 * width**2))
    savemat(out / "input.mat", values)
    manifest = {
        "acquisition": str(args.acquisition.resolve()),
        "input_sha256": hashlib.sha256((out / "input.mat").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
        "pulse_duration_s": 3 / original["source_frequency_hz"],
        "source_frequency_hz": float(original["source_frequency_hz"]),
        "training_speeds_mps": [1460, 1540],
        "test_speeds_mps": [1480, 1520],
        "cfl_max": float(1540 * values["dt"] / values["dx"]),
        "ppw_min": float(1460 / values["dx"] / original["maximum_frequency_hz"]),
        "tx_parent_indices": indices[values["tx_indices"]].tolist(),
        "rx_parent_indices": indices.tolist(),
        "device": args.device,
        "ground_truth_breast_used": False,
        "medium_shape": args.medium_shape,
        "gaussian_width_m": width if args.medium_shape == "gaussian" else None,
        "simulation_script_sha256": hashlib.sha256(
            (repo / "scripts/kwave_arrival_calibration.m").read_bytes()
        ).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def quote(path):
        return str(path).replace("'", "''")

    expression = (
        f"addpath('{quote(repo / 'scripts')}'); "
        f"kwave_arrival_calibration('{quote(out / 'input.mat')}',"
        f"'{quote(out / 'pressure.partial.mat')}','{quote(args.kwave_path)}',{args.device});"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}
    }
    with (out / "simulation.log").open("w") as log:
        subprocess.run(
            [args.matlab, "-batch", expression],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    (out / "pressure.partial.mat").rename(out / "pressure.mat")
    evaluate(out)


if __name__ == "__main__":
    main()
