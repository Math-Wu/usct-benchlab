#!/usr/bin/env python3
"""Compare a fixed finite-frequency delay observable with its Born derivative.

Forward-at-GT attribution only. Never produces corrected breast observations or
selects reconstruction parameters. No new reconstruction algorithm is registered.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np

from usctbench.core.io import read_case_hdf5
from usctbench.core.schema import GeometrySpec
from usctbench.data.arrival import water_relative_delays
from usctbench.data.validation_acquisition import read_validation_acquisition
from usctbench.data.waveforms import _read_channels, pressure_spectrum
from usctbench.operators.forward.correlation_delay import (
    CorrelationDelayDerivative,
    CorrelationTravelTimeJacobian,
)
from usctbench.operators.ray_born import RayBornOperator, RayBornForward
from usctbench.operators.straight_ray import StraightRayProjector
from audit_tof_adaptation import stats, numerical_delays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tx-count", type=int, default=4)
    parser.add_argument("--full-green", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo) or not 1 <= args.tx_count <= 64:
        parser.error("outputs must be outside repo; tx-count in [1,64]")
    acquisition = read_validation_acquisition(args.acquisition)
    case = read_case_hdf5(args.case)
    if case.metadata.get("input_sha256") != acquisition.manifest["input_sha256"]:
        raise ValueError("different acquisition")
    if not np.array_equal(
        case.ground_truth.sound_speed_mps, acquisition.image_speed_mps
    ):
        raise ValueError("different property map")
    args.out.mkdir(parents=True, exist_ok=False)
    tx = np.arange(0, 128, 128 // args.tx_count)[: args.tx_count]
    rx = np.arange(0, 128, 2)
    geometry = GeometrySpec(
        tx_pos_m=acquisition.positions_yx_m[tx], rx_pos_m=acquisition.positions_yx_m[rx]
    )
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    active = distance > distance.max() * 0.5
    with h5py.File(args.acquisition / "pressure.mat") as f:
        t = np.asarray(f["time"]).ravel()
        p = _read_channels(
            f["full_dataset"], ("tx", "rx", "time"), tx, rx, len(t), 2**30
        )
        w = _read_channels(
            f["water_dataset"], ("tx", "rx", "time"), tx, rx, len(t), 2**30
        )
    fc = acquisition.manifest["source_frequency_hz"]
    frequencies = np.linspace(0.4 * fc, acquisition.manifest["maximum_frequency_hz"], 9)
    measured = pressure_spectrum(p, t, frequencies)
    water = pressure_spectrum(w, t, frequencies)
    obs = CorrelationDelayDerivative(frequencies, np.where(active[None], water, 0))
    ratio = np.divide(
        measured - water, water, out=np.zeros_like(water), where=obs.frequency_valid
    )
    data = obs.forward(ratio)
    born = RayBornOperator(
        case.grid, geometry, frequencies, max_cache_bytes=512 * 1024**2
    )
    op = CorrelationTravelTimeJacobian(born, obs)
    speed = acquisition.image_speed_mps
    start = time.perf_counter()
    prediction = op.forward(1 / speed**2 - 1 / 1500**2).reshape(distance.shape)
    arrays = {
        "observed_linearized_delay_s": data,
        "born_delay_s": prediction,
        "active": active,
        "observed_relative_scattered": ratio,
    }
    report = {
        "case_id": case.case_id,
        "acquisition_input_sha256": acquisition.manifest["input_sha256"],
        "frequencies_hz": frequencies.tolist(),
        "tx_parent_indices": tx.tolist(),
        "rx_parent_indices": rx.tolist(),
        "observable": "small_perturbation_water_weighted_correlation_delay_derivative",
        "time_window": "full_record_same_for_object_and_water",
        "source_response": "ratio_to_independent_water; no source fitted to breast",
        "noise_variance_claimed": False,
        "gt_used_to_correct_data": False,
        "gt_use": "posthoc_forward_attribution_only",
        "born_vs_observed_ns": stats((prediction - data)[active] * 1e9),
        "born_forward_elapsed_s": time.perf_counter() - start,
        "code_sha256": {
            str(x.relative_to(repo)): hashlib.sha256(x.read_bytes()).hexdigest()
            for x in (repo / "src/usctbench").rglob("*.py")
        },
    }
    straight = StraightRayProjector.from_grid_geometry(case.grid, geometry)
    arrays["straight_delay_s"] = straight.forward(1 / speed - 1 / 1500).reshape(
        distance.shape
    )
    arrays["eikonal_delay_s"], _ = numerical_delays(speed, case.grid, geometry, 2)
    for picker in ("xcorr", "envelope"):
        delay, valid, _, _ = water_relative_delays(
            p, w, t, distance, pulse_duration_s=3 / fc, picker=picker
        )
        arrays[picker + "_delay_s"] = delay
        report["linearized_vs_" + picker + "_ns"] = stats(
            (data - delay)[active & valid] * 1e9
        )
        for model in ("straight", "eikonal"):
            report[model + "_vs_" + picker + "_ns"] = stats(
                (arrays[model + "_delay_s"] - delay)[active & valid] * 1e9
            )
    for model in ("straight", "eikonal"):
        report[model + "_vs_observed_ns"] = stats(
            (arrays[model + "_delay_s"] - data)[active] * 1e9
        )

    def save():
        (args.out / "summary.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.out / "diagnostics.npz", **arrays)

    save()
    if args.full_green:
        full = RayBornForward(
            case.grid,
            geometry,
            frequencies,
            green_backend="volume_integral",
            green_solver_rtol=1e-6,
            green_solver_maxiter=30,
            max_cache_bytes=512 * 1024**2,
        )
        prediction_pressure = full.forward(1 / speed**2)
        relative = np.divide(
            prediction_pressure - op.reference,
            op.reference,
            out=np.zeros_like(prediction_pressure),
            where=obs.frequency_valid,
        )
        arrays["full_green_delay_s"] = obs.forward(relative)
        arrays["full_green_relative_scattered"] = relative
        report["full_green_vs_observed_ns"] = stats(
            (arrays["full_green_delay_s"] - data)[active] * 1e9
        )
        use = obs.frequency_valid
        report["full_green_complex_ratio_relative_residual"] = float(
            np.linalg.norm((relative - ratio)[use]) / np.linalg.norm(ratio[use])
        )
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
