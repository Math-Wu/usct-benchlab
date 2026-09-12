"""Frozen ToF observable diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
from pathlib import Path
import gc
import json
import time
import numpy as np
from usctbench.core.io import read_case_hdf5
from usctbench.data.arrival import water_relative_delays
from usctbench.operators.eikonal import EikonalForward, fast_march
from usctbench.core.schema import GeometrySpec, GridSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="shared audit directory, outside the source repository",
    )
    parser.add_argument(
        "--handoff", type=Path, required=True, help="extracted usct_handoff_64"
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    R = args.out.resolve()
    c = read_case_hdf5(
        args.handoff.resolve() / "cases/high_band/D510022534/envelope_case.h5"
    )
    m = c.measurement
    P, W, t = (m.time_data, m.water_reference_time, m.time_axis_s)
    mask = m.valid_mask & np.isfinite(m.delta_tof_s)
    distance = np.linalg.norm(
        c.geometry.tx_pos_m[:, None] - c.geometry.rx_pos_m[None], axis=-1
    )
    params = {
        "reference_speed_mps": 1500,
        "speed_bounds": (1300, 1700),
        "pulse_duration_s": 6e-06,
    }
    out = R / "measurement_diagnosis"
    out.mkdir(parents=True, exist_ok=False)

    def brief(x):
        x = np.asarray(x)
        return {
            "count": int(x.size),
            "rms": float(np.sqrt(np.mean(x * x))),
            "median": float(np.median(x)),
            "p95_abs": float(np.quantile(np.abs(x), 0.95)),
            "max_abs": float(np.max(np.abs(x))),
        }

    report = {
        "purpose": "diagnostic, not data correction or GT-based picker selection",
        "sampling_dt_s": float(np.mean(np.diff(t))),
        "time_start_s": float(t[0]),
        "picker_controls": [],
    }
    picks = {}
    for picker in ["envelope", "xcorr"]:
        a, v, w, q = water_relative_delays(P, W, t, distance, picker=picker, **params)
        picks[picker] = (a, v)
        report[picker + "_valid_original_channels"] = int(np.sum(v & mask))
        if picker == "envelope":
            report["stored_envelope_reproduction_error_ns"] = brief(
                (a - m.delta_tof_s)[mask & v] * 1000000000.0
            )
    np.savez_compressed(
        out / "repicked.npz",
        envelope=picks["envelope"][0],
        xcorr=picks["xcorr"][0],
        valid=mask,
    )
    report["picker_difference_ns"] = brief(
        (picks["xcorr"][0] - picks["envelope"][0])[
            mask & picks["xcorr"][1] & picks["envelope"][1]
        ]
        * 1000000000.0
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    pairs = np.argwhere(np.triu(mask, 1))
    pairs = pairs[np.linspace(0, len(pairs) - 1, 128, dtype=int)]
    w = W[:, pairs[:, 0], pairs[:, 1]].astype(float)[:, :, None]
    d = distance[pairs[:, 0], pairs[:, 1]][:, None]
    for shift in [-2.5e-07, 0, 2.5e-07]:
        shifted = np.column_stack(
            [
                np.interp(t - shift, t, w[:, k, 0], left=0, right=0)
                for k in range(len(pairs))
            ]
        )[:, :, None]
        for picker in ["envelope", "xcorr"]:
            a, v, weight, q = water_relative_delays(
                shifted, w, t, d, picker=picker, **params
            )
            report["picker_controls"].append(
                {
                    "picker": picker,
                    "true_shift_ns": shift * 1000000000.0,
                    "valid": int(v.sum()),
                    "total": int(v.size),
                    "error_ns": brief((a[v] - shift) * 1000000000.0),
                    "shift_engine": "linear interpolation with zero exterior; finite-sampling control",
                }
            )
    distortions = {}
    for picker, (delays, valid) in picks.items():
        errors = []
        for tx, rx in pairs:
            if not valid[tx, rx]:
                continue
            d = distance[tx, rx]
            tau = delays[tx, rx]
            window = (t >= d / 1700 - 6e-06) & (t <= d / 1300 + 1.2e-05)
            pred = np.interp(t[window] - tau, t, W[:, tx, rx], left=0, right=0)
            target = P[window, tx, rx].astype(float)
            gain = np.dot(pred, target) / np.dot(pred, pred)
            errors.append(np.linalg.norm(target - gain * pred) / np.linalg.norm(target))
        distortions[picker] = brief(errors)
    report["shift_and_scalar_gain_waveform_residual"] = distortions
    report["translation_scope"] = (
        "A picker can be accurate on shifted water yet measure a different finite-band quantity on distorted object traces. These tests cannot certify physical first arrivals."
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    m.time_data = m.water_reference_time = None
    del P, W, w
    gc.collect()
    indices = np.arange(0, 64, 8)
    geometry = GeometrySpec(
        tx_pos_m=c.geometry.tx_pos_m[indices], rx_pos_m=c.geometry.rx_pos_m
    )
    values = {}
    timing = {}
    for factor in [1, 2, 4]:
        start = time.perf_counter()
        grid = GridSpec(
            shape=tuple((factor * n for n in c.grid.shape)),
            spacing_m=tuple((h / factor for h in c.grid.spacing_m)),
            origin_m=c.grid.origin_m,
        )
        op = EikonalForward(grid, geometry, background_speed_mps=1500)
        s = np.repeat(
            np.repeat(1 / c.ground_truth.sound_speed_mps, factor, axis=0),
            factor,
            axis=1,
        )
        ext = op.extend(s, background=1 / 1500)
        water = np.full(op.shape, 1 / 1500)
        times = []
        for k, source in enumerate(op.source_coordinates):
            tape = fast_march(ext, grid.spacing_m, source)
            a = op.sample_receivers(tape.times)
            del tape
            tape = fast_march(water, grid.spacing_m, source)
            b = op.sample_receivers(tape.times)
            del tape
            times.append(a - b)
        values[str(256 * factor)] = np.array(times)
        timing[str(256 * factor)] = time.perf_counter() - start
        np.savez_compressed(out / "grid_fields.npz", tx_indices=indices, **values)
        del op, ext, water, s
        gc.collect()
        print("REFINED", factor, timing, flush=True)
    sel = mask[indices]
    obs = m.delta_tof_s[indices]
    refinement = {
        "tx_indices": indices.tolist(),
        "rx_count": 64,
        "active_channels": int(sel.sum()),
        "fixed_field": "nearest repetition preserves original 256-grid piecewise field; not the unavailable k-Wave simulation grid field",
        "vs_kwave_us": {
            n: brief((a - obs)[sel] * 1000000.0) for n, a in values.items()
        },
        "consecutive_change_us": {
            "256_to_512": brief((values["512"] - values["256"])[sel] * 1000000.0),
            "512_to_1024": brief((values["1024"] - values["512"])[sel] * 1000000.0),
        },
        "elapsed_s": timing,
        "continuum_certified": False,
    }
    report["refinement"] = refinement
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(refinement), flush=True)


if __name__ == "__main__":
    main()
