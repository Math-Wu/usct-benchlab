#!/usr/bin/env python3
"""Audit recorded k-Wave medium, raw-pressure features and ray predictions.

No reconstruction, new simulation, GT-derived data correction or picker
selection. Requires a validate_physics acquisition and its (possibly subset)
feature case. Outputs are private arrays, summary JSON and a diagnostic figure.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from usctbench.core.io import read_case_hdf5
from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.data.arrival import arrival_observation_metadata, water_relative_delays
from usctbench.data.validation_acquisition import (
    read_validation_acquisition,
    sample_speed,
)
from usctbench.data.waveforms import _read_channels
from usctbench.evaluation.arrival_consistency import translated_reference_residual
from usctbench.operators.eikonal import EikonalForward, fast_march
from usctbench.operators.straight_ray import StraightRayProjector


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"count": 0, "rms": None, "mean": None, "median": None, "p95_abs": None}
    return {
        "count": int(a.size),
        "rms": float(np.sqrt(np.mean(a**2))),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95_abs": float(np.quantile(np.abs(a), 0.95)),
    }


def numerical_delays(speed, grid, geometry, spatial_order=1):
    op = EikonalForward(grid, geometry)
    ext = op.extend(1 / speed, background=1 / 1500)
    water = np.full(op.shape, 1 / 1500)
    a, b = [], []
    for source in op.source_coordinates:
        tape = fast_march(ext, grid.spacing_m, source, spatial_order=spatial_order)
        a.append(op.sample_receivers(tape.times))
        del tape
        tape = fast_march(water, grid.spacing_m, source, spatial_order=spatial_order)
        b.append(op.sample_receivers(tape.times))
        del tape
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    return np.asarray(a) - np.asarray(b), np.asarray(b) - distance / 1500


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tx-count", type=int, default=8)
    parser.add_argument("--refine", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--eikonal-order", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not 1 <= args.tx_count <= 64:
        parser.error("tx-count must be in [1,64]")
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo):
        parser.error("private outputs must be outside the source repository")
    acquisition = read_validation_acquisition(args.acquisition)
    case = read_case_hdf5(args.case)
    if case.metadata.get("input_sha256") != acquisition.manifest["input_sha256"]:
        raise ValueError("feature case belongs to a different simulation input")
    if case.metadata.get("reference_sound_speed_mps") != 1500:
        raise ValueError("this validation acquisition uses a 1500 m/s water reference")
    if not np.array_equal(
        case.ground_truth.sound_speed_mps, acquisition.image_speed_mps
    ):
        raise ValueError(
            "case GT differs from original image labels; do not compare different grids"
        )
    for key in ("shape", "spacing_m", "origin_m"):
        if not np.allclose(
            getattr(case.grid, key),
            getattr(acquisition.image_grid, key),
            rtol=1e-10,
            atol=1e-14,
        ):
            raise ValueError("case image grid differs from recorded image grid")
    tx = np.asarray(case.metadata["selected_tx_indices"], dtype=int)
    rx = np.asarray(case.metadata["selected_rx_indices"], dtype=int)
    for positions, indices in (
        (case.geometry.tx_pos_m, tx),
        (case.geometry.rx_pos_m, rx),
    ):
        if not np.array_equal(positions, acquisition.positions_yx_m[indices]):
            raise ValueError(
                "feature case geometry/order differs from recorded acquisition"
            )
    selected = np.linspace(
        0, len(tx), min(args.tx_count, len(tx)), endpoint=False, dtype=int
    )
    geometry = GeometrySpec(
        tx_pos_m=case.geometry.tx_pos_m[selected], rx_pos_m=case.geometry.rx_pos_m
    )
    measured = case.measurement.delta_tof_s[selected].copy()
    active = case.measurement.valid_mask[selected] & np.isfinite(measured)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "case_id": case.case_id,
        "feature_case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "identity": acquisition.identity,
        "acquisition": acquisition.manifest,
        "stored_feature_method": case.metadata["feature_source"],
        "source_tx_indices": tx[selected].tolist(),
        "source_rx_indices": rx.tolist(),
        "active_channels": int(active.sum()),
        "policy": "GT only for posthoc forward comparison; no delay correction, mask/weight change or reconstruction",
        "models": {},
        "pickers": {},
        "implementation_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((repo / "src/usctbench").rglob("*.py"))
        },
    }
    arrays = {"stored_delay_s": measured, "valid": active}

    def save():
        (args.out / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
        np.savez_compressed(args.out / "diagnostics.npz", **arrays)

    save()
    # Reproduce the actual picked observable from original object/water pressure,
    # including low-band waveforms omitted from the web-model handoff.
    with h5py.File(args.acquisition / "pressure.mat") as f:
        t = np.asarray(f["time"]).ravel()
        p, w = [
            _read_channels(
                f[key], ("tx", "rx", "time"), tx[selected], rx, len(t), 512 * 1024**2
            )
            for key in ("full_dataset", "water_dataset")
        ]
        signal = np.asarray(f["source_signal"]).ravel()
        report["source"] = {
            "signal_length": len(signal),
            "signal_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
            "time_start_s": float(t[0]),
            "dt_s": float(np.mean(np.diff(t))),
            "signal_role": "additive k-Wave input, not a calibrated Helmholtz source",
        }
    distance = np.linalg.norm(
        geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1
    )
    duration = 3 / acquisition.manifest["source_frequency_hz"]
    for picker in ("xcorr", "envelope", "aic"):
        delay, valid, _, qc = water_relative_delays(
            p, w, t, distance, picker=picker, pulse_duration_s=duration
        )
        arrays[picker + "_delay_s"] = delay
        residual, _ = translated_reference_residual(
            p, w, t, delay, distance, pulse_duration_s=duration
        )
        arrays[picker + "_translation_residual"] = residual
        row = {
            "qc": qc,
            "observation_contract": arrival_observation_metadata(qc),
            "valid_original_channels": int((valid & active).sum()),
            "shift_gain_residual": stats(residual[active & valid]),
            "vs_stored_ns": stats((delay - measured)[active & valid] * 1e9),
        }
        control = water_relative_delays(
            w, w, t, distance, picker=picker, pulse_duration_s=duration
        )
        row["water_identity_ns"] = stats(control[0][active & control[1]] * 1e9)
        # Fractional-sample shifted raw-water controls test sign and timing only.
        # They are not a surrogate replacement for the real object acquisition.
        row["water_shift_controls"] = []
        flat = w.reshape(len(t), -1)
        for shift in (-250e-9, 250e-9):
            shifted = np.column_stack(
                [np.interp(t - shift, t, trace, left=0, right=0) for trace in flat.T]
            ).reshape(w.shape)
            control = water_relative_delays(
                shifted, w, t, distance, picker=picker, pulse_duration_s=duration
            )
            row["water_shift_controls"].append(
                {
                    "shift_s": shift,
                    "error_ns": stats((control[0] - shift)[active & control[1]] * 1e9),
                    "valid_original_channels": int(np.sum(active & control[1])),
                }
            )
        report["pickers"][picker] = row
        save()
    report["picker_disagreement_ns"] = stats(
        (arrays["xcorr_delay_s"] - arrays["envelope_delay_s"])[active] * 1e9
    )
    # Parent surrogate comparison retains its own acquisition; the controlled
    # models below instead use the identical snapped k-Wave geometry.
    parent = read_case_hdf5(acquisition.manifest["property_case"])
    parent_op = StraightRayProjector.from_case(parent)
    original = parent_op.forward(
        1 / parent.ground_truth.sound_speed_mps - 1 / 1500
    ).reshape(parent_op.ray_shape)
    parent_valid = parent.measurement.valid_mask & np.isfinite(
        parent.measurement.delta_tof_s
    )
    report["parent_surrogate"] = {
        "feature_source": parent.metadata.get("feature_source"),
        "transducers": list(parent_op.ray_shape),
        "grid_shape": list(parent.grid.shape),
        "valid_directed_channels": int(parent_valid.sum()),
        "roi_pixels": (
            None if parent.grid.roi_mask is None else int(parent.grid.roi_mask.sum())
        ),
        "stored_vs_current_Siddon_ns": stats(
            (original - parent.measurement.delta_tof_s)[parent_valid] * 1e9
        ),
        "image_equals_current": bool(
            np.array_equal(
                parent.ground_truth.sound_speed_mps, case.ground_truth.sound_speed_mps
            )
        ),
    }
    for label, grid, speed in (
        ("image_grid", case.grid, acquisition.image_speed_mps),
        ("exact_simulation_grid", acquisition.grid, acquisition.speed_mps),
    ):
        start = time.perf_counter()
        op = StraightRayProjector.from_grid_geometry(grid, geometry)
        straight = op.forward(1 / speed - 1 / 1500).reshape(op.ray_shape)
        arrays[label + "_straight_s"] = straight
        eikonal, water_error = numerical_delays(
            speed, grid, geometry, args.eikonal_order
        )
        arrays[label + "_eikonal_s"] = eikonal
        arrays[label + "_water_error_s"] = water_error
        report["models"][label] = {
            "grid_shape": list(grid.shape),
            "eikonal_order": args.eikonal_order,
            "elapsed_s": time.perf_counter() - start,
            "straight_vs_stored_ns": stats((straight - measured)[active] * 1e9),
            "eikonal_vs_stored_ns": stats((eikonal - measured)[active] * 1e9),
            "raw_eikonal_water_vs_geometry_ns": stats(water_error[active] * 1e9),
        }
        for picker in ("xcorr", "envelope", "aic"):
            report["models"][label]["eikonal_vs_" + picker + "_ns"] = stats(
                (eikonal - arrays[picker + "_delay_s"])[active] * 1e9
            )
        print(label, json.dumps(report["models"][label]), flush=True)
        save()
    report["representation_change_ns"] = stats(
        (arrays["exact_simulation_grid_straight_s"] - arrays["image_grid_straight_s"])[
            active
        ]
        * 1e9
    )
    previous = arrays["exact_simulation_grid_eikonal_s"]
    for factor in ((2, 4) if args.refine == 4 else (2,) if args.refine == 2 else ()):
        g = acquisition.grid
        # Preserve the EXACT original nodes and extent of the centre interpolant.
        # Re-centering to doubled pixel count would silently change that field.
        grid = GridSpec(
            shape=tuple(factor * (n - 1) + 1 for n in g.shape),
            spacing_m=tuple(h / factor for h in g.spacing_m),
            origin_m=tuple(
                o + h / 2 - h / (2 * factor) for o, h in zip(g.origin_m, g.spacing_m)
            ),
        )
        speed = sample_speed(acquisition.speed_mps, g, grid)
        start = time.perf_counter()
        refined, water_error = numerical_delays(
            speed, grid, geometry, args.eikonal_order
        )
        arrays["refined_eikonal_s"] = refined
        arrays[f"refined_{factor}_eikonal_s"] = refined
        report["refinement"] = {
            "grid_shape": list(grid.shape),
            "elapsed_s": time.perf_counter() - start,
            "original_nodes_preserved_max_mps": float(
                np.max(np.abs(speed[::factor, ::factor] - acquisition.speed_mps))
            ),
            "successive_delta_change_ns": stats((refined - previous)[active] * 1e9),
            "refined_vs_stored_ns": stats((refined - measured)[active] * 1e9),
            "raw_water_error_ns": stats(water_error[active] * 1e9),
            "continuum_convergence_certified": False,
        }
        report.setdefault("refinements", []).append(
            dict(report["refinement"], factor=factor)
        )
        previous = refined
        save()
        print("refinement", json.dumps(report["refinement"]), flush=True)
    save()
    render(arrays, args.out, case.case_id)
    print(json.dumps(report, indent=2), flush=True)


def render(arrays, out, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [
        "stored_delay_s",
        "xcorr_delay_s",
        "envelope_delay_s",
        "aic_delay_s",
        "image_grid_straight_s",
        "exact_simulation_grid_eikonal_s",
    ]
    labels = [
        "Stored feature",
        "Bounded xcorr",
        "Envelope onset",
        "Modified AIC onset (candidate)",
        "Siddon / image grid",
        "Eikonal / exact medium",
    ]
    scale = (
        max(float(np.nanmax(np.abs(arrays[k][arrays["valid"]]))) for k in keys) * 1e6
    )
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 12), constrained_layout=True)
    for ax, key, label in zip(axes, keys, labels):
        a = np.where(arrays["valid"], arrays[key] * 1e6, np.nan)
        im = ax.imshow(a, aspect="auto", cmap="coolwarm", vmin=-scale, vmax=scale)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("TX subset")
    axes[-1].set_xlabel("RX index")
    fig.colorbar(im, ax=axes, label="Water-relative delay (microseconds)", shrink=0.75)
    fig.suptitle(f"{name}: observable / forward-model audit", fontsize=14)
    fig.savefig(out / "adaptation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
