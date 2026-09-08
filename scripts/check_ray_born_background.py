#!/usr/bin/env python3
"""Post-hoc physics diagnostic against an independent pressure case with labels.

Labels are used only to evaluate the forward model, never to calibrate its source
or choose reconstruction parameters. This is not an inverse reconstruction run.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from usctbench.core.io import read_case_hdf5
from usctbench.data.calibration import fit_water_source
from usctbench.evaluation.data import residual_statistics
from usctbench.operators.forward.ray_born import RayBornForward, RayBornOperator


def check(case_path, out, green_backend="eikonal_wkb"):
    case = read_case_hdf5(case_path)
    speed = case.ground_truth.sound_speed_mps
    if speed is None or case.measurement.water_reference is None:
        raise ValueError(
            "independent pressure, water reference and evaluation label required"
        )
    freqs = case.measurement.frequencies_hz
    water = RayBornOperator(
        case.grid, case.geometry, freqs, max_cache_bytes=512 * 1024**2
    )
    valid = water.valid_pair_mask.copy()
    if case.measurement.valid_mask is not None:
        valid &= case.measurement.valid_mask
    source, calibration = fit_water_source(
        water.background_data(), case.measurement.water_reference, valid_mask=valid
    )
    water.source_spectrum = source
    water_prediction = water.background_data()
    forward = RayBornForward(
        case.grid,
        case.geometry,
        freqs,
        source_spectrum=source,
        max_cache_bytes=512 * 1024**2,
        green_backend=green_backend,
    )
    model = 1 / speed**2
    lin = forward.linearize(model)
    rng = np.random.default_rng(29)
    direction = gaussian_filter(rng.normal(size=case.grid.shape), 6)
    direction *= 1e-9 / np.max(np.abs(direction))
    eps = 0.02
    numerical = (
        forward.forward(model + eps * direction)
        - forward.forward(model - eps * direction)
    ) / (2 * eps)
    born = lin.jacobian.forward(direction)
    active = np.broadcast_to(valid, born.shape)
    u, v = numerical[active], born[active]
    stats = {
        "case_id": case.case_id,
        "green_backend": green_backend,
        "image_shape": list(case.grid.shape),
        "transmitters": len(case.geometry.tx_pos_m),
        "receivers": len(case.geometry.rx_pos_m),
        "scope": "posthoc_GT_forward_diagnostic_not_parameter_selection",
        "source_calibration": calibration,
        "inverse_grid_ppw": float(
            speed.min() / (max(freqs) * max(case.grid.spacing_m))
        ),
        "initial_water_model_vs_object": residual_statistics(
            water_prediction, case.measurement.freq_data, mask=valid
        ),
        "GT_model_vs_object": residual_statistics(
            lin.value, case.measurement.freq_data, mask=valid
        ),
        "born_vs_background_derivative_relative_error": float(
            np.linalg.norm(v - u) / np.linalg.norm(u)
        ),
        "born_vs_background_derivative_cosine": float(
            np.vdot(u, v).real / (np.linalg.norm(u) * np.linalg.norm(v))
        ),
        "GT_log_amplitude_ratio_min": float(lin.jacobian.log_amplitude_ratio.min()),
        "GT_log_amplitude_ratio_max": float(lin.jacobian.log_amplitude_ratio.max()),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "background_diagnostic.json").write_text(json.dumps(stats, indent=2))
    index = len(freqs) // 2
    obs, pred = case.measurement.freq_data[index], lin.value[index]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    maximum = max(np.abs(obs[valid]).max(), np.abs(pred[valid]).max())
    for ax, values, title in zip(
        axes[:2], (obs, pred), ("k-Wave measured amplitude", "Model amplitude at GT")
    ):
        image = ax.imshow(
            np.where(valid, np.abs(values), np.nan),
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=maximum,
        )
        ax.set_title(title)
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Transmitter")
        fig.colorbar(image, ax=ax)
    phase = np.abs(np.angle(pred * obs.conj()))
    image = axes[2].imshow(
        np.where(valid, phase, np.nan), origin="lower", cmap="gray", vmin=0, vmax=np.pi
    )
    axes[2].set_title("Absolute phase error at GT (rad)")
    fig.colorbar(image, ax=axes[2])
    fig.suptitle(
        f"{case.case_id} | {freqs[index]/1000:.0f} kHz | post-hoc forward check"
    )
    fig.savefig(out / "background_diagnostic.png", dpi=150)
    plt.close(fig)
    print(json.dumps(stats, indent=2), flush=True)


def check_ray_observable(case_path, out):
    from usctbench.operators.forward.eikonal import EikonalForward
    from usctbench.operators.forward.straight_ray import StraightRayProjector

    case = read_case_hdf5(case_path)
    truth = case.ground_truth.sound_speed_mps
    measured = case.measurement.delta_tof_s
    if truth is None or measured is None:
        raise ValueError(
            "post-hoc ToF check requires evaluation label and measured delays"
        )
    straight = StraightRayProjector.from_case(case)
    linear = straight.forward(1 / truth - 1 / 1500).reshape(measured.shape)
    eikonal = EikonalForward(case.grid, case.geometry)
    distance = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    bent = eikonal.forward(1 / truth).reshape(measured.shape) - distance / 1500
    valid = np.isfinite(measured) & case.measurement.valid_mask
    stats = {
        "case_id": case.case_id,
        "scope": "posthoc_GT_forward_only_not_surrogate_measurement_generation",
    }
    for name, prediction in (("straight_GT", linear), ("eikonal_GT", bent)):
        stats[name] = residual_statistics(prediction, measured, mask=valid)
        stats[name]["correlation"] = float(
            np.corrcoef(prediction[valid], measured[valid])[0, 1]
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "tof_model_check.json").write_text(json.dumps(stats, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    limit = np.max(np.abs(measured[valid])) * 1e6
    for ax, values, title in zip(
        axes,
        (measured, linear, bent),
        (
            "Measured pulse delay",
            "Straight prediction at GT",
            "Eikonal prediction at GT",
        ),
    ):
        im = ax.imshow(
            np.where(valid, values * 1e6, np.nan),
            origin="lower",
            cmap="gray",
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(title)
        ax.set_xlabel("Receiver")
    fig.colorbar(im, ax=axes, label="microseconds", shrink=0.8)
    fig.savefig(out / "tof_model_check.png", dpi=150)
    plt.close(fig)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ray-tof", action="store_true")
    parser.add_argument(
        "--green-backend",
        choices=["eikonal_wkb", "volume_integral"],
        default="eikonal_wkb",
    )
    args = parser.parse_args()
    if args.ray_tof:
        check_ray_observable(args.case, args.out)
    else:
        check(args.case, args.out, args.green_backend)
