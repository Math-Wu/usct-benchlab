#!/usr/bin/env python3
"""Frozen 64-channel, 256-image comparison using common calibrated xcorr features.

No GT-derived ROI, no tuning, no new simulations. Same weights, receiver split,
regularization presets and online stopping for the four solver families.
First/second-order Bent is an explicit discretization ablation, not two algorithms.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from usctbench.benchmark.runner import run_algorithm_case
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.data.arrival import water_relative_delays, arrival_observation_metadata
from usctbench.data.waveforms import _read_channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="verify and reuse complete outputs after interruption",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve().is_relative_to(repo):
        parser.error("outputs must be outside repository")
    calibration = json.loads(args.calibration.read_text())
    if calibration["selected_on_training_only"] != "xcorr":
        raise ValueError(
            "this frozen experiment requires independent xcorr calibration"
        )
    if (
        Path(calibration["manifest"]["acquisition"]).resolve()
        != args.acquisition.resolve()
    ):
        raise ValueError("calibration belongs to different acquisition settings")
    case = read_case_hdf5(args.case)
    if case.grid.shape != (256, 256) or case.geometry.tx_pos_m.shape != (64, 2):
        raise ValueError("expected existing 256-image, 64-channel case")
    record = json.loads((args.acquisition / "manifest.json").read_text())
    if case.metadata.get("input_sha256") != record["input_sha256"]:
        raise ValueError("case and raw pressure manifest mismatch")
    with h5py.File(args.acquisition / "pressure.mat") as f:
        t = np.asarray(f["time"]).ravel()
        selected = np.arange(0, 128, 2)
        positions = np.asarray(f["transducerPositionsXY"])[:, ::-1][selected]
        if not np.array_equal(positions, case.geometry.tx_pos_m):
            raise ValueError("unexpected channel geometry")
        p = _read_channels(
            f["full_dataset"], ("tx", "rx", "time"), selected, selected, len(t), 2**31
        )
        w = _read_channels(
            f["water_dataset"], ("tx", "rx", "time"), selected, selected, len(t), 2**31
        )
    d = np.linalg.norm(
        case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
    )
    delta, valid, weights, qc = water_relative_delays(
        p, w, t, d, pulse_duration_s=3 / record["source_frequency_hz"], picker="xcorr"
    )
    del p, w
    valid &= d > d.max() * 0.5
    case.grid.roi_mask = None
    case.measurement.delta_tof_s = np.where(valid, delta, np.nan)
    case.measurement.valid_mask = valid
    case.measurement.ray_weights = np.where(valid, weights, 0)
    case.metadata.update(arrival_observation_metadata(qc))
    case.metadata.update(
        {
            "arrival_qc": qc,
            "source_acquisition": str(args.acquisition.resolve()),
            "calibration_sha256": hashlib.sha256(
                args.calibration.read_bytes()
            ).hexdigest(),
            "gt_used_for_data_adaptation": False,
        }
    )
    args.out.mkdir(parents=True, exist_ok=args.resume)
    path = args.out / "common_case.h5"
    if path.exists():
        prior = read_case_hdf5(path)
        for old, new in (
            (prior.measurement.delta_tof_s, case.measurement.delta_tof_s),
            (prior.measurement.ray_weights, case.measurement.ray_weights),
        ):
            np.testing.assert_array_equal(old, new)
        if prior.grid.roi_mask is not None:
            raise ValueError("resume case contains an inversion ROI")
    else:
        write_case_hdf5(case, path)
    register_builtin_algorithms()
    rows = []
    for label, name, order in (
        ("CGLS", "straight_cgls", None),
        ("SIRT", "straight_sirt", None),
        ("SART", "straight_sart", None),
        ("Bent1", "bent_ray_gn", 1),
        ("Bent2", "bent_ray_gn", 2),
    ):
        config = yaml.safe_load(
            (args.handoff / "configs" / (name + ".yaml")).read_text()
        )
        config["parameters"]["roi_update_only"] = False
        if order is not None:
            config["parameters"]["eikonal_order"] = order
        config_path = args.out / (label + ".yaml")
        if config_path.exists():
            if yaml.safe_load(config_path.read_text()) != config:
                raise ValueError("resume config changed")
        else:
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        dest = args.out / label / case.case_id
        if not (args.resume and (dest / "metadata.yaml").exists()):
            print("START", label, flush=True)
            dest = run_algorithm_case(name, path, config_path, args.out / label)
        metadata = yaml.safe_load((dest / "metadata.yaml").read_text())
        if metadata["status"] != "success":
            raise RuntimeError(metadata.get("failure_reason"))
        metrics = json.loads((dest / "metrics.json").read_text())
        row = {
            "label": label,
            "result_dir": str(dest),
            **{
                k: metrics.get(k)
                for k in (
                    "rmse",
                    "ssim",
                    "psnr",
                    "data_relative_residual",
                    "water_background_rmse",
                    "stopping",
                    "evaluation",
                    "evaluation_split",
                )
            },
        }
        rows.append(row)
        (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    splits = [row["evaluation_split"] for row in rows]
    if any(split != splits[0] for split in splits[1:]):
        raise AssertionError("algorithms used different data splits")


if __name__ == "__main__":
    main()
