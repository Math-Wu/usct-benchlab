"""Frozen existing-delay versus source-frequency phase pilot on the 64 handoff.

No simulations, parameter sweep, raw time re-picking, GT initialization, or
postprocessing. Existing observations/weights/geometry are preserved, except
for explicitly replacing the delay feature. Outputs and private source
snapshots stay outside Git. Each completed reconstruction is saved immediately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.data.phase_travel_time import water_relative_phase_delays

ALGORITHMS = ("straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(case, feature):
    """No GT fields inspected: labels are carried unchanged for later evaluation."""
    m = case.measurement
    before = np.asarray(m.valid_mask, dtype=bool) & np.isfinite(m.delta_tof_s)
    if feature == "existing":
        values, valid, qc = (
            m.delta_tof_s,
            before,
            {"method": case.metadata.get("feature_source", "existing")},
        )
    else:
        if case.metadata.get("frequency_convention") != "exp(-i omega t)":
            raise ValueError("explicit exp(-i omega t) pressure convention required")
        frequency = float(case.metadata["source_frequency_hz"])
        index = np.flatnonzero(m.frequencies_hz == frequency)
        if index.size != 1:
            raise ValueError(
                "source frequency must be present exactly; no nearest-bin substitution"
            )
        k = int(index[0])
        distance = np.linalg.norm(
            case.geometry.tx_pos_m[:, None] - case.geometry.rx_pos_m[None], axis=-1
        )
        values, valid, qc = water_relative_phase_delays(
            m.freq_data[k],
            m.water_reference[k],
            frequency,
            m.delta_tof_s,
            distance,
            valid_mask=before,
        )
        if not np.array_equal(before, valid):
            raise ValueError(f"phase pilot refuses a changed valid mask: {qc}")
    measurement = m.model_copy(
        update={
            "delta_tof_s": np.array(values, copy=True),
            "tof_s": None,
            "valid_mask": valid,
            "time_data": None,
            "water_reference_time": None,
        }
    )
    metadata = {
        **case.metadata,
        "feature_source": qc["method"],
        "feature_ablation": qc,
        "uses_gt_generated_measurement": True,
        "raw_storage_note": "raw remains in original supplied case; no time-domain repicking in this pilot",
    }
    return (
        case.model_copy(update={"measurement": measurement, "metadata": metadata}),
        qc,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument(
        "--case",
        type=Path,
        required=True,
        help="case path relative to handoff, or absolute",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=["existing", "source_phase"],
        default=["existing", "source_phase"],
    )
    parser.add_argument(
        "--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS)
    )
    parser.add_argument(
        "--model-grid-shape",
        nargs=2,
        type=int,
        default=None,
        help="optional coefficient grid only; propagation remains unchanged",
    )
    args = parser.parse_args()
    root, out = args.handoff.resolve(), args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if out == repo or repo in out.parents:
        raise ValueError("output must be outside the source repository")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "source_snapshot"
    shutil.copytree(
        repo,
        snapshot,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "*.egg-info", ".ruff_cache"
        ),
    )
    source_hashes = {
        str(p.relative_to(snapshot)): file_hash(p)
        for p in sorted(snapshot.rglob("*.py"))
    }
    (out / "source_hashes.json").write_text(json.dumps(source_hashes, indent=2))
    case_path = (root / args.case).resolve()
    initial_hash = file_hash(case_path)
    case = read_case_hdf5(case_path)
    protocol = {
        "case": str(args.case),
        "case_sha256": initial_hash,
        "feature_frequency_policy": "exact source_frequency_hz",
        "image_grid": list(case.grid.shape),
        "model_grid_shape": args.model_grid_shape,
        "data_subset": "unaltered supplied channels",
        "source_tree_sha256": hashlib.sha256(
            json.dumps(source_hashes, sort_keys=True).encode()
        ).hexdigest(),
        "ground_truth_policy": "evaluation only; these are development cases, not blind tests",
        "features": args.features,
        "algorithms": args.algorithms,
        "configs": {},
    }
    rows = []
    split = None
    # Configs and source are fixed before running the first algorithm.
    for algorithm in args.algorithms:
        source_config = root / "configs" / (algorithm + ".yaml")
        config = yaml.safe_load(source_config.read_text())
        if args.model_grid_shape is not None:
            config["parameters"]["model_grid_shape"] = args.model_grid_shape
        if (
            config["parameters"].get("roi_update_only")
            and case.grid.roi_mask is not None
        ):
            raise ValueError(
                "handoff pilot requires no inversion ROI; do not use a GT mask"
            )
        if config["parameters"]["stopping"].get("allow_ground_truth_stopping", False):
            raise ValueError("oracle stopping is disallowed")
        destination = out / (algorithm + ".yaml")
        destination.write_text(yaml.safe_dump(config))
        protocol["configs"][algorithm] = config
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for feature in args.features:
        derived, qc = prepare(case, feature)
        stage = out / feature
        stage.mkdir()
        (stage / "feature_qc.json").write_text(json.dumps(qc, indent=2))
        derived_path = stage / "case.h5"
        write_case_hdf5(derived, derived_path)
        for algorithm in args.algorithms:
            command = [
                sys.executable,
                "-m",
                "usctbench.cli",
                "run",
                algorithm,
                "--case",
                str(derived_path),
                "--config",
                str(out / (algorithm + ".yaml")),
                "--out",
                str(stage / algorithm),
            ]
            with (stage / (algorithm + ".log")).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=snapshot,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={
                        **os.environ,
                        "PYTHONPATH": str(snapshot / "src"),
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                    },
                    check=False,
                )
            destination = stage / algorithm / case.case_id
            metadata = yaml.safe_load((destination / "metadata.yaml").read_text())
            if result.returncode or metadata["status"] != "success":
                raise RuntimeError(f"failed reconstruction: inspect {destination}")
            metrics = json.loads((destination / "metrics.json").read_text())
            if metrics["implementation"]["source_changed_during_run"]:
                raise RuntimeError("source changed during run")
            if split is None:
                split = metrics["evaluation_split"]
            if split != metrics["evaluation_split"]:
                raise RuntimeError(
                    "input weights/validation partition changed between methods or features"
                )
            row = {
                "feature": feature,
                "algorithm": algorithm,
                "runtime_s": metadata["runtime_s"],
                **{
                    k: metrics.get(k)
                    for k in (
                        "rmse",
                        "psnr",
                        "ssim",
                        "water_background_rmse",
                        "water_background_bias",
                        "full_image_rmse",
                        "data_relative_residual",
                        "stop_reason",
                    )
                },
                "receiver_relative_residual": metrics["evaluation"]["receiver"][
                    "weighted_relative_residual"
                ],
                "stopping": metrics["stopping"],
            }
            rows.append(row)
            (out / "summary.json").write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    if file_hash(case_path) != initial_hash:
        raise RuntimeError("input file changed")
    (out / "COMPLETE.json").write_text(
        json.dumps({"input_unchanged": True, "completed_runs": len(rows)})
    )


if __name__ == "__main__":
    main()
