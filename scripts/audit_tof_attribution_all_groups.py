"""Oracle discrepancy attribution; no measured data correction or default change.

Run after audit_tof_prepare.py. Five fixed seeds change the spatial structure of
weighted pair errors, not their partition energy, mean or antisymmetric part.
Cross-partition pairs are left unchanged. This is not an iid noise experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from tof_audit_common import diagnostic_case
from usctbench.benchmark.runner import run_algorithm_case
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.evaluation.tof_diagnostics import (
    error_decomposition,
    reciprocity_floor,
    shuffled_pair_error,
)

SEEDS = tuple(range(20260908, 20260913))
ALGORITHMS = ("straight_cgls", "straight_sirt", "straight_sart")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def conservation_check(before, after, valid, weights, groups):
    """Verify the full groups, including unchanged cross-partition observations."""
    rows = {}
    for group in np.unique(groups[valid]):
        active = valid & (groups == group)
        w, a, b = weights[active], before[active], after[active]
        energy = float(w @ (a * a))
        norm_error = abs(float(w @ (b * b)) - energy) / max(energy, 1e-300)
        mean_error = abs(float(w @ (b - a))) / max(np.sqrt(energy * w.sum()), 1e-300)
        if norm_error > 1e-10 or mean_error > 1e-10:
            raise ValueError("decorrelation did not preserve partition moments")
        rows[str(group)] = {
            "count": int(active.sum()),
            "relative_energy_error": norm_error,
            "normalized_mean_error": mean_error,
        }
    pairs = valid & valid.T
    defect = (after - after.T - before + before.T)[pairs]
    scale = max(float(np.max(np.abs(before[valid]))), 1e-300)
    if np.max(np.abs(defect)) / scale > 1e-10:
        raise ValueError("decorrelation changed the antisymmetric error")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    args = parser.parse_args()
    root, handoff = args.out.resolve(), args.handoff.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("private outputs must stay outside the repository")
    destination = root / "attribution"
    destination.mkdir(parents=True, exist_ok=False)
    register_builtin_algorithms()
    report = {"groups": {}, "controls": [], "oracle_diagnostic_only": True}
    for group in ("high_d", "low_d", "low_ob"):
        case = read_case_hdf5(root / "controls" / group / "kwave.h5")
        if not np.array_equal(case.geometry.tx_pos_m, case.geometry.rx_pos_m):
            raise ValueError("this audit requires aligned reciprocal geometry")
        with np.load(root / "controls" / group / "predictions.npz") as data:
            data = dict(data)
        sections = {}
        for name, mask in (
            ("all", data["valid"]),
            ("train", data["train"]),
            ("validation", data["validation"]),
        ):
            sections[name] = {
                "identity": error_decomposition(
                    data["straight"],
                    data["eikonal"],
                    data["observed"],
                    mask=mask,
                    weights=data["weights"],
                ),
                "reciprocity_floor": reciprocity_floor(
                    data["observed"],
                    mask=mask,
                    weights=data["weights"],
                ),
            }
        report["groups"][group] = sections
    path = root / "controls/high_d/predictions.npz"
    with np.load(path) as stored:
        data = dict(stored)
    case = read_case_hdf5(root / "controls/high_d/kwave.h5")
    error = data["observed"] - data["straight"]
    groups = np.full(error.shape, 2, dtype=int)
    groups[data["train"]], groups[data["validation"]] = 0, 1
    protocol = {
        "seeds": SEEDS,
        "algorithms": ALGORITHMS,
        "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "partition_codes": {"0": "train", "1": "validation", "2": "unused"},
        "cross_partition_pairs": "unchanged; no validation errors enter training",
        "not_iid_gaussian_noise": True,
        "configs": {},
    }
    for algorithm in ALGORITHMS:
        cfg = yaml.safe_load((handoff / "configs" / (algorithm + ".yaml")).read_text())
        if cfg["parameters"]["stopping"].get("allow_ground_truth_stopping", False):
            raise ValueError("oracle stopping is forbidden")
        protocol["configs"][algorithm] = cfg
        (destination / (algorithm + ".yaml")).write_text(yaml.safe_dump(cfg))
    atomic_json(destination / "protocol.json", protocol)
    atomic_json(destination / "summary.json", report)
    for seed in SEEDS:
        shuffled, qc = shuffled_pair_error(
            error, mask=data["valid"], weights=data["weights"], groups=groups, seed=seed
        )
        qc["conservation"] = conservation_check(
            error, shuffled, data["valid"], data["weights"], groups
        )
        control = diagnostic_case(case, data["straight"] + shuffled, "matched_straight")
        control.metadata.update(
            diagnostic_control="decorrelated_straight_discrepancy",
            forward_model="siddon_plus_decorrelated_oracle_discrepancy",
            inverse_crime_risk="GT_assisted_attribution_not_acquired_data",
            decorrelation_qc=qc,
        )
        folder = destination / str(seed)
        folder.mkdir()
        input_path = folder / "control.h5"
        write_case_hdf5(control, input_path)
        read_case_hdf5(input_path)
        for algorithm in ALGORITHMS:
            result = run_algorithm_case(
                algorithm,
                input_path,
                destination / (algorithm + ".yaml"),
                folder / algorithm,
            )
            status = yaml.safe_load((result / "metadata.yaml").read_text())
            if status["status"] != "success":
                raise RuntimeError(status.get("failure_reason"))
            metrics = json.loads((result / "metrics.json").read_text())
            row = {
                "seed": seed,
                "algorithm": algorithm,
                "qc": qc,
                "result": str(result),
                **{
                    key: metrics.get(key)
                    for key in (
                        "rmse",
                        "ssim",
                        "psnr",
                        "water_background_rmse",
                        "stopping",
                        "evaluation",
                        "evaluation_split",
                    )
                },
            }
            report["controls"].append(row)
            atomic_json(destination / "summary.json", report)
            print(json.dumps(row), flush=True)
    atomic_json(destination / "COMPLETE.json", {"successful_reconstructions": 15})


if __name__ == "__main__":
    main()
