"""Oracle residual attribution controls. Never repairs a measured k-Wave case."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from usctbench.benchmark.runner import run_algorithm_case
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.evaluation.tof_diagnostics import (
    error_decomposition,
    reciprocity_floor,
    shuffled_pair_error,
)
from tof_audit_common import diagnostic_case


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, required=True, help="shared audit output directory"
    )
    p.add_argument("--handoff", type=Path, required=True)
    args = p.parse_args()
    root = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("diagnostic output must remain outside Git")
    out = root / "attribution"
    out.mkdir(parents=True, exist_ok=False)
    summary = {
        "purpose": "oracle attribution; not clinical or deployable data correction",
        "groups": {},
        "runs": [],
    }
    register_builtin_algorithms()
    for group in ("high_d", "low_d", "low_ob"):
        case = read_case_hdf5(root / "controls" / group / "kwave.h5")
        d = np.load(root / "controls" / group / "predictions.npz")
        np.testing.assert_array_equal(case.geometry.tx_pos_m, case.geometry.rx_pos_m)
        group_report = {}
        for partition, sel in [
            ("all", d["valid"]),
            ("train", d["train"]),
            ("validation", d["validation"]),
        ]:
            group_report[partition] = {
                "decomposition": error_decomposition(
                    d["straight"],
                    d["eikonal"],
                    d["observed"],
                    mask=sel,
                    weights=d["weights"],
                ),
                "reciprocity_floor": reciprocity_floor(
                    d["observed"], mask=sel, weights=d["weights"]
                ),
            }
        summary["groups"][group] = group_report
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    case = read_case_hdf5(root / "controls/high_d/kwave.h5")
    d = np.load(root / "controls/high_d/predictions.npz")
    error = d["observed"] - d["straight"]
    groups = np.full(error.shape, 2, dtype=int)
    groups[d["train"]] = 0
    groups[d["validation"]] = 1
    for seed in range(5):
        noise, info = shuffled_pair_error(
            error,
            mask=d["valid"],
            weights=d["weights"],
            groups=groups,
            seed=seed,
            shuffle_group_ids=[0],
        )
        norm_checks = {}
        for k, label in enumerate(["train", "validation", "unused"]):
            sel = d["valid"] & (groups == k)
            w = d["weights"][sel]
            a, b = error[sel], noise[sel]
            before, after = np.sum(w * a * a), np.sum(w * b * b)
            np.testing.assert_allclose(after, before, rtol=2e-13)
            np.testing.assert_allclose(
                np.dot(w, a), np.dot(w, b), rtol=2e-13, atol=1e-18
            )
            norm_checks[label] = {
                "weighted_sse_before_s2": float(before),
                "weighted_sse_after_s2": float(after),
                "weighted_mean_before_s": float(np.dot(w, a) / w.sum()),
                "weighted_mean_after_s": float(np.dot(w, b) / w.sum()),
            }
        pair_mask = d["valid"] & d["valid"].T
        np.testing.assert_allclose(
            (noise - noise.T)[pair_mask], (error - error.T)[pair_mask], atol=1e-18
        )
        np.testing.assert_array_equal(noise[d["validation"]], error[d["validation"]])
        changed = diagnostic_case(case, d["straight"] + noise, "matched_straight")
        changed.metadata.update(
            diagnostic_control="pair_decorrelated_model_error",
            forward_model="straight_plus_oracle_derived_error",
            shuffle_seed=seed,
            validation_errors_unchanged=True,
            deployable_correction=False,
        )
        case_path = out / f"control_seed_{seed}.h5"
        write_case_hdf5(changed, case_path)
        (out / f"control_seed_{seed}.json").write_text(
            json.dumps({"info": info, "partition_invariants": norm_checks}, indent=2)
        )
        for alg in ("straight_cgls", "straight_sirt", "straight_sart"):
            config = args.handoff / "configs" / f"{alg}.yaml"
            dest = run_algorithm_case(
                alg, case_path, config, out / "runs" / f"seed_{seed}" / alg
            )
            meta = yaml.safe_load((dest / "metadata.yaml").read_text())
            if meta["status"] != "success":
                raise RuntimeError(meta["failure_reason"])
            m = json.loads((dest / "metrics.json").read_text())
            row = {
                "seed": seed,
                "algorithm": alg,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "result_dir": str(dest),
                "metrics": m,
            }
            summary["runs"].append(row)
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(seed, alg, m["rmse"], m["ssim"], m["stopping"]["reason"], flush=True)
    print("COMPLETED 15 split-safe oracle decorrelation controls", flush=True)


if __name__ == "__main__":
    main()
