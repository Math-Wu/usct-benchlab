"""Collect executed ToF controls; missing outputs cannot masquerade as completed runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from usctbench.core.io import read_result_hdf5

ALGORITHMS = ("straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("private reports must stay outside the repository")
    report = root / "report"
    report.mkdir(exist_ok=True)
    expected = []
    for group in ("high_d", "low_d", "low_ob"):
        for feature in ("matched_straight", "matched_eikonal", "kwave"):
            for alg in ALGORITHMS:
                base = (
                    root / "baseline" / alg
                    if group == "high_d" and feature == "kwave"
                    else root / "runs" / group / feature / alg
                )
                expected.append(("matrix", group, feature, alg, base))
    for seed in range(5):
        for alg in ALGORITHMS[:3]:
            expected.append(
                (
                    "decorrelation",
                    "high_d",
                    f"seed_{seed}",
                    alg,
                    root / "attribution/runs" / f"seed_{seed}" / alg,
                )
            )
    for feature in ("matched_eikonal", "kwave"):
        expected.append(
            ("warm", "high_d", feature, "bent_ray_gn", root / "warm" / feature)
        )
    for alg in ALGORITHMS:
        expected.append(("smooth", "analytic", "matched", alg, root / "smooth" / alg))
    for feature in ("matched_straight", "kwave"):
        expected.append(
            (
                "coarse",
                "high_d",
                feature,
                "straight_cgls",
                root / "coarse_optimum/cgls" / feature,
            )
        )
    rows, records, missing = [], [], []
    for family, group, feature, alg, base in expected:
        paths = list(base.glob("*/result.h5")) if base.exists() else []
        if not paths:
            missing.append(str(base.relative_to(root)))
            continue
        if len(paths) != 1:
            raise ValueError(f"ambiguous results in {base}")
        p = paths[0]
        status = yaml.safe_load((p.parent / "metadata.yaml").read_text())
        if status["status"] != "success":
            raise ValueError(f"failed result: {p}")
        result = read_result_hdf5(p)
        if result.algorithm != alg or not np.isfinite(result.sound_speed_mps).all():
            raise ValueError(f"nonfinite or wrong algorithm: {p}")
        m = json.loads((p.parent / "metrics.json").read_text())
        stop = m["stopping"]
        if stop["ground_truth_used_for_stopping"]:
            raise ValueError(f"GT stopping: {p}")
        val = m.get("evaluation", {}).get("receiver", {})
        row = {
            "family": family,
            "group": group,
            "feature": feature,
            "algorithm": alg,
            **{
                k: m.get(k)
                for k in (
                    "rmse",
                    "ssim",
                    "psnr",
                    "full_image_rmse",
                    "water_background_rmse",
                    "water_background_bias",
                )
            },
            "train_relative": m.get("data_relative_residual"),
            "validation_relative": val.get("weighted_relative_residual"),
            "stop_reason": stop["reason"],
            "selected_iteration": stop["selected_iteration"],
            "completed_iterations": stop["completed_iterations"],
            "elapsed_s": stop["elapsed_s"],
            "forward_calls": stop["work"].get("forward_calls"),
            "adjoint_calls": stop["work"].get("adjoint_calls"),
            "result_path": str(p.relative_to(root)),
            "result_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        rows.append(row)
        records.append(
            {
                **row,
                "stopping": stop,
                "split": m.get("evaluation_split"),
                "image_evaluation": m.get("image_evaluation"),
            }
        )
    # Controls within a breast group must use the same masks and evaluation policy.
    for group in ("high_d", "low_d", "low_ob"):
        primary = [
            r for r in records if r["family"] == "matrix" and r["group"] == group
        ]
        if primary:
            for r in primary:
                if (
                    r["split"] != primary[0]["split"]
                    or r["image_evaluation"] != primary[0]["image_evaluation"]
                ):
                    raise ValueError(
                        f"changed acquisition/evaluation policy: {r['result_path']}"
                    )
    with (report / "runs.csv").open("w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "scope": "current execution only; oracle controls are NOT acquired-data improvements",
        "expected_pipeline_runs": len(expected),
        "completed_pipeline_runs": len(rows),
        "missing": missing,
        "complete": not missing,
        "distinct_breast_anatomies": 2,
        "independent_patient_test": False,
        "runs": records,
        "decorrelation_summary": {},
    }
    for alg in ALGORITHMS[:3]:
        selected = [
            r for r in rows if r["family"] == "decorrelation" and r["algorithm"] == alg
        ]
        summary["decorrelation_summary"][alg] = {
            "seeds_completed": len(selected),
            **(
                {
                    metric: {
                        "mean": float(np.mean([r[metric] for r in selected])),
                        "min": min(r[metric] for r in selected),
                        "max": max(r[metric] for r in selected),
                    }
                    for metric in ("rmse", "ssim")
                }
                if selected
                else {}
            ),
        }
    (report / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"{len(rows)}/{len(expected)} executed pipeline results; {len(missing)} missing"
    )
    if missing and not args.allow_partial:
        raise RuntimeError("incomplete campaign; see report/summary.json")


if __name__ == "__main__":
    main()
