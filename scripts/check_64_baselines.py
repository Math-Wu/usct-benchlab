"""Check frozen baseline arrays, evaluation splits and actual stopping records.

This is an acceptance check, not reconstruction or GT-based model selection.
It never adjusts tolerances, inputs, parameters, or iteration limits to pass.
Missing/failed/nonfinite results are failures, including partially completed runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

ALGORITHMS = ("straight_cgls", "straight_sirt", "straight_sart", "bent_ray_gn")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_result(reference, actual, *, array_atol=1e-9, metric_atol=1e-9):
    """Return an auditable comparison, even when an input is missing or corrupt."""
    if (
        not np.isfinite([array_atol, metric_atol]).all()
        or min(array_atol, metric_atol) < 0
    ):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    reference, actual = Path(reference), Path(actual)
    record = {"reference": str(reference), "actual": str(actual), "passed": False}
    try:
        for name, directory in (("reference", reference), ("actual", actual)):
            record[name + "_sha256"] = {
                filename: file_hash(directory / filename)
                for filename in ("result.h5", "metrics.json", "metadata.yaml")
            }
        with (
            h5py.File(reference / "result.h5") as ref,
            h5py.File(actual / "result.h5") as run,
        ):
            a, b = ref["sound_speed_mps"][()], run["sound_speed_mps"][()]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite sound-speed array")
        ref_metrics = json.loads((reference / "metrics.json").read_text())
        run_metrics = json.loads((actual / "metrics.json").read_text())
        ref_meta = yaml.safe_load((reference / "metadata.yaml").read_text())
        run_meta = yaml.safe_load((actual / "metadata.yaml").read_text())
        delta = a.astype(np.float64) - b.astype(np.float64)
        record["array_max_absolute_error_mps"] = float(np.max(np.abs(delta)))
        record["array_rms_error_mps"] = float(np.sqrt(np.mean(delta**2)))
        checks = {
            "array": record["array_max_absolute_error_mps"] <= array_atol,
            "status": ref_meta["status"] == run_meta["status"] == "success",
        }
        for key in ("case_id", "algorithm"):
            checks[key] = ref_meta[key] == run_meta[key]
        for key in (
            "evaluation_split",
            "image_evaluation",
            "primary_image_region",
            "stop_reason",
        ):
            checks[key] = ref_metrics[key] == run_metrics[key]
        for key in (
            "reason",
            "completed_iterations",
            "selected_iteration",
            "triggered_rules",
        ):
            checks["stopping." + key] = (
                ref_metrics["stopping"][key] == run_metrics["stopping"][key]
            )
        checks["truth_free_stopping"] = not run_metrics["stopping"][
            "ground_truth_used_for_stopping"
        ]
        record["metric_absolute_errors"] = {}
        for key in (
            "rmse",
            "psnr",
            "ssim",
            "water_background_rmse",
            "water_background_bias",
            "data_relative_residual",
        ):
            expected, measured = ref_metrics[key], run_metrics[key]
            if expected is None or measured is None:
                raise ValueError(f"baseline metric {key} must be available")
            if not np.isfinite([expected, measured]).all():
                raise ValueError(f"nonfinite baseline metric {key}")
            error = abs(float(expected) - float(measured))
            record["metric_absolute_errors"][key] = error
            checks["metric." + key] = error <= metric_atol
        record["checks"] = checks
        record["passed"] = bool(all(checks.values()))
        record["failed_checks"] = [
            name for name, passed in checks.items() if not passed
        ]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="packaged baselines_64 directory"
    )
    parser.add_argument(
        "--actual",
        type=Path,
        required=True,
        help="directory containing algorithm/case_id outputs",
    )
    parser.add_argument("--case-id", default="D510022534")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--array-atol",
        type=float,
        default=1e-9,
        help="absolute sound-speed tolerance, m/s",
    )
    parser.add_argument(
        "--metric-atol",
        type=float,
        default=1e-9,
        help="absolute numerical metric tolerance",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("use a fresh comparison output path")
    rows = []
    for algorithm in ALGORITHMS:
        rows.append(
            {
                "algorithm": algorithm,
                **compare_result(
                    args.reference / algorithm / args.case_id,
                    args.actual / algorithm / args.case_id,
                    array_atol=args.array_atol,
                    metric_atol=args.metric_atol,
                ),
            }
        )
    report = {
        "passed": all(row["passed"] for row in rows),
        "array_atol_mps": args.array_atol,
        "metric_atol": args.metric_atol,
        "runtime_equality_required": False,
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.out)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
