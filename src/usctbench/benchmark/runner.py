"""Benchmark run and evaluation helpers."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import numpy as np
import os
import platform
import re
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from usctbench.benchmark.report import write_failure_report
from usctbench.core.config import coerce_bool
from usctbench.core.io import read_case_hdf5, write_result_hdf5
from usctbench.core.provenance import case_measurement_metadata
from usctbench.core.registry import get_algorithm
from usctbench.core.schema import (
    AlgorithmConfig,
    ReconstructionResult,
    ResultStatus,
    USCTCase,
)
from usctbench.viz import write_preview_png
from usctbench.metrics import compute_regional_image_metrics

_ENV_DEFAULT_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")


def _implementation_digest():
    """Fingerprint actual package sources, including uncommitted development code."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".m"} and path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_algorithm_config(path: str | Path) -> AlgorithmConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("algorithm config must be a YAML mapping")
    parameters = payload.get("parameters", {})
    metadata = payload.get("metadata", {})
    if not isinstance(parameters, dict):
        raise ValueError("config parameters must be a mapping")
    if not isinstance(metadata, dict):
        raise ValueError("config metadata must be a mapping")
    from usctbench.algorithms.configuration import validate_algorithm_config

    unknown = payload.keys() - {
        "name",
        "algorithm",
        "parameters",
        "metadata",
        "run_controls",
        "budget_caps",
    }
    if unknown:
        raise ValueError(f"unknown algorithm config fields: {sorted(unknown)}")
    if (
        payload.get("name")
        and payload.get("algorithm")
        and payload["name"] != payload["algorithm"]
    ):
        raise ValueError("conflicting name/algorithm aliases")
    config = AlgorithmConfig(
        name=payload.get("name") or payload.get("algorithm"),
        parameters=_expand_config_value(parameters),
        metadata=_expand_config_value(metadata),
        run_controls=payload.get("run_controls"),
        budget_caps=payload.get("budget_caps"),
    )
    if config.name:
        return validate_algorithm_config(config.name, config)
    return config


def run_algorithm_case(
    algorithm_name: str,
    case_path: str | Path,
    config_path: str | Path,
    out_root: str | Path,
) -> Path:
    """Run one algorithm on one case and write the standard artifact set."""

    out_root = Path(out_root)
    case_id = Path(case_path).stem
    algorithm_for_report = algorithm_name
    config_for_report = str(config_path)
    memory_before_mb = _peak_memory_mb()
    implementation_before = _implementation_digest()
    resolved_config = None
    case: USCTCase | None = None
    try:
        case = read_case_hdf5(case_path)
        case_id = case.case_id
        config = load_algorithm_config(config_path)
        resolved_config = config.model_dump(mode="json")
        if config.name is not None and config.name != algorithm_name:
            raise ValueError(
                "algorithm/config mismatch: "
                f"requested {algorithm_name!r}, config declares {config.name!r}"
            )
        config.parameters.setdefault("_run_output_dir", str(out_root / case_id))
        algorithm = get_algorithm(algorithm_name)
        result = algorithm.run(case, config)
        if result.sound_speed_mps is not None and result.status == ResultStatus.SUCCESS:
            # Reporting only: the solver has already selected its final state.
            image_policy = dict(config.parameters.get("image_evaluation", {}))
            image_policy.setdefault(
                "water_speed_mps",
                config.parameters.get(
                    "reference_sound_speed_mps",
                    case.metadata.get("reference_sound_speed_mps", 1500.0),
                ),
            )
            result.metrics.update(
                compute_regional_image_metrics(
                    result.sound_speed_mps,
                    case.ground_truth.sound_speed_mps,
                    **image_policy,
                )
            )
    except Exception as exc:
        result = ReconstructionResult(
            algorithm=algorithm_for_report,
            case_id=case_id,
            runtime_s=0.0,
            status=ResultStatus.FAILED,
            failure_reason=f"{type(exc).__name__}: {exc}",
        )
    memory_after_mb = _peak_memory_mb()
    implementation_after = _implementation_digest()
    result.metrics["implementation"] = {
        "source_sha256_at_start": implementation_before,
        "source_changed_during_run": implementation_before != implementation_after,
        "source_sha256_at_finish": implementation_after,
    }
    result.metrics["resolved_config"] = resolved_config
    if case is not None:
        for key, value in _feature_qc_metrics(case).items():
            result.metrics.setdefault(key, value)

    out_dir = out_root / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    peak_memory_mb = max(memory_before_mb, memory_after_mb)
    try:
        _write_result_artifacts(
            result,
            out_dir,
            config=config_for_report,
            peak_memory_mb=peak_memory_mb,
            case=case,
        )
    except Exception as exc:
        fallback = ReconstructionResult(
            algorithm=result.algorithm,
            case_id=result.case_id,
            runtime_s=result.runtime_s,
            status=ResultStatus.FAILED,
            failure_reason=f"artifact write failed: {type(exc).__name__}: {exc}",
            metrics={"artifact_write_failed": True},
        )
        _write_result_artifacts(
            fallback,
            out_dir,
            config=config_for_report,
            peak_memory_mb=peak_memory_mb,
            case=case,
        )
    return out_dir


def evaluate_run(
    run_dir: str | Path, protocol_path: str | Path | None = None
) -> dict[str, Any]:
    """Aggregate per-case metrics into CSV and markdown reports."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    protocol = _load_yaml(protocol_path) if protocol_path is not None else {}
    records = []
    for metrics_path in sorted(run_path.rglob("metrics.json")):
        case_dir = metrics_path.parent
        metadata_path = case_dir / "metadata.yaml"
        metadata = _load_yaml(metadata_path) if metadata_path.exists() else {}
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record = {
            "case_id": metadata.get("case_id", case_dir.name),
            "case_type": metadata.get("case_type", ""),
            "benchmark_type": metadata.get("benchmark_type", ""),
            "algorithm": metadata.get("algorithm", case_dir.parent.name),
            "status": metadata.get("status", "unknown"),
            "runtime_s": metadata.get("runtime_s", ""),
            "peak_memory_mb": metadata.get("peak_memory_mb", ""),
            "failure_reason": metadata.get("failure_reason") or "",
            "feature_qc_passed": metadata.get("feature_qc_passed", ""),
            "feature_failed_qc": metadata.get("feature_failed_qc", ""),
            "failure_report_present": (case_dir / "failure_report.md").exists(),
        }
        record.update(metrics)
        record["artifacts_complete"], artifact_reasons = _artifact_check(
            case_dir, record
        )
        record["pass"], pass_reasons, fail_reasons = _assess_record(
            record, protocol, artifact_reasons
        )
        record["pass_reasons"] = "; ".join(pass_reasons)
        record["fail_reasons"] = "; ".join(fail_reasons)
        records.append(record)

    run_pass_reasons, run_fail_reasons = _assess_run_records(records, protocol)
    run_checks = {
        "passed": not run_fail_reasons,
        "pass_reasons": run_pass_reasons,
        "fail_reasons": run_fail_reasons,
    }
    run_checks_json = run_path / "benchmark_run_checks.json"
    run_checks_json.write_text(
        json.dumps(run_checks, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_csv = run_path / "benchmark_summary.csv"
    _write_summary_csv(records, summary_csv)
    report_md = run_path / "benchmark_report.md"
    provenance = (
        _load_yaml(run_path / "run_metadata.yaml")
        if (run_path / "run_metadata.yaml").exists()
        else _collect_run_provenance(run_path)
    )
    _write_benchmark_report(
        records,
        report_md,
        protocol=protocol,
        run_checks=run_checks,
        provenance=provenance,
    )
    return {
        "records": records,
        "run_checks": run_checks,
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
        "run_checks_json": str(run_checks_json),
    }


def run_benchmark_suite(suite_path: str | Path) -> dict[str, Any]:
    """Run all algorithms listed in a benchmark suite YAML."""

    suite_file = Path(suite_path)
    suite = _load_yaml(suite_file)
    suite_name = suite.get("name", suite_file.stem)
    case_glob = _expand(str(suite["case_glob"]))
    suite_cases = sorted(Path(path) for path in glob.glob(case_glob, recursive=True))
    if not suite_cases and not coerce_bool(suite.get("allow_empty", False)):
        raise ValueError(f"benchmark suite matched no cases: {case_glob}")
    output_root = Path(
        _expand(str(suite.get("outputs", {}).get("root", "runs/usctbench_runs")))
    )
    run_id = (
        suite.get("run_id")
        or f"{suite_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    _write_run_metadata(
        run_root / "run_metadata.yaml", suite_file=suite_file, suite=suite
    )

    algorithms = suite.get("algorithms", [])
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError("benchmark suite must define a non-empty algorithms list")

    for algorithm in algorithms:
        algorithm_name = algorithm["name"]
        config_path = _resolve_algorithm_config_path(algorithm["config"], suite_file)
        algorithm_case_glob = algorithm.get("case_glob")
        cases = suite_cases
        if algorithm_case_glob:
            expanded_glob = _expand(str(algorithm_case_glob))
            cases = sorted(
                Path(path) for path in glob.glob(expanded_glob, recursive=True)
            )
            if not cases and not coerce_bool(
                algorithm.get("allow_empty", suite.get("allow_empty", False))
            ):
                raise ValueError(
                    f"benchmark suite algorithm {algorithm_name} matched no cases: {expanded_glob}"
                )
        for case_path in cases:
            run_algorithm_case(
                algorithm_name, case_path, config_path, run_root / algorithm_name
            )

    evaluation = evaluate_run(run_root, suite_file)
    return {"run_root": str(run_root), "num_cases": len(suite_cases), **evaluation}


def _write_result_artifacts(
    result: ReconstructionResult,
    out_dir: Path,
    *,
    config: str,
    peak_memory_mb: float,
    case: USCTCase | None = None,
) -> None:
    preview_image = result.sound_speed_mps
    if preview_image is not None:
        preview_path = write_preview_png(preview_image, out_dir / "preview.png")
        result.artifacts.setdefault("preview", str(preview_path))
    _write_straight_ray_diagnostics(result, case, out_dir)

    result_path = write_result_hdf5(result, out_dir / "result.h5")
    (out_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    measurement_metadata = (
        case_measurement_metadata(case.metadata) if case is not None else {}
    )
    (out_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "algorithm": result.algorithm,
                "case_id": result.case_id,
                "case_type": (
                    case.metadata.get("case_type") if case is not None else None
                ),
                "benchmark_type": (
                    case.metadata.get("benchmark_type") if case is not None else None
                ),
                "feature_qc_passed": (
                    case.metadata.get("feature_qc_passed") if case is not None else None
                ),
                "feature_failed_qc": (
                    case.metadata.get("feature_failed_qc") if case is not None else None
                ),
                **measurement_metadata,
                "config": config,
                "resolved_config": result.metrics.get("resolved_config"),
                "implementation": result.metrics.get("implementation"),
                "image_evaluation": result.metrics.get("image_evaluation"),
                "error_type": _classify_failure(result.failure_reason),
                "runtime_s": result.runtime_s,
                "peak_memory_mb": peak_memory_mb,
                "status": str(result.status),
                "failure_reason": result.failure_reason,
                "result_h5": str(result_path),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if result.status != ResultStatus.SUCCESS:
        write_failure_report(
            out_dir / "failure_report.md",
            algorithm=result.algorithm,
            case_id=result.case_id,
            config=config,
            error_type=_classify_failure(result.failure_reason),
            symptom=result.failure_reason or "algorithm did not report success",
            likely_causes=[
                "schema mismatch",
                "data/geometry mismatch",
                "numerical instability",
            ],
            actions=[
                "inspect case metadata",
                "inspect sinogram/features",
                "lower relaxation or iterations",
            ],
        )
    else:
        stale_failure_report = out_dir / "failure_report.md"
        if stale_failure_report.exists():
            stale_failure_report.unlink()


def _write_straight_ray_diagnostics(
    result: ReconstructionResult, case: USCTCase | None, out_dir: Path
) -> None:
    if case is None or not str(result.algorithm).startswith("straight_"):
        return
    try:
        from usctbench.algorithms.ray import (
            ray_weights,
            valid_ray_mask,
        )
        from usctbench.operators.straight_ray import StraightRayProjector
    except Exception:
        return

    projector = StraightRayProjector.from_case(case)
    mask = valid_ray_mask(case, projector)
    weights = ray_weights(case, projector, mask)
    coverage = projector.adjoint(mask.astype(float))
    row_norm_l1 = projector.row_norms(power=1)
    row_norm_l2 = projector.row_norms(power=2)
    col_norm_l1 = projector.col_norms(power=1)

    coverage_path = write_preview_png(coverage, out_dir / "coverage.png")
    result.artifacts.setdefault("coverage", str(coverage_path))
    stats = {
        "valid_ray_count": int(np.sum(mask)),
        "total_ray_count": int(mask.size),
        "valid_ray_fraction": float(np.mean(mask)) if mask.size else 0.0,
        "ray_weight_mean": float(np.mean(weights[mask])) if np.any(mask) else 0.0,
        "ray_weight_nonzero_fraction": (
            float(np.mean(weights[mask] > 0.0)) if np.any(mask) else 0.0
        ),
        "coverage": _array_stats(coverage),
        "coverage_nonzero_fraction": float(np.mean(coverage > 0.0)),
        "row_norm_l1": _array_stats(row_norm_l1[mask] if np.any(mask) else row_norm_l1),
        "row_norm_l2": _array_stats(row_norm_l2[mask] if np.any(mask) else row_norm_l2),
        "col_norm_l1": _array_stats(col_norm_l1),
    }
    if (
        result.sound_speed_mps is not None
        and case.ground_truth.sound_speed_mps is not None
    ):
        error = np.asarray(result.sound_speed_mps, dtype=float) - np.asarray(
            case.ground_truth.sound_speed_mps, dtype=float
        )
        stats["abs_error_coverage_corr"] = _finite_corr(np.abs(error), coverage)
        stats["ring_artifact_index"] = _radial_artifact_index(error, case.grid.roi_mask)
        result.metrics["coverage_abs_error_corr"] = stats["abs_error_coverage_corr"]
        result.metrics["ring_artifact_index"] = stats["ring_artifact_index"]
    result.metrics["coverage_nonzero_fraction"] = stats["coverage_nonzero_fraction"]
    result.metrics["valid_ray_fraction"] = stats["valid_ray_fraction"]
    result.metrics.setdefault("ray_weight_mean", stats["ray_weight_mean"])
    result.metrics.setdefault(
        "ray_weight_nonzero_fraction", stats["ray_weight_nonzero_fraction"]
    )
    result.metrics["row_norm_l1_min"] = stats["row_norm_l1"]["min"]
    result.metrics["row_norm_l1_max"] = stats["row_norm_l1"]["max"]
    result.metrics["col_norm_l1_min"] = stats["col_norm_l1"]["min"]
    result.metrics["col_norm_l1_max"] = stats["col_norm_l1"]["max"]

    stats_path = out_dir / "coverage_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    result.artifacts.setdefault("coverage_stats", str(stats_path))
    if "residual_curve" in result.metrics:
        curve_path = out_dir / "residual_curve.json"
        curve_path.write_text(
            json.dumps(result.metrics["residual_curve"], indent=2), encoding="utf-8"
        )
        result.artifacts.setdefault("residual_curve", str(curve_path))


def _array_stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _feature_qc_metrics(case: USCTCase) -> dict[str, Any]:
    qc = case.metadata.get("feature_qc", {})
    if not isinstance(qc, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key, value in qc.items():
        if isinstance(value, (int, float, bool, str)) or value is None:
            metrics[key] = value
    return metrics


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) < 3:
        return math.nan
    x = x[finite] - float(np.mean(x[finite]))
    y = y[finite] - float(np.mean(y[finite]))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return 0.0
    return float(np.dot(x, y) / denom)


def _radial_artifact_index(error: np.ndarray, roi_mask: np.ndarray | None) -> float:
    image = np.asarray(error, dtype=float)
    yy, xx = np.indices(image.shape, dtype=float)
    center_y = 0.5 * (image.shape[0] - 1)
    center_x = 0.5 * (image.shape[1] - 1)
    radius = np.rint(np.hypot(yy - center_y, xx - center_x)).astype(int)
    mask = np.isfinite(image)
    if roi_mask is not None:
        mask &= np.asarray(roi_mask, dtype=bool)
    if not np.any(mask):
        return math.nan
    values = np.abs(image)
    radial_means = []
    for ridx in np.unique(radius[mask]):
        shell = mask & (radius == ridx)
        if int(np.sum(shell)) >= 4:
            radial_means.append(float(np.mean(values[shell])))
    if len(radial_means) < 3:
        return 0.0
    denom = float(np.std(values[mask]))
    if denom == 0.0:
        return 0.0
    return float(np.std(np.asarray(radial_means, dtype=float)) / denom)


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _expand(value: str) -> str:
    def _replace_default(match: re.Match[str]) -> str:
        env_value = os.environ.get(match.group(1))
        return env_value if env_value not in (None, "") else match.group(2)

    value = _ENV_DEFAULT_RE.sub(_replace_default, value)
    return os.path.expandvars(os.path.expanduser(value))


def _resolve_algorithm_config_path(config: Any, suite_file: Path) -> Path:
    config_path = Path(_expand(str(config)))
    if config_path.is_absolute():
        return config_path

    suite_candidate = suite_file.parent / config_path
    if suite_candidate.exists():
        return suite_candidate

    root_candidate = _repo_root() / config_path
    if root_candidate.exists():
        return root_candidate

    cwd_candidate = Path.cwd() / config_path
    if cwd_candidate.exists():
        return cwd_candidate

    if config_path.parts and config_path.parts[0] == "configs":
        return root_candidate
    return suite_candidate


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _expand_config_value(value: Any) -> Any:
    if isinstance(value, str):
        return _expand(value)
    if isinstance(value, list):
        return [_expand_config_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_config_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_config_value(item) for key, item in value.items()}
    return value


def _assess_record(
    record: dict[str, Any],
    protocol: dict[str, Any],
    artifact_reasons: list[str],
) -> tuple[bool, list[str], list[str]]:
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    if str(record.get("status", "")).lower() != "success":
        fail_reasons.append(f"status is {record.get('status')}")
        if record.get("failure_reason"):
            fail_reasons.append(str(record["failure_reason"]))
    else:
        pass_reasons.append("status is success")

    if artifact_reasons:
        fail_reasons.extend(artifact_reasons)
    else:
        pass_reasons.append("required artifacts present")

    algorithm_name = str(record.get("algorithm", ""))
    if record.get("stopping", {}).get("termination_category") == "failure":
        fail_reasons.append(
            f"solver stopped unsuccessfully: {record.get('stop_reason')}"
        )

    for key in _required_metrics_for_algorithm(
        protocol.get("required_metrics", []), algorithm_name
    ):
        if key not in record or not _is_number(record[key]):
            fail_reasons.append(f"missing required metric {key}")

    thresholds = _metric_limits_for_algorithm(
        protocol.get("thresholds", {}), algorithm_name
    )
    if isinstance(thresholds, dict):
        for key, limit in thresholds.items():
            if key not in record or not _is_number(record[key]):
                fail_reasons.append(f"missing threshold metric {key}")
                continue
            value = float(record[key])
            if value > float(limit):
                fail_reasons.append(f"{key}={value:g} exceeds max {float(limit):g}")
            else:
                pass_reasons.append(f"{key}={value:g} <= {float(limit):g}")

    minimums = _metric_limits_for_algorithm(
        protocol.get("minimums", {}), algorithm_name
    )
    if isinstance(minimums, dict):
        for key, limit in minimums.items():
            if key not in record or not _is_number(record[key]):
                fail_reasons.append(f"missing minimum metric {key}")
                continue
            value = float(record[key])
            if value < float(limit):
                fail_reasons.append(f"{key}={value:g} below min {float(limit):g}")
            else:
                pass_reasons.append(f"{key}={value:g} >= {float(limit):g}")

    return not fail_reasons, pass_reasons, fail_reasons


def _assess_run_records(
    records: list[dict[str, Any]], protocol: dict[str, Any]
) -> tuple[list[str], list[str]]:
    pass_reasons: list[str] = []
    fail_reasons: list[str] = []

    if records:
        pass_reasons.append(f"{len(records)} benchmark records present")
    else:
        fail_reasons.append("no benchmark records present")

    min_records = protocol.get("min_records")
    if min_records is not None:
        if len(records) < int(min_records):
            fail_reasons.append(
                f"record count {len(records)} below required {int(min_records)}"
            )
        else:
            pass_reasons.append(f"record count {len(records)} >= {int(min_records)}")

    observed_algorithms = {str(record.get("algorithm", "")) for record in records}
    expected_algorithms = _expected_algorithms(protocol)
    missing_algorithms = sorted(expected_algorithms - observed_algorithms)
    if expected_algorithms:
        if missing_algorithms:
            fail_reasons.append(f"missing algorithms: {', '.join(missing_algorithms)}")
        else:
            pass_reasons.append("all expected algorithms present")

    observed_cases = {str(record.get("case_id", "")) for record in records}
    min_cases = protocol.get("min_cases")
    if min_cases is not None:
        if len(observed_cases) < int(min_cases):
            fail_reasons.append(
                f"case count {len(observed_cases)} below required {int(min_cases)}"
            )
        else:
            pass_reasons.append(f"case count {len(observed_cases)} >= {int(min_cases)}")

    if (
        expected_algorithms
        and observed_cases
        and coerce_bool(protocol.get("require_algorithm_case_matrix", True))
    ):
        observed_pairs = {
            (str(record.get("algorithm", "")), str(record.get("case_id", "")))
            for record in records
        }
        missing_pairs = [
            f"{algorithm}/{case_id}"
            for algorithm in sorted(expected_algorithms)
            for case_id in sorted(observed_cases)
            if (algorithm, case_id) not in observed_pairs
        ]
        if missing_pairs:
            fail_reasons.append(
                f"missing algorithm-case records: {', '.join(missing_pairs)}"
            )
        else:
            pass_reasons.append("complete algorithm-case matrix present")

    allowed_statuses = protocol.get("expected_statuses")
    if isinstance(allowed_statuses, list) and allowed_statuses:
        allowed = {str(status).lower() for status in allowed_statuses}
        bad = [
            f"{record.get('algorithm')}/{record.get('case_id')}={record.get('status')}"
            for record in records
            if str(record.get("status", "")).lower() not in allowed
        ]
        if bad:
            fail_reasons.append(f"unexpected statuses: {', '.join(bad)}")
        else:
            pass_reasons.append("all statuses are expected")

    return pass_reasons, fail_reasons


def _expected_algorithms(protocol: dict[str, Any]) -> set[str]:
    explicit = protocol.get("expected_algorithms")
    if isinstance(explicit, list):
        return {str(value) for value in explicit}
    algorithms = protocol.get("algorithms", [])
    if isinstance(algorithms, list):
        return {
            str(entry.get("name"))
            for entry in algorithms
            if isinstance(entry, dict) and entry.get("name")
        }
    return set()


def _metric_limits_for_algorithm(spec: Any, algorithm_name: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        return {}
    if any(isinstance(value, dict) for value in spec.values()):
        limits: dict[str, Any] = {}
        default_limits = spec.get("default", {})
        algorithm_limits = spec.get(algorithm_name, {})
        if isinstance(default_limits, dict):
            limits.update(default_limits)
        if isinstance(algorithm_limits, dict):
            limits.update(algorithm_limits)
        return limits
    return spec


def _required_metrics_for_algorithm(spec: Any, algorithm_name: str) -> list[str]:
    if isinstance(spec, list):
        return [str(item) for item in spec]
    if not isinstance(spec, dict):
        return []
    required: list[str] = []
    for key in ("default", algorithm_name):
        values = spec.get(key, [])
        if isinstance(values, list):
            required.extend(str(item) for item in values)
    return sorted(set(required))


def _artifact_check(case_dir: Path, record: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    required = {
        "result.h5": case_dir / "result.h5",
        "metrics.json": case_dir / "metrics.json",
        "metadata.yaml": case_dir / "metadata.yaml",
    }
    for label, path in required.items():
        if not path.exists():
            reasons.append(f"missing {label}")
    if (
        str(record.get("status", "")).lower() == "success"
        and not (case_dir / "preview.png").exists()
    ):
        reasons.append("missing preview.png for successful run")
    if (
        str(record.get("status", "")).lower() != "success"
        and not (case_dir / "failure_report.md").exists()
    ):
        reasons.append("missing failure_report.md for non-success run")
    return not reasons, reasons


def _write_summary_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for record in records for key in record})
    preferred = [
        "algorithm",
        "case_id",
        "case_type",
        "benchmark_type",
        "status",
        "pass",
        "pass_reasons",
        "fail_reasons",
        "runtime_s",
        "primary_image_region",
        "rmse",
        "psnr",
        "ssim",
        "water_background_rmse",
        "peak_memory_mb",
        "artifacts_complete",
        "failure_report_present",
    ]
    fieldnames = preferred + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _write_benchmark_report(
    records: list[dict[str, Any]],
    path: Path,
    *,
    protocol: dict[str, Any],
    run_checks: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    passed = sum(1 for record in records if record.get("pass"))
    runtimes = [
        float(record["runtime_s"])
        for record in records
        if _is_number(record.get("runtime_s"))
    ]
    memory_values = [
        float(record["peak_memory_mb"])
        for record in records
        if _is_number(record.get("peak_memory_mb"))
    ]
    lines = [
        "# Benchmark report",
        "",
        f"- Records: {len(records)}",
        f"- Passed: {passed}",
        f"- Failed: {len(records) - passed}",
        f"- Protocol: {protocol.get('name', 'ad hoc')}",
        f"- Run checks: {'passed' if run_checks.get('passed') else 'failed'}",
        f"- Git commit: {provenance.get('git', {}).get('commit', 'unknown')}",
        f"- Git branch: {provenance.get('git', {}).get('branch', 'unknown')}",
        f"- Hostname: {provenance.get('host', {}).get('hostname', 'unknown')}",
        f"- Python: {provenance.get('python', {}).get('version', 'unknown')}",
        f"- USCT_DATA_ROOT: {provenance.get('environment', {}).get('USCT_DATA_ROOT', '')}",
        f"- USCT_SAMPLE_ROOT: {provenance.get('environment', {}).get('USCT_SAMPLE_ROOT', '')}",
        f"- USCT_RUN_ROOT: {provenance.get('environment', {}).get('USCT_RUN_ROOT', '')}",
        (
            f"- Runtime total seconds: {sum(runtimes):.6g}"
            if runtimes
            else "- Runtime total seconds:"
        ),
        (
            f"- Runtime max seconds: {max(runtimes):.6g}"
            if runtimes
            else "- Runtime max seconds:"
        ),
        (
            f"- Peak memory max MB: {max(memory_values):.6g}"
            if memory_values
            else "- Peak memory max MB:"
        ),
        "",
        "## Run checks",
        "",
    ]
    for reason in run_checks.get("fail_reasons", []):
        lines.append(f"- FAIL: {reason}")
    for reason in run_checks.get("pass_reasons", []):
        lines.append(f"- PASS: {reason}")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "Image scores use each row's declared region. Do not mix historical full-image and tissue scores.",
            "",
            "| Algorithm | Case | Region | RMSE (m/s) | PSNR (dB) | SSIM | Water RMSE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:

        def score(name):
            value = record.get(name)
            return f"{value:.5g}" if _is_number(value) else "N/A"

        lines.append(
            f"| {record.get('algorithm')} | {record.get('case_id')} | "
            f"{record.get('primary_image_region', 'legacy/unspecified')} | "
            f"{score('rmse')} | {score('psnr')} | {score('ssim')} | {score('water_background_rmse')} |"
        )
    lines.extend(["", "### Execution", ""])
    for record in records:
        lines.append(
            f"- `{record.get('algorithm')}` / `{record.get('case_id')}`: "
            f"case_type={record.get('case_type') or 'unknown'}, "
            f"benchmark_type={record.get('benchmark_type') or 'unknown'}, "
            f"status={record.get('status')}, pass={record.get('pass')}, runtime_s={record.get('runtime_s')}, "
            f"reasons={record.get('fail_reasons') or record.get('pass_reasons')}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_run_metadata(path: Path, *, suite_file: Path, suite: dict[str, Any]) -> None:
    metadata = _collect_run_provenance(path.parent)
    metadata["suite"] = {
        "path": str(suite_file),
        "name": suite.get("name", suite_file.stem),
        "case_glob": suite.get("case_glob"),
        "algorithms": [
            entry.get("name")
            for entry in suite.get("algorithms", [])
            if isinstance(entry, dict)
        ],
    }
    path.write_text(yaml.safe_dump(metadata, sort_keys=True), encoding="utf-8")


def _collect_run_provenance(run_path: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_path": str(run_path),
        "git": {
            "commit": _run_text(["git", "rev-parse", "HEAD"]),
            "branch": _run_text(["git", "branch", "--show-current"]),
            "remote_origin": _run_text(["git", "remote", "get-url", "origin"]),
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "USCT_WORKSPACE",
                "USCT_DATA_ROOT",
                "USCT_SAMPLE_ROOT",
                "USCT_RUN_ROOT",
            )
        },
        "runtime": {
            "torch": _torch_info(),
            "nvidia_smi": _run_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                timeout_s=5,
            ),
        },
    }


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional runtime.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _run_text(command: list[str], *, timeout_s: int = 2) -> str:
    try:
        proc = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        return f"unavailable: {proc.stderr.strip() or proc.stdout.strip()}"
    return proc.stdout.strip()


def _classify_failure(reason: str | None) -> str:
    text = (reason or "").lower()
    if any(
        token in text
        for token in (
            "matlab",
            "dependency",
            "modulenotfounderror",
            "importerror",
            "not installed",
            "executable",
        )
    ):
        return "external-dependency"
    if any(
        token in text
        for token in ("pydantic", "validation", "schema", "model", "field required")
    ):
        return "schema"
    if any(
        token in text
        for token in (
            "hdf5",
            "h5py",
            "file not found",
            "no such file",
            "case",
            "dataset",
            "path",
        )
    ):
        return "data"
    if any(
        token in text
        for token in (
            "nan",
            "inf",
            "singular",
            "overflow",
            "underflow",
            "ill-conditioned",
            "non-finite",
        )
    ):
        return "numerical"
    if any(
        token in text
        for token in ("converge", "diverge", "residual increased", "line search")
    ):
        return "convergence"
    return "unknown"


def _peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if usage > 10_000_000:
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)
