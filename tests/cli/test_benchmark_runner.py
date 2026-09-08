from __future__ import annotations

import yaml

from usctbench.benchmark.runner import run_algorithm_case, run_benchmark_suite
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import write_case_hdf5


def test_line_search_failure_is_not_a_passing_benchmark():
    from usctbench.benchmark.runner import _assess_record

    passed, _, failures = _assess_record(
        {
            "status": "success",
            "algorithm": "rwave_adapter",
            "stop_reason": "line_search_failed",
            "stopping": {"termination_category": "failure"},
        },
        {},
        [],
    )
    assert not passed
    assert any("unsuccessfully" in reason for reason in failures)


def test_run_algorithm_case_writes_standard_artifacts(synthetic_case, tmp_path):
    register_builtin_algorithms()
    case_path = tmp_path / "case.h5"
    config_path = tmp_path / "cgls.yaml"
    write_case_hdf5(synthetic_case, case_path)
    config_path.write_text(
        "name: straight_cgls\nparameters:\n  iterations: 2\n", encoding="utf-8"
    )

    out_dir = run_algorithm_case(
        "straight_cgls", case_path, config_path, tmp_path / "runs"
    )

    assert (out_dir / "result.h5").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "metadata.yaml").exists()
    assert (out_dir / "preview.png").exists()


def test_run_algorithm_case_rejects_config_algorithm_mismatch(synthetic_case, tmp_path):
    register_builtin_algorithms()
    case_path = tmp_path / "case.h5"
    config_path = tmp_path / "sirt.yaml"
    write_case_hdf5(synthetic_case, case_path)
    config_path.write_text(
        "name: straight_sirt\nparameters:\n  iterations: 2\n", encoding="utf-8"
    )

    out_dir = run_algorithm_case(
        "straight_cgls", case_path, config_path, tmp_path / "runs"
    )
    metadata = yaml.safe_load((out_dir / "metadata.yaml").read_text())

    assert metadata["status"] == "failed"
    assert "algorithm/config mismatch" in metadata["failure_reason"]


def test_benchmark_suite_runs_synthetic_case(synthetic_case, tmp_path):
    register_builtin_algorithms()
    case_dir = tmp_path / "cases"
    case_dir.mkdir()
    write_case_hdf5(synthetic_case, case_dir / "case.h5")
    config_path = tmp_path / "sirt.yaml"
    config_path.write_text(
        "name: straight_sirt\nparameters:\n  iterations: 2\n", encoding="utf-8"
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "test_suite",
                "case_glob": str(case_dir / "*.h5"),
                "outputs": {"root": str(tmp_path / "runs")},
                "algorithms": [{"name": "straight_sirt", "config": str(config_path)}],
                "min_cases": 1,
                "min_records": 1,
                "expected_statuses": ["success"],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(suite_path)

    assert result["run_checks"]["passed"] is True
    assert result["records"]


def test_benchmark_suite_resolves_config_relative_to_suite_file(
    synthetic_case, tmp_path
):
    register_builtin_algorithms()
    case_dir = tmp_path / "cases"
    config_dir = tmp_path / "configs"
    case_dir.mkdir()
    config_dir.mkdir()
    write_case_hdf5(synthetic_case, case_dir / "case.h5")
    (config_dir / "sirt.yaml").write_text(
        "name: straight_sirt\nparameters:\n  iterations: 1\n", encoding="utf-8"
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "relative_suite",
                "case_glob": str(case_dir / "*.h5"),
                "outputs": {"root": str(tmp_path / "runs")},
                "algorithms": [
                    {"name": "straight_sirt", "config": "configs/sirt.yaml"}
                ],
                "min_cases": 1,
                "min_records": 1,
                "expected_statuses": ["success"],
            }
        ),
        encoding="utf-8",
    )

    result = run_benchmark_suite(suite_path)

    assert result["run_checks"]["passed"] is True
