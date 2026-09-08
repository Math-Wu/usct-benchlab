from __future__ import annotations

import yaml
import json
import h5py
import numpy as np

from usctbench.benchmark.runner import run_algorithm_case, run_benchmark_suite
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import write_case_hdf5


def test_no_ground_truth_keeps_holdout_scores_without_image_scores(
    synthetic_case, tmp_path
):
    register_builtin_algorithms()
    case = synthetic_case.model_copy(deep=True)
    case.ground_truth.sound_speed_mps = None
    case_path = write_case_hdf5(case, tmp_path / "no_gt.h5")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "name": "straight_cgls",
                "parameters": {
                    "stopping": {"max_iterations": 3},
                    "evaluation": {"receiver_fraction": 0.125, "seed": 42},
                },
            }
        )
    )
    out = run_algorithm_case("straight_cgls", case_path, cfg, tmp_path / "run")
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["rmse"] is metrics["psnr"] is metrics["ssim"] is None
    assert metrics["image_evaluation"]["status"] == "ground_truth_unavailable"
    assert metrics["evaluation"]["receiver"]["weighted_relative_residual"] >= 0
    metadata = yaml.safe_load((out / "metadata.yaml").read_text())
    assert metadata["image_evaluation"] == metrics["image_evaluation"]


def test_image_region_changes_only_reporting_not_reconstruction(
    synthetic_case, tmp_path
):
    register_builtin_algorithms()
    case_path = write_case_hdf5(synthetic_case, tmp_path / "case.h5")
    outputs = []
    for region in ("tissue", "full_image"):
        cfg = tmp_path / f"{region}.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "name": "straight_cgls",
                    "parameters": {
                        "stopping": {"max_iterations": 3},
                        "image_evaluation": {"primary_region": region},
                    },
                }
            )
        )
        outputs.append(
            run_algorithm_case("straight_cgls", case_path, cfg, tmp_path / region)
        )
    with (
        h5py.File(outputs[0] / "result.h5") as a,
        h5py.File(outputs[1] / "result.h5") as b,
    ):
        np.testing.assert_array_equal(
            a["sound_speed_mps"][()], b["sound_speed_mps"][()]
        )
    a, b = [json.loads((p / "metrics.json").read_text()) for p in outputs]
    assert a["rmse"] == a["tissue_rmse"]
    assert b["rmse"] == b["full_image_rmse"]
    # Timing is naturally different; compare the numerical history only.
    for row_a, row_b in zip(a["iteration_history"], b["iteration_history"]):
        for key in ("iteration", "residual_norm", "objective"):
            assert row_a[key] == row_b[key]
    assert a["stop_reason"] == b["stop_reason"]


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
