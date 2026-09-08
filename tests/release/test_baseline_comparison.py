import json
import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

compare = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/check_64_baselines.py")
)["compare_result"]


@pytest.fixture
def pair(tmp_path):
    metrics = {
        "evaluation_split": {"seed": 42, "mask_sha256": "same"},
        "image_evaluation": {"mask_sha256": "tissue"},
        "primary_image_region": "tissue",
        "stop_reason": "max_iterations",
        "stopping": {
            "reason": "max_iterations",
            "completed_iterations": 5,
            "selected_iteration": 4,
            "triggered_rules": ["max_iterations"],
            "ground_truth_used_for_stopping": False,
        },
    }
    for key in (
        "rmse",
        "psnr",
        "ssim",
        "water_background_rmse",
        "water_background_bias",
        "data_relative_residual",
    ):
        metrics[key] = 0.25
    paths = tmp_path / "reference", tmp_path / "actual"
    for path in paths:
        path.mkdir()
        with h5py.File(path / "result.h5", "w") as f:
            f["sound_speed_mps"] = np.full((3, 3), 1500.0)
        (path / "metrics.json").write_text(json.dumps(metrics))
        (path / "metadata.yaml").write_text(
            yaml.safe_dump(
                {
                    "status": "success",
                    "case_id": "case",
                    "algorithm": "straight_cgls",
                    "runtime_s": 1,
                }
            )
        )
    return paths


def test_identical_baseline_and_different_runtime_pass(pair):
    a, b = pair
    m = yaml.safe_load((b / "metadata.yaml").read_text())
    m["runtime_s"] = 100
    (b / "metadata.yaml").write_text(yaml.safe_dump(m))
    result = compare(a, b)
    assert result["passed"] and result["array_max_absolute_error_mps"] == 0
    assert len(result["actual_sha256"]["result.h5"]) == 64


@pytest.mark.parametrize("value", [1499.0, np.nan, np.inf])
def test_changed_or_nonfinite_array_fails(pair, value):
    a, b = pair
    with h5py.File(b / "result.h5", "a") as f:
        f["sound_speed_mps"][0, 0] = value
    assert not compare(a, b)["passed"]


@pytest.mark.parametrize("key", ["evaluation_split", "image_evaluation", "stop_reason"])
def test_policy_changes_cannot_pass_on_array_equality(pair, key):
    a, b = pair
    m = json.loads((b / "metrics.json").read_text())
    m[key] = "changed"
    (b / "metrics.json").write_text(json.dumps(m))
    result = compare(a, b)
    assert not result["passed"] and key in result["failed_checks"]


def test_selected_iteration_change_is_detected(pair):
    a, b = pair
    m = json.loads((b / "metrics.json").read_text())
    m["stopping"]["selected_iteration"] = 5
    (b / "metrics.json").write_text(json.dumps(m))
    assert not compare(a, b)["passed"]


def test_failed_result_and_missing_file_are_failures(pair):
    a, b = pair
    m = yaml.safe_load((b / "metadata.yaml").read_text())
    m["status"] = "failed"
    (b / "metadata.yaml").write_text(yaml.safe_dump(m))
    assert not compare(a, b)["passed"]
    (b / "result.h5").unlink()
    assert "FileNotFoundError" in compare(a, b)["error"]


def test_changed_metrics_and_invalid_tolerance(pair):
    a, b = pair
    m = json.loads((b / "metrics.json").read_text())
    m["ssim"] += 0.01
    (b / "metrics.json").write_text(json.dumps(m))
    assert "metric.ssim" in compare(a, b)["failed_checks"]
    with pytest.raises(ValueError, match="tolerance"):
        compare(a, b, array_atol=-1)
