from __future__ import annotations

import h5py
import numpy as np
import pytest

from usctbench.algorithms.fwi.adapter import (
    KWaveFWIAdapterAlgorithm,
    _iteration_stack,
    read_kwave_fwi_result,
)
from usctbench.algorithms.fwi.tiny import TinyFWIAlgorithm
from usctbench.core.schema import AlgorithmConfig, ResultStatus
from usctbench.algorithms.configuration import validate_algorithm_config
from usctbench.metrics import compute_baseline_improvement_metrics


@pytest.mark.parametrize("override", [None, 1510.0])
def test_external_baseline_uses_case_reference_unless_explicit(
    synthetic_case, tmp_path, override
):
    synthetic_case.metadata["reference_sound_speed_mps"] = 1470.0
    path = tmp_path / "baseline.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "VEL_ESTIM", data=np.full(synthetic_case.grid.shape, 1490.0)
        )
    config = AlgorithmConfig(
        parameters={"result_path": str(path), "baseline_sound_speed_mps": override}
    )
    resolved = validate_algorithm_config("fwi_kwave_adapter", config)
    assert validate_algorithm_config("fwi_kwave_adapter", resolved) == resolved
    if override is None:
        assert "baseline_sound_speed_mps" not in resolved.parameters
    result = KWaveFWIAdapterAlgorithm().run(synthetic_case, resolved)
    assert result.status == ResultStatus.SUCCESS
    expected = compute_baseline_improvement_metrics(
        result.sound_speed_mps,
        synthetic_case.ground_truth.sound_speed_mps,
        1470.0 if override is None else override,
        mask=synthetic_case.grid.roi_mask,
    )
    assert result.metrics["water_baseline_rmse"] == expected["water_baseline_rmse"]


def test_tiny_fwi_loss_decreases(synthetic_case):
    result = TinyFWIAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(parameters={"steps": 5, "learning_rate": 1.0e6}),
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.metrics["loss_decreased"] is True


def test_kwave_adapter_ingests_result_file(synthetic_case, tmp_path):
    result_path = tmp_path / "fwi_result.mat"
    with h5py.File(result_path, "w") as handle:
        handle.create_dataset("VEL_ESTIM", data=np.full((12, 12), 1490.0))
        handle.create_dataset(
            "C_INTERP", data=np.asarray(synthetic_case.ground_truth.sound_speed_mps)
        )
        handle.create_dataset("LOSS_ITER", data=np.array([3.0, 2.0, 1.0]))

    external = read_kwave_fwi_result(result_path)
    result = KWaveFWIAdapterAlgorithm().run(
        synthetic_case, AlgorithmConfig(parameters={"result_path": str(result_path)})
    )

    assert external["sound_speed_mps"].shape == (12, 12)
    assert result.status == ResultStatus.SUCCESS
    assert result.metrics["external_result_loaded"] is True
    assert result.sound_speed_mps is not None


def test_kwave_adapter_skips_missing_result(synthetic_case, tmp_path):
    result = KWaveFWIAdapterAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(parameters={"result_path": str(tmp_path / "missing.mat")}),
    )

    assert result.status == ResultStatus.SKIPPED


def test_kwave_adapter_treats_string_false_run_external_as_loader_only(
    synthetic_case, tmp_path
):
    result_path = tmp_path / "fwi_result.mat"
    with h5py.File(result_path, "w") as handle:
        handle.create_dataset("VEL_ESTIM", data=np.full((12, 12), 1490.0))
        handle.create_dataset("LOSS_ITER", data=np.array([1.0]))

    result = KWaveFWIAdapterAlgorithm().run(
        synthetic_case,
        AlgorithmConfig(
            parameters={"result_path": str(result_path), "run_external": "false"}
        ),
    )

    assert result.status == ResultStatus.SUCCESS
    assert result.failure_reason is None
    assert result.metrics["external_result_loaded"] is True


def test_iteration_stack_uses_iteration_count_to_select_axis():
    matlab_stack = np.zeros((5, 5, 7))
    matlab_stack[:, :, 3] = 3.0
    normalized = _iteration_stack(matlab_stack, iterations=7)

    assert normalized is not None
    assert normalized.shape == (7, 5, 5)
    assert float(normalized[3, 0, 0]) == 3.0

    python_stack = np.zeros((7, 5, 5))
    python_stack[4, :, :] = 4.0
    normalized = _iteration_stack(python_stack, iterations=7)

    assert normalized is not None
    assert normalized.shape == (7, 5, 5)
    assert float(normalized[4, 0, 0]) == 4.0
