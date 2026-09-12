"""Hardware-free WUST orchestration; not a substitute for the CUDA gate."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from usctbench.algorithms.fwi import algorithm as module
from usctbench.algorithms.fwi.io import (
    array_hash,
    frequency_input,
    read_artifact,
    write_artifact,
)
from usctbench.algorithms.fwi.parameters import WUSTParameters
from usctbench.algorithms.fwi.runtime import WUST_SHA, WUST_VERSION, WUST_MANIFEST
from usctbench.core.algorithm_specs import make_agent_config, SPECS
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)


@pytest.fixture
def wave_case():
    return USCTCase(
        case_id="asymmetric",
        grid=GridSpec(
            shape=(23, 27), spacing_m=(0.001, 0.0012), origin_m=(-0.011, -0.014)
        ),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=[[-0.004, -0.007], [0.006, 0.007]],
            rx_pos_m=[
                [-0.003, 0.005],
                [0.007, -0.005],
                [0.004, 0.011],
                [-0.006, 0.009],
            ],
        ),
        measurement=MeasurementSpec(
            domain="frequency",
            frequencies_hz=[120000.0, 100000.0],
            freq_data=np.arange(16).reshape(2, 2, 4).astype(complex) * (1 + 2j),
            valid_mask=[[True, False, True, True], [True, True, True, True]],
        ),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "pressure_contract": {
                "fourier_sign": 1,
                "real_pressure": True,
                "pressure_type": "total_pressure",
                "data_units": "unknown",
                "spectrum_normalization": "dtft_dt",
            },
        },
        ground_truth=GroundTruthSpec(sound_speed_mps=np.full((23, 27), 1510.0)),
    )


@pytest.fixture
def config(tmp_path):
    return AlgorithmConfig(
        name="fwi_wust",
        parameters={
            "sound_speed_bounds_mps": [1400.0, 1700.0],
            "pml_m": 0.003,
            "source_batch_size": 3,
            "frequency_schedule_hz": [100000.0, 100000.0, 120000.0],
            "scratch_root": str(tmp_path),
        },
        run_controls={"max_elapsed_s": 60.0},
    )


class FakeClient:
    """Deterministic protocol stub. Canonicalization here is test-only."""

    requests = []
    provenance = {
        "git_sha": WUST_SHA,
        "runtime_version": WUST_VERSION,
        "dirty": False,
        "source_manifest_verified": True,
        "source_manifest_sha256": WUST_MANIFEST,
    }

    def __init__(self, p, deadline, folder):
        self.p, self.deadline, self.folder = p, deadline, folder

    def admit(self):
        return {"role": "production", "test_stub": True}

    def invoke(self, req):
        self.deadline.remaining()
        self.requests.append(copy.deepcopy(req))
        if req["operation"] == "ingest_frequency":
            doc, a = read_artifact(req["input_manifest"], "wust.frequency_input")
            order = np.argsort(a["frequencies_hz"])
            p = a["pressure"]
            if doc["metadata"]["fourier_sign"] == 1:
                p = p.conj()
            p = p[order].transpose(1, 2, 0)
            m = np.broadcast_to(a["mask"], a["pressure"].shape)[order].transpose(
                1, 2, 0
            )
            p = np.where(m, p, 0)
            return write_artifact(
                req["output_manifest"],
                "wust.measurements",
                {"fourier_sign": -1, "runtime_provenance": self.provenance},
                {"Y": p, "mask": m, "frequencies_hz": a["frequencies_hz"][order]},
                {
                    "Y": "tx,rx,frequency",
                    "mask": "tx,rx,frequency",
                    "frequencies_hz": "frequency",
                },
                {"Y": "unknown", "mask": "1", "frequencies_hz": "Hz"},
            )
        _, a = read_artifact(req["initial_manifest"], "wust.initial_model")
        n = len(req["schedule"])
        c = a["initial_mps"].copy() + a["update_mask"] * n
        reason = (
            "zero_updates"
            if not n
            else (
                "caller_truncated_schedule"
                if n < req["planned_schedule_length"]
                else "schedule_complete"
            )
        )
        meta = {
            "runtime_provenance": self.provenance,
            "completed_updates": n,
            "executed_schedule": req["schedule"],
            "completion": {"reason": reason, "converged": False},
            "environment": {"backend": self.p.backend},
            "records": [{"loss_before_update": 1.0} for _ in range(n)],
            "final_data_residual": None,
        }
        return write_artifact(
            req["output_manifest"],
            "wust.reconstruction",
            meta,
            {"c_mps": c},
            {"c_mps": "y,x"},
            {"c_mps": "m/s"},
        )


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    FakeClient.requests = []
    monkeypatch.setattr(module, "WUSTClient", FakeClient)


@pytest.mark.parametrize(
    "count,cap,expected,reason",
    [
        (None, None, 3, "schedule_complete"),
        (9, None, 3, "schedule_complete"),
        (2, None, 2, "max_iterations"),
        (None, 1, 1, "max_iterations"),
        (0, None, 0, "max_iterations"),
    ],
)
def test_schedule_and_budget_mapping(wave_case, config, count, cap, expected, reason):
    config.run_controls.max_iterations = count
    config.budget_caps = {"max_iterations": cap}
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "success", result.failure_reason
    assert result.metrics["completed_iterations"] == expected
    assert result.metrics["resolved_run_policy"]["max_iterations"] == expected
    assert result.metrics["stop_reason"] == reason
    assert result.metrics["converged"] is False
    assert result.metrics["final_data_residual"] is None
    assert FakeClient.requests[-1]["schedule"] == [1, 1, 2][:expected]


def test_input_geometry_sign_axes_masks_and_no_gt_leak(wave_case, config, tmp_path):
    first = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert first.status == "success", first.failure_reason
    doc, a = read_artifact(
        Path(first.artifacts["wust_directory"]) / "frequency.json",
        "wust.frequency_input",
    )
    np.testing.assert_array_equal(a["tx_xy_m"], wave_case.geometry.tx_pos_m[:, ::-1])
    np.testing.assert_array_equal(a["pressure"], wave_case.measurement.freq_data)
    assert doc["arrays"]["pressure"]["axes"] == "frequency,tx,rx"
    assert doc["metadata"]["fourier_sign"] == 1
    wave_case.ground_truth.sound_speed_mps[:] = 1450
    second = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert second.status == "success", second.failure_reason
    for name in ("frequency", "initial"):
        _, a = read_artifact(
            Path(first.artifacts["wust_directory"]) / (name + ".json"),
            "wust.frequency_input" if name == "frequency" else "wust.initial_model",
        )
        _, b = read_artifact(
            Path(second.artifacts["wust_directory"]) / (name + ".json"),
            "wust.frequency_input" if name == "frequency" else "wust.initial_model",
        )
        assert {k: array_hash(v) for k, v in a.items()} == {
            k: array_hash(v) for k, v in b.items()
        }
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)


@pytest.mark.parametrize(
    "key,value",
    [("update_rtol", 1e-3), ("max_forward_calls", 1), ("max_adjoint_calls", 0)],
)
def test_unsupported_controls_fail(wave_case, config, key, value):
    setattr(config.run_controls, key, value)
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "failed"
    assert not FakeClient.requests


def test_deadline_is_shared_and_failure_has_no_image(wave_case, config, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class SlowClient(FakeClient):
        def invoke(self, req):
            if req["operation"] == "ingest_frequency":
                clock[0] += 61
            return super().invoke(req)

    monkeypatch.setattr(module, "WUSTClient", SlowClient)
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "failed" and result.sound_speed_mps is None
    assert result.metrics["stop_reason"] == "time_budget"


@pytest.mark.parametrize(
    "key,value",
    [
        ("pipeline_module", "x"),
        ("pipeline_args", []),
        ("warm_start_builder", "bulk_support"),
        ("regularization_lambda", 0),
        ("inner_iterations", 2),
    ],
)
def test_obsolete_parameters_rejected(key, value):
    with pytest.raises(ValueError):
        WUSTParameters.model_validate({key: value})


def test_agent_and_runtime_admission_contract():
    description = SPECS["fwi_wust"].describe()
    assert description["family"] == "full_wave"
    assert len(description["variants"]) == 1
    assert not {
        "runtime_root",
        "backend",
        "matlab_executable",
        "initial_map_mps",
    } & set(description["allowed_parameters"])
    for p in ({"backend": "cpu"}, {"initialization": "map"}, {"runtime_root": "/tmp"}):
        with pytest.raises(ValueError):
            make_agent_config("fwi_wust", p)
    assert (
        make_agent_config(
            "fwi_wust", {"initialization": "scalar", "initial_sound_speed_mps": 1490.0}
        ).parameters["initial_sound_speed_mps"]
        == 1490.0
    )
    with pytest.raises(ValueError):
        module.resolve_schedule([1.0, 2.0], [1.01])


def test_unknown_convention_and_invalid_mask_fail(wave_case, tmp_path):
    del wave_case.metadata["pressure_contract"]
    with pytest.raises(ValueError):
        frequency_input(wave_case, tmp_path / "bad.json")


def test_tiny_fwi_loss_decreases(synthetic_case):
    from usctbench.algorithms.fwi.tiny import TinyFWIAlgorithm

    result = TinyFWIAlgorithm().run(
        synthetic_case, AlgorithmConfig(parameters={"steps": 5, "learning_rate": 1e6})
    )
    assert result.status == "success" and result.metrics["loss_decreased"] is True


@pytest.mark.parametrize(
    "corruption",
    ["axes", "units", "shape", "pin", "backend", "schedule", "convergence"],
)
def test_corrupt_runtime_result_never_becomes_success(
    wave_case, config, monkeypatch, corruption
):
    class CorruptClient(FakeClient):
        def invoke(self, req):
            output = super().invoke(req)
            if req["operation"] == "reconstruct":
                if corruption in {"axes", "units", "shape"}:
                    output["arrays"]["c_mps"][corruption] = {
                        "axes": "x,y",
                        "units": "mm/s",
                        "shape": [27, 23],
                    }[corruption]
                elif corruption == "pin":
                    output["metadata"]["runtime_provenance"] = {"git_sha": "wrong"}
                elif corruption == "backend":
                    output["metadata"]["environment"]["backend"] = "cpu"
                elif corruption == "schedule":
                    output["metadata"]["executed_schedule"] = [1]
                else:
                    output["metadata"]["completion"]["converged"] = True
                Path(req["output_manifest"]).write_text(json.dumps(output))
            return output

    monkeypatch.setattr(module, "WUSTClient", CorruptClient)
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "failed" and result.sound_speed_mps is None


def test_explicit_map_empty_schedule_and_frequency_mask(wave_case, config):
    initial = np.full(wave_case.grid.shape, 1495.0)
    initial[10:12, 14:16] = 1502.0
    config.parameters.update(
        initialization="map", initial_map_mps=initial.tolist(), frequency_schedule_hz=[]
    )
    wave_case.measurement.valid_mask = np.broadcast_to(
        wave_case.measurement.valid_mask, (2, 2, 4)
    ).copy()
    wave_case.measurement.valid_mask[1, 0, 2] = False
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "success", result.failure_reason
    np.testing.assert_array_equal(result.sound_speed_mps, initial)
    assert result.metrics["completed_iterations"] == 0
    assert result.metrics["resolved_run_policy"]["max_iterations"] == 0


@pytest.mark.parametrize(
    "mutation",
    ["missing_reference", "wrong_map_shape", "empty_mask", "unknown_schedule"],
)
def test_invalid_scientific_input_fails(wave_case, config, mutation):
    if mutation == "missing_reference":
        del wave_case.metadata["reference_sound_speed_mps"]
    elif mutation == "wrong_map_shape":
        config.parameters.update(initialization="map", initial_map_mps=[[1500.0]])
    elif mutation == "empty_mask":
        config.parameters["update_mask"] = np.zeros(
            wave_case.grid.shape, dtype=bool
        ).tolist()
    else:
        config.parameters["frequency_schedule_hz"] = [100001.0]
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "failed" and result.sound_speed_mps is None


def test_no_gt_case_is_valid(wave_case, config):
    wave_case.ground_truth.sound_speed_mps = None
    result = module.WUSTFWIAlgorithm().run(wave_case, config)
    assert result.status == "success", result.failure_reason
    assert result.metrics["final_data_residual"] is None
