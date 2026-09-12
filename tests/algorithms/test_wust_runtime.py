"""Fail-closed subprocess/identity tests without MATLAB or CUDA."""

import os

import pytest

from usctbench.algorithms.fwi.parameters import WUSTParameters
from usctbench.algorithms.fwi.runtime import (
    Deadline,
    RuntimeFailure,
    WUSTClient,
    WUST_MANIFEST,
    WUST_SHA,
    WUST_VERSION,
)


def capabilities():
    return {
        "schema": "wust.capabilities",
        "schema_version": 1,
        "runtime_name": "WUST",
        "runtime_version": WUST_VERSION,
        "schemas": {
            key: 1
            for key in (
                "wust.request",
                "wust.frequency_input",
                "wust.measurements",
                "wust.reconstruction",
            )
        },
        "provenance": {
            "git_sha": WUST_SHA,
            "runtime_version": WUST_VERSION,
            "dirty": False,
            "source_manifest_verified": True,
            "source_manifest_sha256": WUST_MANIFEST,
        },
        "sound_speed_reconstruction": {"supported": True},
        "attenuation_reconstruction": {"supported": False},
        "operations": ["ingest_frequency", "reconstruct"],
        "profiles": {"gpu": "gpu_complex64"},
        "iteration_unit": "frequency_schedule_update",
        "fourier_sign": -1,
        "source_scale": {
            "method": "per_tx_per_frequency_complex_least_squares",
            "minimum_fitting_receivers": 2,
            "requires_water_reference": False,
            "requires_source_spectrum": False,
        },
        "budgets": {
            "resolved_schedule": True,
            "hard_timeout_s": True,
            "forward_call_cap": False,
            "adjoint_call_cap": False,
            "update_rtol": False,
            "online_convergence": False,
        },
    }


@pytest.fixture
def client(tmp_path):
    (tmp_path / "Runtime/python").mkdir(parents=True)
    return WUSTClient(
        WUSTParameters(runtime_root=str(tmp_path)), Deadline(10), tmp_path
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_version", "9.0.0"),
        ("fourier_sign", 1),
        ("iteration_unit", "inner_step"),
        ("source_scale", {}),
        ("budgets", {}),
    ],
)
def test_identity_and_capabilities_fail_closed(client, monkeypatch, field, value):
    info = capabilities()
    info[field] = value
    monkeypatch.setattr(client, "command", lambda args: info)
    with pytest.raises(RuntimeFailure, match="clean approved SHA"):
        client.admit()


@pytest.mark.parametrize(
    "field,value",
    [
        ("git_sha", "0" * 40),
        ("dirty", True),
        ("source_manifest_verified", False),
        ("source_manifest_sha256", "wrong"),
    ],
)
def test_provenance_rejection(client, monkeypatch, field, value):
    info = capabilities()
    info["provenance"][field] = value
    monkeypatch.setattr(client, "command", lambda args: info)
    with pytest.raises(RuntimeFailure):
        client.admit()


def test_cpu_cannot_satisfy_production_and_missing_mex_fails(client, monkeypatch):
    environment = {"cpu_available": True, "gpu_available": False}
    monkeypatch.setattr(
        client,
        "command",
        lambda args: (
            {**capabilities(), "environment": environment}
            if "--probe" in args
            else capabilities()
        ),
    )
    with pytest.raises(RuntimeFailure, match="gpu unavailable"):
        client.admit()
    environment.update(gpu_available=True, mex=[])
    with pytest.raises(RuntimeFailure, match="CUDA MEX"):
        client.admit()
    client.parameters.backend = "cpu"
    admission = client.admit()
    assert admission["role"] == "reference"
    assert admission["approved_for_production"] is False


def test_runtime_failure_and_unclean_stdout(client):
    client.launcher.write_text(
        'import sys\nsys.stderr.write(\'{"schema":"wust.execution_failure","reason":"numerical_failure","message":"bad factor"}\\n\')\nsys.exit(3)\n'
    )
    with pytest.raises(RuntimeFailure) as error:
        client.command([])
    assert error.value.reason == "numerical_failure"
    client.launcher.write_text('print("startup banner")\nprint("{}")\n')
    with pytest.raises(RuntimeFailure, match="clean JSON"):
        client.command([])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group contract")
def test_deadline_reaps_launcher(client):
    pidfile = client.folder / "child.pid"
    client.launcher.write_text(
        f"import os,time\nfrom pathlib import Path\nPath({str(pidfile)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    client.deadline = Deadline(0.5)
    with pytest.raises(RuntimeFailure) as error:
        client.command([])
    assert error.value.reason == "time_budget"
    assert pidfile.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
