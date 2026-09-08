import importlib.util
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import pytest

from usctbench.core.io import write_case_hdf5
from usctbench.data.synthetic import make_sound_speed_case


def pipeline():
    path = Path(__file__).resolve().parents[2] / "scripts/validate_physics.py"
    spec = importlib.util.spec_from_file_location("validation_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared(tmp_path):
    module = pipeline()
    case_path = tmp_path / "case.h5"
    write_case_hdf5(make_sound_speed_case(shape=(8, 8), n_transducers=8), case_path)
    options = SimpleNamespace(
        case=str(case_path),
        out=str(tmp_path / "work" / "case"),
        transducers=8,
        source_frequency=200e3,
        max_frequency=250e3,
    )
    module.prepare(options)
    return module, options


def test_simulation_reuse_requires_matching_settings(tmp_path):
    module, options = prepared(tmp_path)
    with pytest.raises(FileExistsError):
        module.prepare(options)
    options.out = str(tmp_path / "work")
    options.cases = [options.case]
    options.workers, options.device, options.transducers = 1, 0, 4
    with pytest.raises(ValueError, match="settings/property checksum"):
        module.batch(options)


def test_incomplete_matlab_output_is_not_published(tmp_path, monkeypatch):
    module, options = prepared(tmp_path)
    options.matlab, options.kwave_path, options.device = "matlab", "external/kwave", 0
    pending = Path(options.out) / "pressure.partial.mat"

    def fail(*args, **kwargs):
        with h5py.File(pending, "w") as f:
            f["full_dataset"] = [1]
        raise subprocess.CalledProcessError(1, "matlab")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.simulate(options)
    assert pending.exists()
    assert not (pending.parent / "pressure.mat").exists()


def test_run_snapshot_is_immutable_and_existing_output_is_not_overwritten(tmp_path):
    module = pipeline()
    repo = Path(__file__).resolve().parents[2]
    run = tmp_path / "run"
    snapshot = module.freeze_run_source(repo, run)
    source = repo / "src/usctbench/algorithms/rwave.py"
    copied = snapshot / "usctbench/algorithms/rwave.py"
    assert copied.read_bytes() == source.read_bytes()
    assert not list(snapshot.rglob("__pycache__"))
    with pytest.raises(FileExistsError):
        module.freeze_run_source(repo, run)
    assert copied.read_bytes() == source.read_bytes()
