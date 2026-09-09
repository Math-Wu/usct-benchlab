import hashlib
import json
from pathlib import Path
import runpy

import h5py
import numpy as np
from scipy.io import savemat


def test_calibration_selects_without_consulting_validation_speeds(tmp_path, capsys):
    script = Path(__file__).resolve().parents[2] / "scripts/calibrate_arrival.py"
    evaluate = runpy.run_path(str(script))["evaluate"]
    source = tmp_path / "acquisition"
    source.mkdir()
    out = tmp_path / "calibration"
    out.mkdir()
    positions = np.array([[-0.03, 0], [0, -0.03], [0.03, 0], [0, 0.03]])
    distance = np.linalg.norm(positions[:2, None] - positions[None], axis=-1)
    time = np.arange(3000) * 0.05e-6
    speeds = np.array([1500, 1460, 1540, 1480, 1520])
    pressure = np.empty((len(time), 2, 4, 5))
    for k, speed in enumerate(speeds):
        phase_time = time[:, None, None] - distance[None] / speed - 7.5e-6
        pressure[..., k] = np.sin(2 * np.pi * 2e5 * phase_time) * np.exp(
            -((phase_time / 2e-6) ** 2)
        )
    savemat(
        out / "input.mat", {"positions_yx": positions, "tx_indices": np.array([0, 1])}
    )
    manifest = {
        "acquisition": str(source),
        "input_sha256": hashlib.sha256((out / "input.mat").read_bytes()).hexdigest(),
        "tx_parent_indices": [0, 1],
        "rx_parent_indices": [0, 1, 2, 3],
        "pulse_duration_s": 15e-6,
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    with h5py.File(source / "pressure.mat", "w") as f:
        f["time"] = time
        f["water_dataset"] = pressure[..., 0].transpose(1, 2, 0)

    def save():
        with h5py.File(out / "pressure.mat", "w") as f:
            f["time"] = time
            f["speeds"] = speeds
            f["pressure"] = pressure.transpose(3, 2, 1, 0)

    save()
    evaluate(out)
    first = json.loads((out / "summary.json").read_text())
    assert first["selected_on_training_only"] == "xcorr"
    assert first["water_replay_relative_residual"] == 0
    pressure[..., 3:] = np.roll(pressure[..., 3:], 8, axis=0)
    save()
    evaluate(out)
    second = json.loads((out / "summary.json").read_text())
    assert first["training_scores_ns"] == second["training_scores_ns"]
    assert second["selected_on_training_only"] == first["selected_on_training_only"]
    assert (
        second["results"]["xcorr"]["1480"]["rms_ns"]
        > first["results"]["xcorr"]["1480"]["rms_ns"]
    )
