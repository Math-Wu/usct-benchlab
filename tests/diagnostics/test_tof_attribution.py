"""Conservation gates for the executable oracle control experiment."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from usctbench.evaluation.tof_diagnostics import shuffled_pair_error

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "attribution_cli", SCRIPTS / "audit_tof_attribution_all_groups.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path.pop(0)


def test_control_gate_checks_each_partition():
    rng = np.random.default_rng(15)
    error = rng.normal(size=(8, 8)) * 1e-7
    mask = ~np.eye(8, dtype=bool)
    weights = rng.uniform(0.1, 1, (8, 8))
    groups = np.zeros((8, 8), dtype=int)
    groups[4:, 4:] = 1
    actual, _ = shuffled_pair_error(
        error, mask=mask, weights=weights, groups=groups, seed=1
    )
    report = module.conservation_check(error, actual, mask, weights, groups)
    assert set(report) == {"0", "1"}
    assert max(r["relative_energy_error"] for r in report.values()) < 1e-12


@pytest.mark.parametrize("change", ["scale", "mean", "antisymmetry"])
def test_control_gate_rejects_corruption(change):
    error = np.arange(16, dtype=float).reshape(4, 4) * 1e-7
    mask = ~np.eye(4, dtype=bool)
    actual = error.copy()
    if change == "scale":
        actual *= 2
    elif change == "mean":
        actual += 1e-7
    else:
        actual = actual.T.copy()
    with pytest.raises(ValueError):
        module.conservation_check(
            error, actual, mask, np.ones((4, 4)), np.zeros((4, 4), dtype=int)
        )


def test_atomic_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "report.json"
    module.atomic_json(path, {"ok": True})
    with pytest.raises(ValueError):
        module.atomic_json(path, {"bad": float("nan")})
    assert path.read_text().strip() == '{\n  "ok": true\n}'
