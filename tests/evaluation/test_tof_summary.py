"""An incomplete experiment must not be reported as completed evidence."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_script():
    # tests/evaluation lives two directories below the repository root.
    path = Path(__file__).resolve().parents[2] / "scripts/summarize_tof_attribution.py"
    spec = importlib.util.spec_from_file_location("tof_summary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_flag_cannot_fabricate_completed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["summary", "--out", str(tmp_path), "--allow-partial"]
    )
    load_script().main()
    report = json.loads((tmp_path / "report/summary.json").read_text())
    assert not report["complete"]
    assert report["completed_pipeline_runs"] == 0
    assert len(report["missing"]) == 59


def test_default_report_fails_on_missing_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["summary", "--out", str(tmp_path)])
    with pytest.raises(RuntimeError, match="incomplete campaign"):
        load_script().main()
