#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
SMOKE_ROOT="${USCT_SMOKE_ROOT:-/tmp/usctbench_synthetic_demo}"
RUN_ROOT="${USCT_RUN_ROOT:-/tmp/usctbench_runs}"

cd "$REPO_DIR"

echo "Running unit tests"
"$PYTHON_BIN" -m pytest -q

echo "Creating synthetic demo cases in $SMOKE_ROOT"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli data make-synthetic-smoke \
  --out "$SMOKE_ROOT" \
  --shape "${USCT_SMOKE_SHAPE:-12}" \
  --n-transducers "${USCT_SMOKE_TRANSDUCERS:-8}"

echo "Running synthetic demo benchmark"
USCT_SYNTHETIC_CASE_GLOB="$SMOKE_ROOT/cases/*.h5" \
USCT_RUN_ROOT="$RUN_ROOT" \
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/synthetic_demo.yaml

echo "Running release audit"
PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" scripts/audit_release.py
