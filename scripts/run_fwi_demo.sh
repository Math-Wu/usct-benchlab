#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

: "${USCT_WUST_ROOT:?Set USCT_WUST_ROOT to the approved WUST checkout.}"

cd "$REPO_DIR"
FWI_CASE_GLOB="${USCT_WUST_CASE_GLOB:-data/fwi_wust_demo/cases/*.h5}"
if ! compgen -G "$FWI_CASE_GLOB" >/dev/null; then
  echo "No FWI cases matched $FWI_CASE_GLOB" >&2
  echo "Set USCT_WUST_CASE_GLOB to explicit frequency-pressure USCTCase files." >&2
  exit 1
fi

PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m usctbench.cli bench \
  --suite configs/benchmarks/fwi_wust_demo.yaml
