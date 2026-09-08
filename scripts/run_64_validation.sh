#!/usr/bin/env bash
# Reproduce the fixed 64-channel campaign. Runs on CPU even on an A100 host.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_64_validation.sh HANDOFF_DIR FRESH_OUTPUT_DIR" >&2
  exit 2
fi
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF="$(cd "$1" && pwd)"
OUT="$(realpath -m "$2")"
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
case "$OUT/" in "$REPO/"*) echo "Output must be outside the repository" >&2; exit 2;; esac
mkdir -p "$OUT"
ln -s "$HANDOFF/cases" "$OUT/cases"
ln -s "$HANDOFF/configs" "$OUT/configs"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
python "$HANDOFF/tools/verify.py" --repo "$REPO" 2>&1 | tee "$OUT/verify.log"
python -m pytest -q 2>&1 | tee "$OUT/tests.log"
HIGH="cases/high_band/D510022534/envelope_case.h5"
LOW_D="cases/low_band/D510022534/pressure_case.h5"
LOW_OB="cases/low_band/breast_train_speed_class_1_000000/pressure_case.h5"
run_case() {
  local case_path="$1" prefix="$2"
  python scripts/validate_ray_features.py --handoff "$OUT" --case "$case_path" \
    --features existing --out "$OUT/${prefix}_pixels" 2>&1 | tee "$OUT/${prefix}_pixels.log"
  if [[ "$prefix" == "final_high" ]]; then
    python scripts/check_64_baselines.py --reference "$HANDOFF/baselines_64" \
      --actual "$OUT/${prefix}_pixels/existing" --out "$OUT/baseline_acceptance.json" \
      2>&1 | tee "$OUT/baseline_acceptance.log"
  fi
  python scripts/validate_ray_features.py --handoff "$OUT" --case "$case_path" \
    --features existing --model-grid-shape 32 32 --out "$OUT/${prefix}_basis32" \
    2>&1 | tee "$OUT/${prefix}_basis32.log"
}
run_case "$HIGH" final_high
# Preserve the negative control; this is not a recommended feature replacement.
python scripts/validate_ray_features.py --handoff "$OUT" --case "$HIGH" \
  --features source_phase --out "$OUT/phase_high" 2>&1 | tee "$OUT/phase_high.log"
run_case "$LOW_D" low_d
run_case "$LOW_OB" low_ob
# GT is used here only for a post-reconstruction diagnostic, never to change data.
python scripts/audit_ray_numerics.py --case "$HANDOFF/$HIGH" \
  --out "$OUT/numerical_audit.json" 2>&1 | tee "$OUT/numerical_audit.log"
python scripts/render_ray_comparison.py --handoff "$OUT" --out "$OUT/report" \
  2>&1 | tee "$OUT/render.log"
printf 'Completed 28 frozen reconstructions; report: %s/report\n' "$OUT"
