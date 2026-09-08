# Usage Guide

This guide shows the normal workflow for installation, dataset
preparation, single-algorithm runs, benchmark runs, and output inspection.

## Install

Conda:

```bash
conda create -n usctbench python=3.10 -y
conda activate usctbench
pip install -e ".[dev,viz]"
```

Pip:

```bash
pip install -r requirements.txt
pip install -e .
```

Verify:

```bash
usct --help
usct list-algorithms
pytest -q
```

## Workspace Setup

Use a workspace outside Git for data and generated runs:

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
```

Prepare directories:

```bash
bash scripts/setup_workspace.sh
```

The setup script creates workspace directories and lightweight repo-local
symlinks such as `data/raw/openbreastus` and `runs/current`. It does not copy
datasets into Git and it does not make generated runs tracked files.

## Synthetic Demo

Create deterministic local cases:

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

Set the case glob:

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
```

## OpenBreastUS Demo

Inspect the dataset tree:

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"
```

Create one case per density class when available:

```bash
usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

Set the case glob:

```bash
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
```

## NBPslice2D Demo

Inspect the ZIP:

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"
```

Create one case per A/B/C/D class when available:

```bash
usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

Set the case glob:

```bash
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
```

## Run CGLS

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out runs/single_cgls
```

## Run SIRT

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out runs/single_sirt
```

## Run SART

```bash
usct run straight_sart \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sart.yaml \
  --out runs/single_sart
```

## Run Bent-Ray

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out runs/single_bent_ray
```

`bent_ray_gn` uses a nonlinear fast-marching Eikonal model and its discrete adjoint.

## Run rWave Adapter

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/physics/example/pressure_case.h5" \
  --config configs/algorithms/rwave.yaml \
  --out runs/single_rwave
```

`rwave_adapter` requires complex pressure plus source calibration or independent
water pressure. See [physics validation](physics_validation.md) to generate and
import a pressure pair. ToF-only cases are rejected explicitly.

## Run FWI Adapter

Set an external artifact path:

```bash
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
```

The result MAT file must contain `VEL_ESTIM`. Optional fields such as
`C_INTERP`, `VEL_ESTIM_ITER`, and `LOSS_ITER` enable ground-truth metrics and
iteration selection.

Run:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

The adapter can also construct and call an external command when configured to
do so, but release tests focus on artifact ingestion and reporting.

## Run Diffusion + FWI Adapter

Set external DPS artifact paths:

```bash
export USCT_DPS_FWI_RESULT_PATH=/path/to/dps_result.mat
export USCT_DPS_FWI_SUMMARY_PATH=/path/to/dps_result.json
export USCT_DPS_DATASET_PATH=/path/to/kwave_dataset.mat
export USCT_DPS_CHECKPOINT=/path/to/diffusion_checkpoint.pth
```

Run:

```bash
usct run diffusion_fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/diffusion_fwi_kwave.yaml \
  --out runs/single_diffusion_fwi
```

The default config is loader-only. Set `run_external: true` only in an
environment where the external USCT-kwave package, checkpoint, CUDA device,
MATLAB/k-Wave runtime, and dataset path are deliberately available.

## Run Benchmark

Synthetic:

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
```

NBPslice2D:

```bash
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
```

OpenBreastUS:

```bash
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
```

FWI:

```bash
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

Diffusion + FWI:

```bash
usct bench --suite configs/benchmarks/diffusion_fwi_kwave_demo.yaml
```

Run all available demos:

```bash
bash scripts/run_demo_benchmarks.sh
```

## Inspect Outputs

Single runs write:

```text
runs/single_cgls/synthetic_circular_sos/result.h5
runs/single_cgls/synthetic_circular_sos/metrics.json
runs/single_cgls/synthetic_circular_sos/metadata.yaml
runs/single_cgls/synthetic_circular_sos/preview.png
```

Benchmark runs write:

```text
runs/usctbench_runs/<run_id>/<algorithm>/<case_id>/result.h5
runs/usctbench_runs/<run_id>/<algorithm>/<case_id>/metrics.json
runs/usctbench_runs/<run_id>/<algorithm>/<case_id>/metadata.yaml
runs/usctbench_runs/<run_id>/<algorithm>/<case_id>/preview.png
runs/usctbench_runs/<run_id>/benchmark_summary.csv
runs/usctbench_runs/<run_id>/benchmark_report.md
```

Start with `benchmark_report.md` for a readable summary, then inspect
`metrics.json` and `preview.png` for individual algorithm/case behavior.
