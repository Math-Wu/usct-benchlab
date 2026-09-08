# usct-benchlab

[中文说明](README.zh-CN.md)

`usct-benchlab` is a lightweight Python benchmark package for ultrasound
computed tomography (USCT) reconstruction algorithms with unified input/output,
classical baselines, and FWI adapter support. It provides dataset conversion
helpers, runnable algorithm configs, common metrics, preview figures, and
benchmark summaries for reproducible comparisons.

## What is USCT?

**Ultrasound computed tomography is a PDE-constrained inverse problem.** A
source transducer emits an acoustic pulse, the pressure field propagates through
the object according to an acoustic wave equation, and receiver transducers
measure the resulting pressure traces. The inverse problem is to recover
spatial acoustic properties from those measurements.

The main reconstruction target is the sound-speed map $c(x)$. Related physical
properties include density $\rho(x)$ and attenuation $\alpha(x)$. This package
converts datasets to a common `USCTCase` schema and returns every algorithm
output as a `ReconstructionResult`.

## Mathematical Formulation

USCT should be read as a PDE-driven inverse problem, not as a generic image
reconstruction task. A source transducer excites an acoustic pressure field,
the field propagates through the unknown medium, and receiver measurements are
used to infer the medium parameters.

$$
\frac{1}{c(x)^2}\partial_{tt}p_s(t,x)-\Delta p_s(t,x)=q_s(t,x).
$$

In frequency-domain notation, the corresponding Helmholtz form is commonly
written as

$$
\left(\Delta+\omega^2m(x)\right)\hat p_s(\omega,x)=-\hat q_s(\omega,x).
$$

Here $p_s$ is pressure for source $s$, $q_s$ is the emitted source, $c(x)$ is
sound speed, and $m(x)$ is squared slowness:

$$
m(x)=\frac{1}{c(x)^2}.
$$

Most sound-speed methods in this repository estimate either the sound-speed map
$c(x)$ or a slowness map

$$
u(x)=\frac{1}{c(x)}.
$$

Receiver $r$ observes the propagated field through a measurement operator:

$$
d_{sr}(t)=\mathcal M_r p_s(t,\cdot)+\eta_{sr}(t).
$$

The key distinction between algorithms is how much of this wave physics they
keep. Waveform-based FWI keeps the acoustic PDE or Helmholtz solve inside the
optimization and matches measured pressure traces or complex frequency-domain
pressure. Travel-time surrogate methods first reduce the data to arrival-time
features, then invert a ray or eikonal approximation. They are faster and more
stable as baselines, but they discard waveform phase, amplitude, diffraction,
and much of the finite-frequency physics.

The straight-ray travel-time approximation uses a reference speed $c_0$ and
fixed path $\gamma_{sr}$:

$$
\Delta t_{sr}\approx\int_{\gamma_{sr}}\delta u(x)d\ell.
$$

The slowness perturbation is

$$
\delta u(x)=\frac{1}{c(x)}-\frac{1}{c_0}.
$$

After pixel discretization, the straight-ray model becomes

$$
A\delta u\approx b.
$$

CGLS, SIRT, and SART are different solvers or update rules for this algebraic
travel-time system. A representative regularized objective is

$$
\min_{\delta u}\|W(A\delta u-b)\|_2^2+\lambda^2\|L\delta u\|_2^2.
$$

Bent-ray methods keep a high-frequency travel-time model in which paths depend
on the current medium:

$$
|\nabla T_s(x)|=u(x).
$$

The receiver travel time is approximated by

$$
t_{sr}\approx T_s(r).
$$

The idealized nonlinear travel-time objective is

$$
\min_c\sum_{s,r}\left|t_{sr}^{\mathrm{obs}}-T_s(r;c)\right|^2+\lambda R(c).
$$

FWI uses the pressure data directly. In frequency-domain form, a common
PDE-constrained objective is

$$
\min_c\frac{1}{2}\sum_{\omega,s,r}\left|\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)\right|^2+\lambda R(c).
$$

In this expression, $\hat p_s(\omega,r;c)$ is not an arbitrary image operator;
it is the pressure predicted by an acoustic PDE or Helmholtz solver for the
candidate sound speed.

| Method | Modeling assumption | Optimization target | Appropriate use |
| --- | --- | --- | --- |
| CGLS | Fixed straight rays through a reference medium; travel-time delays are linearized in slowness perturbation. | Krylov solve of the weighted regularized least-squares system for $A\delta u\approx b$. | Fast, reproducible sound-speed baseline and regression test for ring-geometry cases. |
| SIRT | Same straight-ray algebraic model as CGLS, but with simultaneous normalized residual backprojection updates. | Iteratively reduce the weighted residual of $A\delta u\approx b$ with relaxation and smoothing. | Robust baseline when stability matters more than sharp convergence. |
| SART | Same straight-ray model, updated by ordered transmitter or ray subsets. | Ordered row-action updates that reduce the algebraic travel-time residual subset by subset. | Faster early iterations and sharper straight-ray baselines, with more sensitivity to ordering and relaxation. |
| Bent-ray | High-frequency travel time follows an eikonal model; rays bend according to the current sound speed or slowness. | Regularized nonlinear travel-time mismatch based on $T_s(r;c)$. | Refraction-aware surrogate comparison when full waveform inversion is too expensive or unavailable. |
| FWI | Full acoustic wave or Helmholtz propagation; measured data are pressure waveforms or complex pressure samples. | PDE-constrained waveform mismatch over sources, receivers, and frequencies. | High-fidelity reporting when external k-Wave/FWI artifacts or an external FWI command are available. |

`bent_ray_gn` solves a nonlinear Eikonal equation with fast marching and an
exact discrete Jacobian/adjoint. `rwave_adapter` requires calibrated complex
pressure and relinearizes finite-frequency Born scattering. Its supplied config
uses full Green backgrounds from a free-space volume-integral solve; the
Eikonal/WKB Green approximation remains an explicit option. Neither uses the
straight-ray projector. These discretizations
do not guarantee a monotone ranking of image quality or reproduce every option
of upstream r-Wave. The production FWI path remains the external MATLAB driver.
For more detail, see [docs/math_formulation.md](docs/math_formulation.md).

## Supported Algorithms

| Algorithm | Command name | Mathematical model | Input requirement | Typical use | Config |
| --- | --- | --- | --- | --- | --- |
| CGLS | `straight_cgls` | Straight-ray weighted least squares | `USCTCase` with ring geometry and travel-time measurements | Fast sound-speed baseline | `configs/algorithms/cgls.yaml` |
| SIRT | `straight_sirt` | Simultaneous iterative ray tomography | `USCTCase` with ring geometry and travel-time measurements | Robust iterative sound-speed baseline | `configs/algorithms/sirt.yaml` |
| SART | `straight_sart` | Ordered/subset algebraic ray update | `USCTCase` with ring geometry and travel-time measurements | Ordered-update straight-ray baseline | `configs/algorithms/sart.yaml` |
| Attenuation SIRT | `attenuation_sirt` | Straight-ray log-amplitude tomography | `USCTCase` with log-amplitude measurements | Attenuation baseline | `configs/algorithms/attenuation.yaml` |
| Bent-ray | `bent_ray_gn` | Nonlinear Eikonal / fast marching | First-arrival times or calibrated delays | Refraction-corrected tomography | `configs/algorithms/bent_ray.yaml` |
| rWave adapter | `rwave_adapter` | Relinearized finite-frequency Ray-Born | Complex `(frequency,tx,rx)` pressure and calibrated source or independent water reference | Scattering-sensitive pressure inversion | `configs/algorithms/rwave.yaml` |
| FWI adapter | `fwi_kwave_adapter` | PDE-level full-wave inversion adapter | `USCTCase` plus external k-Wave/FWI artifact or command path | High-fidelity FWI reporting | `configs/algorithms/fwi_kwave.yaml` |
| Diffusion FWI adapter | `diffusion_fwi_kwave_adapter` | External diffusion-prior k-Wave/FWI DPS adapter | `USCTCase` plus external DPS `.mat`/`.json` artifact or command path | Report existing diffusion + FWI outputs in the same benchmark format | `configs/algorithms/diffusion_fwi_kwave.yaml` |
| Tiny FWI sanity | `fwi_tiny` | Small waveform-inversion sanity model | Small synthetic sound-speed case | Local waveform-inversion plumbing test | `configs/algorithms/fwi_tiny.yaml` |

More details are in [docs/algorithms.md](docs/algorithms.md).

### Physics and Agent Validation

Operators are separated into `operators/forward/` and `operators/adjoint/`, each
containing straight-ray, Eikonal, Ray-Born and external full-wave implementations.
See [operator contracts](docs/operator_contract.md) for units, exact versus
approximate derivatives, and the optional live MATLAB Helmholtz bridge.

Native solvers support grouped receiver/frequency validation and OR stopping:
residual/noise targets, update/objective/validation stagnation, time/call budgets
and iteration caps. Reports retain the actual stop reason, selected checkpoint
and work counts. GT metrics are optional and never select iterates by default.
See [evaluation and stopping](docs/agent_evaluation.md) and the
[reproducible validation workflow](docs/physics_validation.md).

### Diffusion + FWI Adapter

`diffusion_fwi_kwave_adapter` reports reconstructions produced by an external
diffusion-prior k-Wave/FWI pipeline. The important point is the sampling loop:
the current sound-speed image is repeatedly corrected by a waveform physics
step and then nudged by a learned diffusion prior. The default path starts from
a `bulk_support` warm start, uses sparse64 observations, and applies the FWI
physics update before the prior update (`physics_position=pre`).

```math
g_k =
\nabla_c
\frac{1}{2}
\left\|
\hat p(c_k)-\hat p^{\mathrm{obs}}
\right\|_2^2,
\qquad
c_{k+\frac{1}{2}} =
\mathrm{LineSearch}\left(c_k - \eta_k M g_k\right).
```

Here $g_k$ is the FWI/Helmholtz gradient computed from the current waveform
residual, and $M$ denotes the configured preconditioner, such as
`slowness_precond`. After this data-consistency step, the diffusion model is
queried at a low noise level and used as a score/prior correction:

```math
s_k = s_{\theta}(c_{k+\frac{1}{2}}, t),
\qquad
c_{k+1} =
\Pi_{[c_{\min},c_{\max}]}
\left(c_{k+\frac{1}{2}} + \lambda_k s_k\right).
```

In the default smoke configuration, `score_reg_t=0.10` and
`score_reg_lambda=0.1`. The FWI gradient keeps the sample consistent with the
measured pressure data, while the diffusion prior suppresses implausible
textures and biases the iterate toward the distribution learned from
OpenBreastUS-like sound-speed maps. The adapter itself does not train the
diffusion model and does not vendor PyTorch, MATLAB, or k-Wave into
`usct-benchlab`; it only loads an external DPS `.mat`/`.json` result or, when
explicitly configured, launches the external pipeline.

The expected DPS result fields are `VEL_DPS_PHYS`, `VEL_DPS_VIEW`,
`VEL_FINAL_PHYS`, `VEL_FINAL_VIEW`, `VEL_INIT_VIEW`, and `GT_VIEW`. The JSON
summary is used to record checkpoint, dataset, frequency schedule, selected
step, and prior settings.

## Installation

Conda workflow:

```bash
conda create -n usctbench python=3.10 -y
conda activate usctbench
pip install -e ".[dev,viz]"
```

Pip workflow:

```bash
pip install -r requirements.txt
pip install -e .
```

Check the installation:

```bash
usct --help
usct list-algorithms
pytest -q
```

For a minimal end-to-end run that writes only to `/tmp`, use:

```bash
bash examples/synthetic_quickstart.sh
```

## Environment and Workspace Layout

Use environment variables so local data and generated runs stay outside Git:

```bash
export USCT_WORKSPACE=/path/to/usct-benchlab
export USCT_DATA_ROOT=$USCT_WORKSPACE/data/openbreastus
export USCT_RUN_ROOT=$USCT_WORKSPACE/runs/usctbench_runs
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
export USCT_DPS_FWI_RESULT_PATH=/path/to/dps_result.mat
export USCT_DPS_FWI_SUMMARY_PATH=/path/to/dps_result.json
export USCT_DPS_DATASET_PATH=/path/to/kwave_dataset.mat
export USCT_DPS_CHECKPOINT=/path/to/diffusion_checkpoint.pth
```

Recommended workspace layout:

```text
<workspace>/
  code/          # this repository
  data/          # local datasets and converted cases
  runs/          # benchmark outputs
  external/      # optional external projects
  checkpoints/   # local weights or checkpoints
```

`scripts/setup_workspace.sh` can create this layout and repo-local symlinks; it
does not copy datasets into Git.

## Prepare Datasets

Synthetic demo:

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

OpenBreastUS:

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"

usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

NBPslice2D:

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"

usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

See [docs/usage.md](docs/usage.md) and [docs/datasets.md](docs/datasets.md)
for more complete workflows.

## Run One Algorithm

CGLS:

```bash
usct run straight_cgls \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/cgls.yaml \
  --out runs/single_cgls
```

SIRT:

```bash
usct run straight_sirt \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sirt.yaml \
  --out runs/single_sirt
```

SART:

```bash
usct run straight_sart \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/sart.yaml \
  --out runs/single_sart
```

Bent-ray:

```bash
usct run bent_ray_gn \
  --case "$USCT_WORKSPACE/data/synthetic_demo/cases/synthetic_circular_sos.h5" \
  --config configs/algorithms/bent_ray.yaml \
  --out runs/single_bent_ray
```

rWave adapter:

```bash
usct run rwave_adapter \
  --case "$USCT_WORKSPACE/data/physics/example/pressure_case.h5" \
  --config configs/algorithms/rwave.yaml \
  --out runs/single_rwave
```

This case must contain actual complex pressure and source calibration, not a
speed-map-derived ToF case. Create a pressure pair using the validation workflow;
`mode: fixed_background` explicitly selects the linear Born reference instead
of the default nonlinear background updates.

FWI adapter:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

For the FWI adapter, set `USCT_KWAVE_FWI_RESULT_PATH` when the config should
ingest an existing reconstruction artifact. A readable artifact must include
`VEL_ESTIM`; optional fields such as `C_INTERP`, `VEL_ESTIM_ITER`, and
`LOSS_ITER` enable ground-truth metrics and iteration selection.

Diffusion + FWI adapter:

```bash
export USCT_DPS_FWI_RESULT_PATH=/path/to/dps_result.mat
export USCT_DPS_FWI_SUMMARY_PATH=/path/to/dps_result.json
usct run diffusion_fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/diffusion_fwi_kwave.yaml \
  --out runs/single_diffusion_fwi
```

The DPS artifact can contain `VEL_DPS_PHYS`, `VEL_DPS_VIEW`,
`VEL_FINAL_PHYS`, or `VEL_FINAL_VIEW`. The optional JSON summary is used to
record checkpoint, dataset, frequency schedule, and diffusion-prior settings.

To launch the external diffusion + FWI sampler from `usct-benchlab`, set
`run_external: true` in `configs/algorithms/diffusion_fwi_kwave.yaml` and
provide an existing k-Wave dataset plus diffusion checkpoint:

```bash
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/usct-kwave/python
export USCT_DPS_DATASET_PATH=/path/to/kwave_dataset.mat
export USCT_DPS_CHECKPOINT=/path/to/diffusion_checkpoint.pth
export USCT_DPS_FWI_RESULT_PATH="$USCT_RUN_ROOT/dps_results/case001_dps.mat"
export USCT_DPS_FWI_SUMMARY_PATH="$USCT_RUN_ROOT/dps_results/case001_dps.json"

usct run diffusion_fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/diffusion_fwi_kwave.yaml \
  --out runs/single_diffusion_fwi_external
```

The default sampling settings in `configs/algorithms/diffusion_fwi_kwave.yaml`
match the current smoke path: `array_mode=sparse64`, `warm_start_builder` set
to `bulk_support`, `steps=12`, frequency schedule
`0.3 0.3 0.3 0.35 0.35 0.35 0.4 0.4 0.4 0.45 0.45 0.45 MHz`,
`prior_mode=score_reg`, `score_reg_t=0.10`, `score_reg_lambda=0.1`,
`physics_position=pre`, `physics_inner_steps=1`, `eta=0.1`,
`guidance_gain=1.15`, `gradient_mode=slowness_precond`,
`step_strategy=line_search`, `mask_mode=support_alpha`, and
`final_prior_update=false`.

Example external sampling command, useful when running directly inside the
external USCT-kwave checkout:

```bash
cd "$USCT_KWAVE_ROOT"
PYTHONPATH="$USCT_KWAVE_ROOT" "$USCT_KWAVE_PYTHON_BIN" \
  -m openbreastus_diffusion.kwave_dps.run_dps_kwave \
  --dataset-path "$USCT_DPS_DATASET_PATH" \
  --checkpoint "$USCT_DPS_CHECKPOINT" \
  --init-mat /path/to/bulk_support_init.mat \
  --output-path "$USCT_DPS_FWI_RESULT_PATH" \
  --summary-path "$USCT_DPS_FWI_SUMMARY_PATH" \
  --device cuda:0 \
  --seed 1234 \
  --steps 12 \
  --crop-source-size 300 \
  --source-size 480 \
  --sampler-mode reference \
  --prior-mode score_reg \
  --freqs-mhz 0.3 0.3 0.3 0.35 0.35 0.35 0.4 0.4 0.4 0.45 0.45 0.45 \
  --eta 0.1 \
  --guidance-gain 1.15 \
  --prior-strength 1.0 \
  --prior-mask-mode none \
  --score-reg-t 0.10 \
  --score-reg-lambda 0.1 \
  --physics-position pre \
  --physics-inner-steps 1 \
  --output-selection final \
  --no-final-prior-update \
  --gradient-mode slowness_precond \
  --step-strategy line_search \
  --tx-stride 1 \
  --mask-mode support_alpha \
  --support-guidance \
  --sign-conv -1
```

Training a diffusion prior is also external to this repository. A typical
training run belongs in the external project, with generated checkpoints kept
under the workspace `checkpoints/` directory and never committed:

```bash
cd "$USCT_KWAVE_ROOT"
PYTHONPATH="$USCT_KWAVE_ROOT" "$USCT_KWAVE_PYTHON_BIN" \
  openbreastus_diffusion/train_openbreastus.py \
  --data-root /path/to/openbreastus_training_crops \
  --workdir "$USCT_WORKSPACE/checkpoints/openbreastus_diffusion" \
  --include-classes HET FIB FAT EXD \
  --image-size 256 \
  --crop-size 300 \
  --batch-size 32 \
  --epochs 5000 \
  --max-steps 300000 \
  --device-ids 0
```

The external project also supports standalone prior sampling, which is useful
for checking the checkpoint before coupling it to FWI:

```bash
cd "$USCT_KWAVE_ROOT"
PYTHONPATH="$USCT_KWAVE_ROOT" "$USCT_KWAVE_PYTHON_BIN" \
  openbreastus_diffusion/sample_openbreastus.py \
  --checkpoint "$USCT_DPS_CHECKPOINT" \
  --out-dir "$USCT_WORKSPACE/runs/diffusion_prior_samples" \
  --num-samples 16 \
  --batch-size 4 \
  --device-ids 0
```

If the external training or sampling module uses a different name in your
checkout, keep the same contract: train outside `usct-benchlab`, then pass the
resulting checkpoint through `USCT_DPS_CHECKPOINT`.

## Run Benchmarks

Demo suites read these optional case globs:

```bash
export USCT_SYNTHETIC_CASE_GLOB="$USCT_WORKSPACE/data/synthetic_demo/cases/*.h5"
export USCT_NBP_CASE_GLOB="$USCT_WORKSPACE/data/nbpslice2d_demo/cases/*.h5"
export USCT_OPENBREASTUS_CASE_GLOB="$USCT_WORKSPACE/data/openbreastus_demo/cases/*.h5"
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
export USCT_DPS_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
```

Run the suites:

```bash
usct bench --suite configs/benchmarks/synthetic_demo.yaml
usct bench --suite configs/benchmarks/nbpslice2d_demo.yaml
usct bench --suite configs/benchmarks/openbreastus_demo.yaml
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
usct bench --suite configs/benchmarks/diffusion_fwi_kwave_demo.yaml
```

## Output Files

Single-algorithm runs write:

```text
runs/single_cgls/synthetic_circular_sos/result.h5
runs/single_cgls/synthetic_circular_sos/metrics.json
runs/single_cgls/synthetic_circular_sos/metadata.yaml
runs/single_cgls/synthetic_circular_sos/preview.png
```

Benchmark suites write:

```text
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/result.h5
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metrics.json
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/metadata.yaml
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/straight_cgls/synthetic_circular_sos/preview.png
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_summary.csv
runs/usctbench_runs/synthetic_demo_YYYYMMDDTHHMMSSZ/benchmark_report.md
```

`metrics.json` contains per-case image and data-consistency metrics when ground
truth and forward measurements are available. `metadata.yaml` records the
algorithm, config path, case id, runtime, status, and measurement provenance.

## Example Results

The following figures are historical main-branch examples. Their bent/rWave
columns predate the native physics operators and are not validation of the new
implementations. See [physics validation](docs/physics_validation.md) for current
measurements, grids and acceptance boundaries.

OpenBreastUS four-class comparison:

![OpenBreastUS FWI and baseline comparison](docs/assets/openbreastus_readme_fwi_vs_surrogate.png)

NBPslice2D, 2D Acoustic Numerical Breast Phantoms for USCT:

![NBPslice2D FWI and baseline comparison](docs/assets/nbpslice2d_readme_fwi_vs_surrogate.png)

Different algorithms use different measurement assumptions; interpret result
panels together with [docs/algorithms.md](docs/algorithms.md) and case
metadata.

## Troubleshooting

- `algorithm not found`: run `usct list-algorithms` and check the command name.
- Missing `.h5` or `.mat` data: confirm the dataset conversion command
  completed and that the relevant environment variable points to an existing
  path.
- FWI result path missing: set `USCT_KWAVE_FWI_RESULT_PATH` or edit
  `configs/algorithms/fwi_kwave.yaml` to point to the artifact you want to
  ingest.
- NaN/Inf output: inspect `failure_report.md`, check the case units, and lower
  the iteration count or relaxation in the algorithm config.
- No cases matched by glob: print the expanded `USCT_*_CASE_GLOB` value and
  verify that converted cases exist under `data/.../cases/`.
- `matplotlib` or `scikit-image` missing: install the visualization extras with
  `pip install -e ".[viz]"`.

## Development

```bash
black src tests scripts
ruff check src tests scripts --fix
python -m compileall src tests
pytest -q
bash scripts/run_smoke.sh
python scripts/audit_release.py
```

See [docs/development.md](docs/development.md) for release checks and
repository hygiene.

## Citations / Datasets

Please cite the datasets and external tools used in your experiments, including
OpenBreastUS, NBPslice2D, k-Wave, and WaveformInversionUST when applicable. See
[docs/references.bib](docs/references.bib).

## License

This repository is released under the MIT License. See [LICENSE](LICENSE).
