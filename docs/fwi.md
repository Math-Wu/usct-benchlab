# FWI Adapter

The FWI path is exposed through two adapter-style commands:

- `fwi_kwave_adapter` ingests pure external k-Wave/FWI artifacts.
- `diffusion_fwi_kwave_adapter` ingests external diffusion-prior k-Wave/FWI
  DPS artifacts.

Both commands report external reconstructions in the same artifact format as
the classical baselines. They do not vendor the external solver, PyTorch model,
MATLAB runtime, or k-Wave binary into `usct-benchlab`.

## Mathematical Model

The FWI objective is summarized as

$$
\min_c
\frac{1}{2}\sum_{\omega,s,r}
\left|
\hat p_s(\omega,r;c)-\hat p_{sr}^{\mathrm{obs}}(\omega)
\right|^2
+\lambda R(c).
$$

The pressure prediction $\hat p_s(\omega,r;c)$ is produced by an external
full-wave solver.

## Configure an Existing Result

Artifact import defaults to the final iteration. GT-based `best` selection needs
an explicit `allow_ground_truth_selection: true` and is not allowed for truth-free
validation. This adapter cannot infer the production solver's actual stop reason
from a final image and refuses `stopping` / `evaluation` settings it cannot enforce.

Set:

```bash
export USCT_KWAVE_FWI_RESULT_PATH=/path/to/fwi_result.mat
export USCT_KWAVE_ROOT=/path/to/external/USCT_kwave
export USCT_KWAVE_PYTHON_BIN=/path/to/python
```

Then run:

```bash
usct run fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/fwi_kwave.yaml \
  --out runs/single_fwi
```

The result MAT file must contain `VEL_ESTIM`, the final reconstructed sound
speed image. Optional fields enable richer reports:

| Field | Meaning |
| --- | --- |
| `C_INTERP` | Ground-truth sound speed used for FWI-native metrics. |
| `ATTEN_ESTIM` | Final attenuation estimate. |
| `VEL_ESTIM_ITER` | Sound-speed images over iterations. |
| `ATTEN_ESTIM_ITER` | Attenuation images over iterations. |
| `LOSS_ITER` | Per-iteration loss curve and iteration count. |
| `VEL_INIT` | Initial sound-speed model. |
| `ATTEN_INIT_USED` | Initial attenuation model. |
| `psnr_value`, `ssim_value` | Native external metrics, if saved. |
| `datasetPath` | External dataset path recorded by the FWI pipeline. |

If `run_external: true` is enabled in the config, `USCT_KWAVE_ROOT` must point
to the external solver checkout. If `USCT_KWAVE_PYTHON_BIN` is unset, the
adapter uses the current Python interpreter.

## Configure an Existing Diffusion + FWI Result

This pre-existing optional adapter is unchanged by the physics-validation work.

Set:

```bash
export USCT_DPS_FWI_RESULT_PATH=/path/to/dps_result.mat
export USCT_DPS_FWI_SUMMARY_PATH=/path/to/dps_result.json
export USCT_DPS_DATASET_PATH=/path/to/kwave_dataset.mat
export USCT_DPS_CHECKPOINT=/path/to/diffusion_checkpoint.pth
```

Then run:

```bash
usct run diffusion_fwi_kwave_adapter \
  --case "$USCT_WORKSPACE/data/openbreastus_demo/cases/example_case.h5" \
  --config configs/algorithms/diffusion_fwi_kwave.yaml \
  --out runs/single_diffusion_fwi
```

The DPS MAT file may contain any of these sound-speed fields:

| Field | Meaning |
| --- | --- |
| `VEL_DPS_PHYS` | Preferred selected reconstruction in physics-grid coordinates. |
| `VEL_DPS_VIEW` | Preferred selected reconstruction in display/view coordinates. |
| `VEL_FINAL_PHYS` | Final physical-grid reconstruction fallback. |
| `VEL_FINAL_VIEW` | Final display/view reconstruction fallback. |
| `VEL_INIT_VIEW` | Initial model, used for diagnostic metrics when available. |
| `GT_VIEW` | External ground truth, used only when the case has no ground truth. |

The JSON summary is optional, but it is the preferred source for provenance:
checkpoint, dataset path, frequency schedule, diffusion-prior settings, and
selected-step metadata.

## Run the Demo Suite

```bash
export USCT_KWAVE_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
usct bench --suite configs/benchmarks/fwi_kwave_demo.yaml
```

```bash
export USCT_DPS_FWI_CASE_GLOB="$USCT_WORKSPACE/data/fwi_kwave_demo/cases/*.h5"
usct bench --suite configs/benchmarks/diffusion_fwi_kwave_demo.yaml
```

## Outputs

The adapter writes the standard package artifacts:

```text
runs/single_fwi/<case_id>/result.h5
runs/single_fwi/<case_id>/metrics.json
runs/single_fwi/<case_id>/metadata.yaml
runs/single_fwi/<case_id>/preview.png
```

When ground truth is present, metrics include image quality fields such as
RMSE, SSIM, PSNR, and baseline-improvement values.

## Scope

`fwi_kwave_adapter` and `diffusion_fwi_kwave_adapter` are adapter routes. They
do not vendor a production k-Wave, MATLAB, or diffusion model implementation
into this package. External solver setup remains the responsibility of the user
environment.
