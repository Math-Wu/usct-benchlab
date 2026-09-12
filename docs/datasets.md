# Datasets

`usct-benchlab` keeps raw datasets outside Git and converts selected examples
to standard `USCTCase` HDF5 files.

## Synthetic Demo

Synthetic cases are deterministic and suitable for local smoke tests:

```bash
usct data make-synthetic-smoke \
  --out "$USCT_WORKSPACE/data/synthetic_demo" \
  --shape 48 \
  --n-transducers 48
```

The converter writes HDF5 cases under:

```text
$USCT_WORKSPACE/data/synthetic_demo/cases/
```

## OpenBreastUS

Inspect:

```bash
usct data inspect-openbreastus \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_RUN_ROOT/openbreastus_index.json"
```

Create demo cases:

```bash
usct data make-quality \
  --root "$USCT_DATA_ROOT" \
  --out "$USCT_WORKSPACE/data/openbreastus_demo" \
  --cases-per-density 1 \
  --converted-shape 256 \
  --n-transducers 128
```

The conversion records source paths and measurement assumptions in case
metadata. Raw OpenBreastUS files should not be committed.

## NBPslice2D

Set the ZIP path:

```bash
export USCT_NBP_ZIP_PATH=/path/to/NBPslices2D.zip
```

Inspect:

```bash
usct data inspect-nbpslice2d \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_RUN_ROOT/nbpslice2d_index.json"
```

Create demo cases:

```bash
usct data make-nbp-quality \
  --zip "$USCT_NBP_ZIP_PATH" \
  --out "$USCT_WORKSPACE/data/nbpslice2d_demo" \
  --cases-per-type 1 \
  --converted-shape 256 \
  --n-transducers 128
```

NBPslice2D slices are acoustic property maps. Converted cases generate
the feature fields required by the classical baselines.

## Data Hygiene

The repository ignores:

```text
data/
runs/
external/
checkpoints/
*.h5
*.hdf5
*.mat
*.npy
*.npz
*.pt
*.ckpt
```

Only source, configs, tests, docs, and small README figures belong in Git.

## Imported quantities

The case schema imports sound speed and supported pressure/ToF observations.
Additional raw NBPslices2D `att`/`y` and k-Wave material maps are intentionally
ignored; they are not converted into reconstruction labels or features. Density
metadata and geometry handling remain unchanged. Older attenuation-bearing
BenchLab HDF5 files have no compatibility guarantee; see [parameter contract](parameter_contract.md).
