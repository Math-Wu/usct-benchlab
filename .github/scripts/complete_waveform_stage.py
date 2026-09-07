"""Complete two explicitly scoped integration stages on the development branch.

Each stage is formatted and passes the complete validation chain before its
source changes are committed and pushed. A concurrent branch update is never
force-overwritten. This script is not part of the runtime algorithm package.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "work/physics-agent-validation"
LOGS = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "usct-stage-validation"
FIELDS = ("time_water_reference", "frequency_water_reference", "source_spectrum")


def run(*args, log=None):
    process = subprocess.run(
        list(args), cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    if log is not None:
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / log).write_text(process.stdout, encoding="utf-8")
    print(process.stdout, flush=True)
    if process.returncode:
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(args)}")
    return process.stdout.strip()


def write(path, content):
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def rename_field(node, field):
    class Rename(ast.NodeTransformer):
        def visit_Constant(self, value):
            if value.value == "water_reference":
                return ast.copy_location(ast.Constant(field), value)
            return value

        def visit_Attribute(self, value):
            self.generic_visit(value)
            if value.attr == "water_reference":
                value.attr = field
            return value
    return Rename().visit(copy.deepcopy(node))


def extend_io():
    path = ROOT / "src/usctbench/core/io.py"
    source = path.read_text()
    tree = ast.parse(source)

    class Extend(ast.NodeTransformer):
        def extend_sequence(self, node):
            self.generic_visit(node)
            strings = [v.value for v in node.elts if isinstance(v, ast.Constant)]
            if "water_reference" in strings:
                node.elts.extend(ast.Constant(name) for name in FIELDS if name not in strings)
            return node

        visit_Tuple = extend_sequence
        visit_List = extend_sequence

        def visit_Call(self, node):
            self.generic_visit(node)
            keywords = {keyword.arg: keyword for keyword in node.keywords}
            if "water_reference" in keywords:
                node.keywords.extend(
                    ast.keyword(arg=name, value=rename_field(keywords["water_reference"].value, name))
                    for name in FIELDS if name not in keywords
                )
            return node

        def visit_Dict(self, node):
            self.generic_visit(node)
            keys = [key.value if isinstance(key, ast.Constant) else None for key in node.keys]
            if "water_reference" in keys:
                original = node.values[keys.index("water_reference")]
                for name in FIELDS:
                    if name not in keys:
                        node.keys.append(ast.Constant(name))
                        node.values.append(rename_field(original, name))
            return node

        def visit_Expr(self, node):
            self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "_write_dataset" and len(call.args) >= 2
                    and isinstance(call.args[1], ast.Constant)
                    and call.args[1].value == "water_reference"):
                return [node] + [rename_field(node, name) for name in FIELDS]
            return node

    tree = Extend().visit(tree)
    # Compression changes representation only, never numerical values.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_write_dataset":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "create_dataset"):
                    keywords = {keyword.arg for keyword in call.keywords}
                    if "compression" not in keywords and not any(k.arg is None for k in call.keywords):
                        data = next((k.value for k in call.keywords if k.arg == "data"), None)
                        if data is not None:
                            condition = ast.parse("np.asarray(value).ndim >= 3 and np.asarray(value).size > 1024", mode="eval").body
                            call.keywords.append(ast.keyword(
                                arg="compression",
                                value=ast.IfExp(test=condition, body=ast.Constant("gzip"), orelse=ast.Constant(None)),
                            ))
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n")


def waveform_stage():
    waveform = ROOT / "src/usctbench/data/waveforms.py"
    calibration = ROOT / "src/usctbench/evaluation/calibration.py"
    if not waveform.exists() or not calibration.exists():
        raise RuntimeError("pressure and calibration modules must be committed first")
    path = ROOT / "src/usctbench/core/schema.py"
    source = path.read_text()
    declaration = "    water_reference: np.ndarray | None = None"
    if not all(f"    {name}:" in source for name in FIELDS):
        if declaration not in source:
            raise RuntimeError("unexpected MeasurementSpec layout; refuse an ambiguous patch")
        source = source.replace(declaration, declaration + "\n" + "\n".join(
            f"    {name}: np.ndarray | None = None" for name in FIELDS
        ), 1)
        marker = '        "water_reference",'
        if marker not in source:
            raise RuntimeError("cannot identify the optional-array validator")
        source = source.replace(marker, marker + "\n" + "\n".join(
            f'        "{name}",' for name in FIELDS
        ), 1)
        path.write_text(source)
    extend_io()

    path = ROOT / "src/usctbench/data/conversion.py"
    legacy = ROOT / "src/usctbench/data/_image_conversion.py"
    if not legacy.exists():
        shutil.copyfile(path, legacy)
    legacy_tree = ast.parse(legacy.read_text())
    exports = sorted(node.name for node in legacy_tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                     and not node.name.startswith("_")
                     and node.name != "convert_kwave_channel_mat")
    imports = "\n".join(f"from ._image_conversion import {name} as {name}" for name in exports)
    path.write_text('''"""Dataset conversions with explicit separation of pressure and image surrogates."""
from __future__ import annotations

from . import _image_conversion
from .waveforms import convert_kwave_waveforms
''' + imports + '''

def __getattr__(name):
    # Preserve historical helper imports without a second numerical implementation.
    return getattr(_image_conversion, name)


def convert_kwave_channel_mat(
    mat_path, out_dir, *, case_id_prefix=None, output_shape=(64, 64),
    n_transducers=32, retain_waveforms=True, **waveform_options,
):
    """Preserve RF by default; retain_waveforms=False explicitly selects the old surrogate."""
    if not isinstance(retain_waveforms, bool):
        raise ValueError("retain_waveforms must be a boolean")
    if not retain_waveforms:
        if waveform_options:
            raise ValueError("waveform options cannot be used with the legacy surrogate")
        return _image_conversion.convert_kwave_channel_mat(
            mat_path, out_dir, case_id_prefix=case_id_prefix,
            output_shape=output_shape, n_transducers=n_transducers,
        )
    return convert_kwave_waveforms(
        mat_path, out_dir, case_id=case_id_prefix, output_shape=output_shape,
        n_transducers=n_transducers, **waveform_options,
    )
''')
    # Avoid a conversion-module import cycle while preserving private legacy APIs.
    waveform.write_text(waveform.read_text().replace(
        "from usctbench.data.conversion import (", "from usctbench.data._image_conversion import ("
    ))

    path = ROOT / "src/usctbench/algorithms/rwave.py"
    tree = ast.parse(path.read_text())
    source_expression = ast.parse(
        'config.parameters.get("source_spectrum", case.measurement.source_spectrum)', mode="eval"
    ).body
    patched = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RayBornOperator":
            if not any(keyword.arg == "source_spectrum" for keyword in node.keywords):
                node.keywords.append(ast.keyword(arg="source_spectrum", value=copy.deepcopy(source_expression)))
            patched += 1
    if patched == 0:
        raise RuntimeError("could not identify the native Ray-Born operator construction")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == "run":
                    guard = ast.parse('''
if case.metadata.get("frequency_transform") and config.parameters.get("source_spectrum", case.measurement.source_spectrum) is None:
    raise ValueError("transformed pressure requires an independently calibrated source_spectrum; unit-source pressure must not be assumed")
''').body
                    location = 1 if method.body and isinstance(method.body[0], ast.Expr) and isinstance(method.body[0].value, ast.Constant) else 0
                    method.body[location:location] = guard
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n")
    write("tests/data/test_pressure_integration.py", PRESSURE_TESTS)
    write("docs/waveform_integration.md", WAVEFORM_DOC)


def operator_stage():
    path = ROOT / "src/usctbench/algorithms/ray.py"
    tree = ast.parse(path.read_text())
    definitions = {node.name: node for node in tree.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    if "StraightRayProjector" not in definitions:
        raise RuntimeError("projector is already moved or has an unexpected definition")
    selected = {"StraightRayProjector"}
    while True:
        referenced = {item.id for name in selected for item in ast.walk(definitions[name])
                      if isinstance(item, ast.Name)}
        expanded = selected | (referenced & definitions.keys())
        if expanded == selected:
            break
        selected = expanded
    nodes = [node for node in tree.body if node in [definitions[name] for name in selected]]
    projector = definitions["StraightRayProjector"]
    methods = {node.name: node for node in projector.body if isinstance(node, ast.FunctionDef)}
    reverse_name = next((name for name in ("adjoint", "backproject", "rmatvec") if name in methods), None)
    if reverse_name is None:
        raise RuntimeError("no unambiguous projector adjoint method found")
    reverse = copy.deepcopy(methods[reverse_name])
    reverse.name = "straight_ray_adjoint"
    reverse.decorator_list = []
    method = methods[reverse_name]
    arguments = method.args
    positionals = [ast.Name(arg=arg.arg, ctx=ast.Load()) if False else ast.Name(id=arg.arg, ctx=ast.Load())
                   for arg in arguments.posonlyargs + arguments.args]
    if arguments.vararg:
        positionals.append(ast.Starred(value=ast.Name(id=arguments.vararg.arg, ctx=ast.Load()), ctx=ast.Load()))
    keywords = [ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load())) for arg in arguments.kwonlyargs]
    if arguments.kwarg:
        keywords.append(ast.keyword(arg=None, value=ast.Name(id=arguments.kwarg.arg, ctx=ast.Load())))
    method.body = [ast.Return(value=ast.Call(func=ast.Name(id="straight_ray_adjoint", ctx=ast.Load()), args=positionals, keywords=keywords))]
    referenced = {item.id for node in nodes + [reverse] for item in ast.walk(node)
                  if isinstance(item, ast.Name)}
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = {alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                     for alias in node.names}
            if names & referenced or isinstance(node, ast.ImportFrom) and node.module == "__future__":
                item = copy.deepcopy(node)
                if isinstance(item, ast.ImportFrom) and item.level:
                    item.module = importlib.util.resolve_name("." * item.level + (item.module or ""), "usctbench.algorithms")
                    item.level = 0
                imports.append(item)
    constants = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in referenced for target in node.targets):
            constants.append(copy.deepcopy(node))
    forward = ast.Module(body=copy.deepcopy(imports) + ast.parse(
        "from usctbench.operators.adjoint.straight_ray import straight_ray_adjoint"
    ).body + constants + nodes, type_ignores=[])
    # The adjoint is numerical code, not an alias back to an algorithm solver.
    reverse_globals = {item.id for item in ast.walk(reverse) if isinstance(item, ast.Name)}
    helpers = [copy.deepcopy(definitions[name]) for name in selected - {"StraightRayProjector"}
               if name in reverse_globals]
    adjoint = ast.Module(body=copy.deepcopy(imports) + copy.deepcopy(constants) + helpers + [reverse], type_ignores=[])
    tree.body = [node for node in tree.body if node not in nodes]
    compatibility = ast.ImportFrom(module="usctbench.operators.forward.straight_ray",
                                   names=[ast.alias(name=name, asname=name) for name in sorted(selected)], level=0)
    index = max((i + 1 for i, node in enumerate(tree.body) if isinstance(node, (ast.Import, ast.ImportFrom))), default=1)
    tree.body.insert(index, compatibility)
    for module, target in (
        (forward, "src/usctbench/operators/forward/straight_ray.py"),
        (adjoint, "src/usctbench/operators/adjoint/straight_ray.py"),
        (tree, "src/usctbench/algorithms/ray.py"),
    ):
        ast.fix_missing_locations(module)
        write(target, ast.unparse(module) + "\n")

    path = ROOT / "src/usctbench/benchmark/runner.py"
    tree = ast.parse(path.read_text())
    found = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_add_data_domain_metrics":
            if "result" not in {arg.arg for arg in node.args.args + node.args.kwonlyargs}:
                raise RuntimeError("unexpected data-domain evaluation signature")
            guard = ast.parse('''
if not (str(result.algorithm).startswith("straight_ray") or result.algorithm in {"cgls", "sirt", "sart", "attenuation"}):
    result.artifacts.setdefault("data_domain_metric_policy", "native operator metrics only; straight-ray fallback disabled")
    return
''').body
            index = 1 if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else 0
            node.body[index:index] = guard
            found = True
    if not found:
        raise RuntimeError("data-domain metric fallback was not found")
    ast.fix_missing_locations(tree)
    path.write_text(ast.unparse(tree) + "\n")
    write("tests/operators/test_straight_namespace.py", OPERATOR_TESTS)


def validate_and_publish(stage, message):
    run(sys.executable, "-m", "black", "src", "tests", "scripts", log=f"{stage}-format.log")
    # Restrict automatic lint repair to unused imports introduced by moving code.
    run(sys.executable, "-m", "ruff", "check", "--select", "F401", "--fix", "src", log=f"{stage}-imports.log")
    run(sys.executable, "-m", "black", "--check", "src", "tests", "scripts", log=f"{stage}-black.log")
    run(sys.executable, "-m", "ruff", "check", "src", "tests", "scripts", log=f"{stage}-ruff.log")
    run(sys.executable, "-m", "compileall", "-q", "src", "tests", log=f"{stage}-compile.log")
    run(sys.executable, "-m", "pytest", "-q", log=f"{stage}-pytest.log")
    run(sys.executable, "-m", "usctbench.cli", "--help", log=f"{stage}-help.log")
    run(sys.executable, "-m", "usctbench.cli", "list-algorithms", log=f"{stage}-registry.log")
    run("bash", "scripts/run_smoke.sh", log=f"{stage}-smoke.log")
    run("git", "add", "src", "tests", "docs")
    run(sys.executable, "scripts/audit_release.py", log=f"{stage}-audit.log")
    tree = run("git", "write-tree")
    test_output = (LOGS / f"{stage}-pytest.log").read_text()
    report = {
        "stage": stage, "status": "passed", "source_tree_before_report": tree,
        "python": platform.python_version(), "pytest_summary": test_output.splitlines()[-1],
        "pytest_log_sha256": hashlib.sha256(test_output.encode()).hexdigest(),
        "workflow_run": os.environ.get("GITHUB_RUN_ID"),
        "scope": "Python numerical and integration tests; no MATLAB or clinical-data validation",
    }
    write(f"docs/validation_{stage}.json", json.dumps(report, indent=2) + "\n")
    run("git", "add", "docs")
    run("git", "-c", "user.name=github-actions[bot]", "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com", "commit", "-m", message)
    run("git", "push", "origin", f"HEAD:refs/heads/{BRANCH}")


PRESSURE_TESTS = r'''import h5py
import numpy as np
import pytest

from usctbench.core.io import read_case_hdf5, write_case_hdf5
from usctbench.data.conversion import convert_kwave_channel_mat
from usctbench.data.waveforms import pressure_spectrum, read_kwave_waveform_case, to_frequency_case
from usctbench.evaluation.calibration import fit_source_spectrum


def fixture_file(path):
    n, nt = 5, 160
    time = (np.arange(nt) + 7) * 1e-7
    pressure = np.arange(n * n * nt, dtype=np.float32).reshape(n, n, nt)
    angle = np.arange(n) * 2 * np.pi / n
    positions = np.column_stack((0.02 * np.cos(angle), 0.02 * np.sin(angle)))
    with h5py.File(path, "w") as f:
        f["time"] = time
        f["full_dataset"] = pressure
        f["water_reference"] = pressure * 0.5
        f["transducerPositionsXY"] = positions
        f["C"] = np.full((8, 6), 1510.0)
        f["xi_orig"] = (np.arange(8) - 3.5) * 1e-3
        f["yi_orig"] = (np.arange(6) - 2.5) * 2e-3
    return pressure, time, positions


def test_default_conversion_preserves_pressure_without_oracle_features(tmp_path):
    path = tmp_path / "acquisition.mat"
    raw, time, positions = fixture_file(path)
    records = convert_kwave_channel_mat(path, tmp_path / "cases", output_shape=(3, 4), n_transducers=3)
    case = read_case_hdf5(records[0]["path"])
    selected = np.array([0, 1, 3])
    np.testing.assert_array_equal(case.measurement.time_data, raw[selected][:, selected])
    np.testing.assert_array_equal(case.measurement.time_axis_s, time)
    np.testing.assert_allclose(case.geometry.tx_pos_m, positions[selected, ::-1])
    assert case.measurement.tof_s is None
    assert case.measurement.delta_tof_s is None
    assert case.measurement.log_amp is None
    assert case.metadata["ground_truth_used_for_preprocessing"] is False
    with h5py.File(path, "r+") as f:
        f["C"][...] = 1720.0
    changed = read_kwave_waveform_case(path, output_shape=(3, 4), n_transducers=3)
    np.testing.assert_array_equal(changed.measurement.time_data, case.measurement.time_data)
    assert changed.metadata["reference_sound_speed_mps"] == case.metadata["reference_sound_speed_mps"]


def test_fourier_sign_and_absolute_time_origin():
    n, dt = 256, 1e-7
    t = (np.arange(n) + 11.25) * dt
    f = 9 / (n * dt)
    phase = 0.63
    data = np.cos(2 * np.pi * f * t + phase)[None, None, :]
    actual = pressure_spectrum(data, t, [f])[0, 0, 0]
    np.testing.assert_allclose(actual, 0.5 * n * dt * np.exp(-1j * phase), rtol=1e-12)
    with pytest.raises(ValueError, match="Nyquist"):
        pressure_spectrum(data, t, [0.5 / dt])


def test_pressure_water_and_complex_source_round_trip(tmp_path):
    path = tmp_path / "acquisition.mat"
    fixture_file(path)
    case = read_kwave_waveform_case(path, output_shape=(3, 4), n_transducers=3)
    frequency = to_frequency_case(case, [1e5, 2e5])
    frequency.measurement.source_spectrum = np.full((2, 3), 1.2 + 0.8j)
    output = tmp_path / "frequency.h5"
    write_case_hdf5(frequency, output)
    restored = read_case_hdf5(output)
    for name in ("time_data", "freq_data", "time_water_reference", "frequency_water_reference", "source_spectrum"):
        np.testing.assert_array_equal(getattr(restored.measurement, name), getattr(frequency.measurement, name))


def test_calibration_never_uses_held_out_receivers():
    rng = np.random.default_rng(82)
    green = rng.normal(size=(3, 4, 7)) + 1j * rng.normal(size=(3, 4, 7))
    q = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    water = green * q[..., None]
    train = np.ones(green.shape, dtype=bool)
    train[..., -2:] = False
    water[..., -2:] = 1e99j
    np.testing.assert_allclose(fit_source_spectrum(green, water, train), q, rtol=1e-12, atol=1e-12)
    train[0] = False
    with pytest.raises(ValueError, match="independent source calibration"):
        fit_source_spectrum(green, water, train)
'''

OPERATOR_TESTS = r'''import numpy as np

from usctbench.algorithms.ray import StraightRayProjector as LegacyProjector
from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.forward.straight_ray import StraightRayProjector


def test_projector_compatibility_is_a_reexport_not_duplicate_physics():
    assert LegacyProjector is StraightRayProjector
    assert StraightRayProjector.__module__ == "usctbench.operators.forward.straight_ray"


def test_moved_operator_retains_its_adjoint_product():
    grid = GridSpec(shape=(6, 7), spacing_m=(0.01, 0.013), origin_m=(-0.03, -0.0455))
    angle = np.arange(5) * 2 * np.pi / 5
    positions = 0.06 * np.column_stack((np.sin(angle), np.cos(angle)))
    geometry = GeometrySpec(type="custom", tx_pos_m=positions, rx_pos_m=positions)
    op = StraightRayProjector.from_grid_geometry(grid, geometry)
    forward = next(getattr(op, name) for name in ("forward", "project", "matvec") if hasattr(op, name))
    adjoint = next(getattr(op, name) for name in ("adjoint", "backproject", "rmatvec") if hasattr(op, name))
    rng = np.random.default_rng(93)
    x = rng.normal(size=grid.shape)
    predicted = np.asarray(forward(x))
    y = rng.normal(size=predicted.shape)
    back = np.asarray(adjoint(y))
    np.testing.assert_allclose(np.vdot(predicted, y), np.vdot(x.ravel(), back.ravel()), rtol=1e-12, atol=1e-12)
'''

WAVEFORM_DOC = '''# Pressure-preserving k-Wave integration

`convert_kwave_channel_mat` now preserves RF by default. The former speed-map
travel-time conversion remains explicitly available with `retain_waveforms=False`;
that path is still an oracle/image surrogate, not measured pressure.

```python
from usctbench.data.waveforms import read_kwave_waveform_case, to_frequency_case
case = read_kwave_waveform_case("acquisition.mat", n_transducers=32,
                                channel_axis_order="tx,rx,time")
frequency_case = to_frequency_case(case, [2e5, 3e5, 4e5])
```

Canonical arrays are `(tx, rx, time)` and `(frequency, tx, rx)`. MATLAB v7.3/HDF5
reverses the upstream MATLAB channel axes; equal tx/rx lengths cannot disambiguate
ordering, so other exports must supply their actual axis order. The image axes
`xi_orig` and `yi_orig` are required to preserve the physical field of view.

The phasor convention is `p(t)=Re(P exp(-i omega t))`. The analysis transform is
`dt * sum(p(t) exp(+i omega t))` and uses absolute timestamps, including a nonzero
start. No implicit window, source normalization, nearest-bin rounding or truth-
derived features are introduced. `time_water_reference`,
`frequency_water_reference` and complex `source_spectrum` survive case I/O.

Transformed RF requires independently calibrated source spectra for native
Ray-Born inversion. `fit_source_spectrum` can fit a per-frequency/transmitter
source from an independent water acquisition, using an explicitly supplied
training mask only. Groups with no training water data are rejected rather than
calibrated with validation data. In particular, whole-frequency holdouts generally
require an independent calibrated source model. Source wavelet samples alone
are not assumed to encode the k-Wave injection normalization and dimensionality.

An optional Hilbert-envelope threshold picker is available for measured first
arrivals. It is a reference picker, not a validated clinical timing estimator.
Without a measured water reference, absolute arrival times require independent
trigger calibration.

## Scope of the numerical methods

Bent-ray uses the native fast-marching Eikonal solver and its discrete tangent/
adjoint. Native RWave currently means a **fixed-background Ray-Born reference**,
not a full reproduction of the nonlinear upstream r-Wave frequency continuation
and ray-linking implementation. It uses complex finite-frequency scattering,
not the straight-ray forward matrix. Production FWI continues to use the existing
WaveformInversionUST adapter; `fwi_tiny` remains a smoke-test implementation.

The existing native Eikonal/Ray-Born solvers support receiver/frequency validation
and auditable OR stopping. This integration stage does **not** claim that all
legacy straight-ray loops or opaque external MATLAB iterations already expose
that same live stopping contract. Those remain separate integration work.

Python unit/integration tests do not constitute MATLAB, large breast-data, 3-D
calibration, or clinical validation. Per-stage JSON records identify exactly the
Python validation that ran before publication.

## Primary references

- Rehman Ali, WaveformInversionUST: https://github.com/rehmanali1994/WaveformInversionUST
- Javaherian, Lucka and Cox (2020), DOI 10.1088/1361-6420/abc0fc.
- Javaherian and Cox (2021), DOI 10.1088/1361-6420/ac28ed.
- r-Wave source: https://github.com/Ash1362/ray-based-quantitative-ultrasound-tomography
'''


def main():
    if os.environ.get("GITHUB_REF_NAME") != BRANCH:
        raise RuntimeError("this publisher is restricted to the explicitly authorized development branch")
    waveform_stage()
    validate_and_publish("pressure", "feat(data): integrate calibrated pressure-preserving k-Wave cases; full validation passed")
    operator_stage()
    validate_and_publish("operators", "refactor(operators): separate straight forward and adjoint numerics; prevent cross-physics residual fallback")


if __name__ == "__main__":
    main()
