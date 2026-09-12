"""Tests of the canonical physical pairs, including the numerical error trend."""

import importlib.util
import ast
from pathlib import Path

import pytest

from usctbench.operators import eikonal, ray_born, straight_ray
from usctbench.operators.forward import eikonal as old_eikonal
from usctbench.operators.forward import ray_born as old_born
from usctbench.operators.forward import straight_ray as old_straight
from usctbench.operators.adjoint.eikonal import eikonal_adjoint
from usctbench.operators.adjoint.ray_born import ray_born_adjoint
from usctbench.operators.adjoint.straight_ray import backproject


@pytest.fixture(scope="module")
def measured():
    path = Path(__file__).parents[2] / "scripts/verify_numerical_contracts.py"
    spec = importlib.util.spec_from_file_location("numerical_contract_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.measure_contracts()


def test_legacy_imports_share_the_canonical_implementation():
    assert old_straight.StraightRayProjector is straight_ray.StraightRayProjector
    assert old_eikonal.EikonalForward is eikonal.EikonalForward
    assert old_born.RayBornOperator is ray_born.RayBornOperator
    assert backproject is straight_ray.backproject
    assert eikonal_adjoint is eikonal.eikonal_adjoint
    assert ray_born_adjoint is ray_born.ray_born_adjoint
    for module in (straight_ray, eikonal, ray_born):
        text = Path(module.__file__).read_text()
        assert "from usctbench.operators.adjoint." not in text
        assert (
            module.__name__ == "usctbench.operators." + module.__name__.split(".")[-1]
        )
    from usctbench.operators import cuda_green, volume_integral
    from usctbench.operators.forward import cuda_green as old_cuda
    from usctbench.operators.forward import volume_integral as old_volume

    assert old_cuda.CudaBornJacobian is cuda_green.CudaBornJacobian
    assert old_cuda.solve_fields is cuda_green.solve_fields
    assert old_volume.VolumeIntegralGreen is volume_integral.VolumeIntegralGreen


def test_production_imports_do_not_use_compatibility_physical_modules():
    root = Path(__file__).parents[2]
    names = {"straight_ray", "eikonal", "ray_born", "volume_integral", "cuda_green"}
    for path in list((root / "src").rglob("*.py")) + list(
        (root / "scripts").rglob("*.py")
    ):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "usctbench.algorithms.ray":
                    assert "StraightRayProjector" not in {
                        a.name for a in node.names
                    }, path
                if module in {
                    "usctbench.operators.forward",
                    "usctbench.operators.adjoint",
                }:
                    assert not names.intersection(a.name for a in node.names), path
                assert not any(
                    module == f"usctbench.operators.{direction}.{name}"
                    for direction in ("forward", "adjoint")
                    for name in names
                ), path
            elif isinstance(node, ast.Import):
                assert not any(
                    a.name == f"usctbench.operators.{direction}.{name}"
                    for a in node.names
                    for direction in ("forward", "adjoint")
                    for name in names
                ), path
    for name in names:
        path = root / "src/usctbench/operators" / f"{name}.py"
        assert "operators.forward" not in path.read_text()
        assert "operators.adjoint" not in path.read_text()


@pytest.mark.parametrize("backend", ["reference", "csr"])
def test_straight_matches_explicit_transpose_and_homogeneous_integral(
    measured, backend
):
    row = measured["straight"][backend]
    assert row["adjoint_error"] < 2e-13
    assert row["explicit_forward_max_abs"] < 2e-14
    assert row["explicit_adjoint_max_abs"] < 2e-14
    assert row["horizontal_integral_error"] < 2e-14


@pytest.mark.parametrize("order", ["1", "2"])
def test_eikonal_fixed_active_derivative_trend_and_adjoint(measured, order):
    row = measured["eikonal"][order]
    assert row["adjoint_error"] < 2e-12
    differences = row["directional_difference"]
    assert all(r["same_active_parents"] for r in differences)
    errors = [r["relative_forward_difference_error"] for r in differences]
    assert errors[-1] < 0.05 * errors[0]
    assert all(b < a for a, b in zip(errors, errors[1:]))


def test_born_real_complex_adjoint_and_second_order_remainder(measured):
    assert measured["born"]["adjoint_error"] < 2e-12
    assert all(1.9 < order < 2.1 for order in measured["born"]["remainder_orders"])


def test_volume_integral_exact_discrete_derivative(measured):
    row = measured["volume_integral"]
    assert row["adjoint_error"] < 2e-12
    errors = [
        r["relative_central_difference_error"] for r in row["directional_difference"]
    ]
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]
