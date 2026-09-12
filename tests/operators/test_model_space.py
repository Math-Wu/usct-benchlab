import numpy as np
import pytest

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.eikonal import EikonalForward
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.model_space import (
    BilinearBasis,
    FineGridRegularizer,
    ReducedLinearOperator,
    SpatialGradient,
)
from usctbench.solvers.least_squares import normal_regularizer, regularizer


def fixture():
    grid = GridSpec(shape=(15, 17), spacing_m=(0.001, 0.0013))
    geom = GeometrySpec(
        tx_pos_m=[[-0.001, 0.003], [0.005, -0.001]],
        rx_pos_m=[[0.016, 0.015], [0.01, 0.023]],
    )
    return grid, geom, BilinearBasis(grid, (5, 6))


def test_basis_constants_bounds_adjoint_and_identity_grid():
    grid, _, basis = fixture()
    rng = np.random.default_rng(847)
    x, y = rng.normal(size=basis.shape), rng.normal(size=grid.shape)
    np.testing.assert_allclose(basis.forward(np.ones(basis.shape)), 1, atol=2e-16)
    assert basis.forward(x).min() >= x.min()
    assert basis.forward(x).max() <= x.max()
    np.testing.assert_allclose(
        np.vdot(basis.forward(x), y), np.vdot(x, basis.adjoint(y)), rtol=2e-14
    )
    identity = BilinearBasis(grid, grid.shape)
    np.testing.assert_array_equal(identity.forward(y), y)
    np.testing.assert_array_equal(identity.adjoint(y), y)
    # Replication at the boundary, not a zero-valued halo.
    assert basis.forward(x)[0, 0] == x[0, 0]
    assert basis.forward(x)[-1, -1] == x[-1, -1]


def test_physical_gradient_and_reduced_transpose_on_rectangular_grid():
    grid, _, basis = fixture()
    gradient = SpatialGradient(grid, 0.002)
    rng = np.random.default_rng(514)
    image = rng.normal(size=grid.shape)
    expected = np.concatenate(
        [
            (np.diff(image, axis=0) * 0.002 / grid.spacing_m[0]).ravel(),
            (np.diff(image, axis=1) * 0.002 / grid.spacing_m[1]).ravel(),
        ]
    )
    np.testing.assert_allclose(gradient.forward(image), expected, atol=2e-15)
    np.testing.assert_array_equal(gradient.forward(np.ones(grid.shape)), 0)
    edges = rng.normal(size=expected.shape)
    np.testing.assert_allclose(
        np.vdot(gradient.forward(image), edges),
        np.vdot(image, gradient.adjoint(edges)),
        rtol=1e-14,
    )
    reduced = FineGridRegularizer(basis, gradient)
    coarse = rng.normal(size=basis.shape)
    np.testing.assert_allclose(
        np.vdot(reduced.forward(coarse), edges),
        np.vdot(coarse, reduced.adjoint(edges)),
        rtol=1e-14,
    )
    line = SpatialGradient(GridSpec(shape=(1, 3), spacing_m=(1, 1)), 1)
    np.testing.assert_array_equal(line.forward(np.array([[1.0, 2.0, 4.0]])), [1.0, 2.0])
    with pytest.raises(ValueError, match="finite real edge"):
        gradient.adjoint(np.zeros(grid.shape))


def test_straight_composition_and_norms():
    grid, geom, basis = fixture()
    a = StraightRayProjector.from_grid_geometry(grid, geom)
    ab = ReducedLinearOperator(a, basis)
    rng = np.random.default_rng(399)
    x, y = rng.normal(size=basis.shape), rng.normal(size=ab.n_rays)
    np.testing.assert_allclose(ab.forward(x), a.forward(basis.forward(x)), atol=1e-16)
    np.testing.assert_allclose(ab.adjoint(y), basis.adjoint(a.adjoint(y)), atol=1e-16)
    np.testing.assert_allclose(
        np.vdot(ab.forward(x), y), np.vdot(x, ab.adjoint(y)), rtol=1e-13
    )
    dense = ab.matrix.toarray()
    for power in (1, 2):
        np.testing.assert_allclose(ab.row_norms(power), (dense**power).sum(axis=1))
        np.testing.assert_allclose(
            ab.col_norms(power), (dense**power).sum(axis=0).reshape(basis.shape)
        )


@pytest.mark.parametrize("kind", ["identity", "laplacian", (0.7, 1.3)])
def test_fine_penalty_exact_adjoint_gradient_and_original_units(kind):
    grid, _, basis = fixture()
    reg = FineGridRegularizer(basis, kind)
    rng = np.random.default_rng(313)
    x, direction = rng.normal(size=(2, *basis.shape))
    y = rng.normal(size=grid.shape)
    np.testing.assert_allclose(reg.forward(x), regularizer(basis.forward(x), kind))
    np.testing.assert_allclose(
        np.vdot(reg.forward(x), y), np.vdot(x, reg.adjoint(y)), atol=2e-14
    )
    eps = 1e-5

    def cost(a):
        return 0.5 * np.linalg.norm(reg.forward(a)) ** 2

    finite_difference = (cost(x + eps * direction) - cost(x - eps * direction)) / (
        2 * eps
    )
    np.testing.assert_allclose(
        finite_difference, np.vdot(direction, normal_regularizer(x, reg)), rtol=1e-8
    )


def test_reduced_eikonal_is_jb_not_straight_surrogate():
    grid, geom, basis = fixture()
    rng = np.random.default_rng(794)
    a = rng.uniform(-1e-5, 1e-5, size=basis.shape)
    v = rng.normal(size=basis.shape) / 1500
    model = EikonalForward(grid, geom)
    s = 1 / 1500 + basis.forward(a)
    lin = model.linearize(s)
    jb = ReducedLinearOperator(lin.jacobian, basis)
    assert not jb.sparse
    y = rng.normal(size=jb.n_rays)
    np.testing.assert_allclose(
        np.vdot(jb.forward(v), y), np.vdot(v, jb.adjoint(y)), rtol=1e-12
    )
    eps = 1e-6
    finite_difference = (
        model.forward(s + eps * basis.forward(v))
        - model.forward(s - eps * basis.forward(v))
    ) / (2 * eps)
    np.testing.assert_allclose(jb.forward(v), finite_difference, rtol=2e-5, atol=1e-12)


@pytest.mark.parametrize(
    "shape", [(0, 2), (2, 999), (2.0, 3.0), (True, False), (2,), None]
)
def test_bad_coefficient_shapes_rejected(shape):
    grid, _, _ = fixture()
    with pytest.raises(ValueError, match="model_grid_shape"):
        BilinearBasis(grid, shape)
