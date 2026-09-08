from dataclasses import replace

import numpy as np
import pytest

from usctbench.core.schema import GeometrySpec, GridSpec
from usctbench.operators.forward.straight_ray import StraightRayProjector


def projector():
    grid = GridSpec(shape=(11, 13), spacing_m=(0.003, 0.002), origin_m=(-0.016, -0.013))
    theta = np.arange(9) * 2 * np.pi / 9
    positions = 0.028 * np.column_stack([np.sin(theta), np.cos(theta)])
    return StraightRayProjector.from_grid_geometry(
        grid, GeometrySpec(tx_pos_m=positions, rx_pos_m=positions[::-1])
    )


def test_csr_forward_adjoint_equal_reference_and_dot_product():
    op = projector()
    slow = replace(op, backend="reference")
    rng = np.random.default_rng(723)
    x, y = rng.normal(size=op.grid.shape), rng.normal(size=op.n_rays)
    np.testing.assert_allclose(op.forward(x), slow.forward(x), atol=1e-16, rtol=2e-14)
    np.testing.assert_allclose(op.adjoint(y), slow.adjoint(y), atol=1e-16, rtol=2e-14)
    np.testing.assert_allclose(
        np.vdot(op.forward(x), y), np.vdot(x, op.adjoint(y)), rtol=2e-14
    )
    assert op.matrix is op.matrix
    assert op.storage_bytes < op.n_rays * op.n_pixels * 8


@pytest.mark.parametrize("power", [1, 2])
def test_norms_match_explicit_dense_coefficients(power):
    op = projector()
    dense = op.matrix.toarray()
    np.testing.assert_allclose(op.row_norms(power), (dense**power).sum(axis=1))
    np.testing.assert_allclose(
        op.col_norms(power), (dense**power).sum(axis=0).reshape(op.grid.shape)
    )


def test_duplicate_segments_norm_after_summing_and_empty_rows():
    op = projector()
    op = replace(
        op,
        indices_by_ray=(np.array([0, 0, 1]), np.array([], dtype=int)),
        lengths_by_ray_m=(np.array([0.2, 0.3, 0.4]), np.array([])),
    )
    np.testing.assert_allclose(op.row_norms(2), [0.25 + 0.16, 0])
    np.testing.assert_allclose(op.forward(np.ones(op.grid.shape)), [0.9, 0])
    assert np.all(op.adjoint([0, 10]) == 0)


def test_invalid_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        replace(projector(), backend="not-a-backend")
