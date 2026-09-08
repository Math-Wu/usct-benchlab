import numpy as np
import pytest
from usctbench.evaluation.tof_diagnostics import (
    error_decomposition,
    reciprocity_floor,
    shuffled_pair_error,
)


def test_cross_term_not_difference_of_rms():
    a = np.zeros(4)
    e = np.array([1.0, -1.0, 1.0, -1.0])
    b = e / 2
    d = error_decomposition(a, e, b, mask=np.ones(4, bool), weights=np.ones(4))
    assert d["total_straight_gap_rms_s"] == 0.5
    assert d["component_cosine"] == -1
    assert d["identity_absolute_error_s2"] == 0


def test_reciprocity_floor_equals_pair_optimum():
    b = np.array([[0.0, 1.0], [3.0, 0.0]])
    w = np.array([[0.0, 1.0], [3.0, 0.0]])
    m = w > 0
    d = reciprocity_floor(b, mask=m, weights=w)
    optimum = 2.5
    assert d["minimum_weighted_sse_s2"] == np.sum(w * (b - optimum) ** 2)
    b[1, 0] = np.nan
    m[1, 0] = False
    assert reciprocity_floor(b, mask=m, weights=w)["minimum_weighted_sse_s2"] == 0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_shuffle_partition_norm_mean_reciprocity_and_no_leakage(seed):
    rng = np.random.default_rng(72)
    n = 10
    err = rng.normal(size=(n, n))
    mask = ~np.eye(n, dtype=bool)
    weights = rng.uniform(0.1, 2, (n, n))
    groups = np.zeros((n, n), dtype=int)
    groups[:, :2] = 1
    groups[:2, 2:] = 2
    out, info = shuffled_pair_error(
        err, mask=mask, weights=weights, groups=groups, seed=seed
    )
    for k in [0, 1, 2]:
        sel = mask & (groups == k)
        np.testing.assert_allclose(
            np.sum(weights[sel] * out[sel] ** 2),
            np.sum(weights[sel] * err[sel] ** 2),
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            np.sum(weights[sel] * out[sel]), np.sum(weights[sel] * err[sel]), atol=1e-13
        )
    np.testing.assert_allclose(out - out.T, err - err.T, atol=2e-15)
    np.testing.assert_array_equal(out[groups != groups.T], err[groups != groups.T])
    altered = err.copy()
    altered[groups == 1] += 1000
    other, _ = shuffled_pair_error(
        altered, mask=mask, weights=weights, groups=groups, seed=seed
    )
    np.testing.assert_array_equal(out[groups == 0], other[groups == 0])
    assert info["shuffled_pairs"] > 0
    assert not np.allclose(out, err)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_reject_nonfinite_active_data(bad):
    with pytest.raises(ValueError, match="finite real"):
        reciprocity_floor(
            [[0.0, bad], [1.0, 0.0]],
            mask=np.ones((2, 2), bool),
            weights=np.ones((2, 2)),
        )


def test_invalid_masks_weights_and_shapes():
    with pytest.raises(ValueError):
        reciprocity_floor(
            np.ones((2, 3)), mask=np.ones((2, 3), bool), weights=np.ones((2, 3))
        )
    with pytest.raises(ValueError):
        reciprocity_floor(
            np.ones((2, 2)), mask=np.zeros((2, 2), bool), weights=np.ones((2, 2))
        )
    with pytest.raises(ValueError):
        reciprocity_floor(
            np.ones((2, 2)), mask=np.ones((2, 2), bool), weights=-np.ones((2, 2))
        )


def test_training_only_mode_preserves_large_validation_partition():
    rng = np.random.default_rng(17)
    n = 16
    e = rng.normal(size=(n, n))
    groups = np.zeros((n, n), dtype=int)
    groups[:, :5] = 1
    groups[:5, 5:] = 2
    mask = ~np.eye(n, dtype=bool)
    out, _ = shuffled_pair_error(
        e,
        mask=mask,
        weights=np.ones((n, n)),
        groups=groups,
        seed=0,
        shuffle_group_ids=[0],
    )
    np.testing.assert_array_equal(out[groups != 0], e[groups != 0])
