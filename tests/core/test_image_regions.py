import numpy as np
import pytest

from usctbench.metrics import compute_regional_image_metrics, non_water_tissue_mask


def phantom():
    truth = np.full((40, 40), 1500.0)
    truth[8:32, 8:32] = 1450
    truth[15:25, 15:25] = 1530
    truth[18:21, 18:21] = 1500
    return truth


def test_water_artifact_cannot_change_tissue_metrics():
    truth = phantom()
    tissue = non_water_tissue_mask(truth)
    assert tissue[19, 19] and tissue.sum() == 24**2
    clean = truth + np.where(tissue, 2.0, 0.0)
    corrupted = clean + np.where(tissue, 0.0, 100.0)
    a, b = [compute_regional_image_metrics(x, truth) for x in (clean, corrupted)]
    for key in ("rmse", "psnr", "ssim", "tissue_rmse", "tissue_psnr", "tissue_ssim"):
        assert a[key] == b[key]
    assert b["full_image_rmse"] > a["full_image_rmse"]
    assert b["water_background_rmse"] == 100
    assert a["image_evaluation"]["mask_sha256"] == b["image_evaluation"]["mask_sha256"]


def test_range_is_shared_and_constant_tissue_does_not_inflate_psnr():
    truth = np.full((30, 30), 1500.0)
    truth[5:25, 5:25] = 1450
    result = compute_regional_image_metrics(truth + 5, truth)
    assert result["psnr"] == pytest.approx(20)
    assert result["image_evaluation"]["data_range_mps"] == 50
    assert result["psnr"] == result["full_image_psnr"]


def test_missing_or_empty_tissue_has_no_fabricated_primary_score():
    truth = np.full((20, 20), 1500.0)
    for target, status in ((truth, "empty_tissue"), (None, "ground_truth_unavailable")):
        result = compute_regional_image_metrics(truth, target)
        assert result["rmse"] is result["psnr"] is result["ssim"] is None
        assert result["water_reconstruction_rmse"] is None
        assert result["image_evaluation"]["status"] == status
    full = compute_regional_image_metrics(truth, truth, primary_region="full_image")
    assert full["rmse"] == 0


def test_nonfinite_output_in_water_is_not_hidden_by_primary_mask():
    truth = phantom()
    pred = truth.copy()
    pred[0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        compute_regional_image_metrics(pred, truth)


@pytest.mark.parametrize(
    "options",
    [
        {"primary_region": "unknown"},
        {"water_tolerance_mps": -1},
        {"data_range_mps": 0},
        {"water_speed_mps": float("nan")},
    ],
)
def test_invalid_region_policy_is_rejected(options):
    with pytest.raises(ValueError):
        compute_regional_image_metrics(phantom(), phantom(), **options)
