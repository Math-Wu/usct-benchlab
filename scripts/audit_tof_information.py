"""Frozen ToF information diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
from pathlib import Path
import gc
import json
import time
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve, lsmr
from usctbench.core.io import read_case_hdf5, read_result_hdf5
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.model_space import BilinearBasis
from usctbench.metrics import compute_regional_image_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="shared audit directory, outside the source repository",
    )
    parser.add_argument(
        "--handoff", type=Path, required=True, help="extracted usct_handoff_64"
    )
    parser.add_argument("--lsmr-iterations", type=int, default=2000)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    if args.lsmr_iterations <= 0:
        raise ValueError("lsmr iterations must be positive")
    R = args.out.resolve()
    out = R / "information"
    out.mkdir(parents=True, exist_ok=False)
    c = read_case_hdf5(R / "controls/high_d/matched_straight.h5")
    dat = np.load(R / "controls/high_d/predictions.npz")
    gt = c.ground_truth.sound_speed_mps
    p = StraightRayProjector.from_case(c)
    B = BilinearBasis(c.grid, (32, 32))
    A = p.matrix
    train = dat["train"].ravel()
    valid = dat["valid"].ravel()
    val = dat["validation"].ravel()
    weights = dat["weights"].ravel()
    M = A[train] @ B.matrix
    M = M.multiply(np.sqrt(weights[train])[:, None]).toarray()
    u, s, vh = linalg.svd(M, full_matrices=False, check_finite=False)
    report = {
        "purpose": "conditional information and optimization diagnostics, not an attainable global imaging ceiling",
        "unknown_pixels": A.shape[1],
        "valid_directed_data": int(valid.sum()),
        "training_directed_data": int(train.sum()),
        "unique_valid_unordered_pairs": int(np.triu(dat["valid"], 1).sum()),
        "unique_training_unordered_pairs": int(np.triu(dat["train"], 1).sum()),
        "coefficient_grid": [32, 32],
        "singular_values": s.tolist(),
        "coefficient_rank_at_1e_minus_12": int(np.sum(s > s[0] * 1e-12)),
        "coefficient_rank_at_1e_minus_3": int(np.sum(s > s[0] * 0.001)),
    }
    a_truth = spsolve(B.matrix.T @ B.matrix, B.matrix.T @ (1 / gt - 1 / 1500).ravel())
    speed_project = 1 / (1 / 1500 + (B.matrix @ a_truth).reshape(c.grid.shape))
    report["oracle_representation_only"] = compute_regional_image_metrics(
        speed_project, gt
    )
    np.save(out / "oracle_basis_projection.npy", speed_project)
    y, x = np.indices(c.grid.shape)
    y = c.grid.origin_m[0] + (y + 0.5) * c.grid.spacing_m[0]
    x = c.grid.origin_m[1] + (x + 0.5) * c.grid.spacing_m[1]
    support = x * x + y * y < 0.025**2
    z = (
        np.cos(2 * np.pi * x / 0.004)
        * np.cos(2 * np.pi * y / 0.004)
        * np.exp(-(((x * x + y * y) / 0.018**2) ** 2))
    )[support]
    sel = np.triu(dat["valid"], 1).ravel()
    S = A[sel][:, support.ravel()]
    nz = np.asarray(S.power(2).sum(axis=1)).ravel() > 0
    S = S[nz]
    G = (S @ S.T).toarray()
    w, q = linalg.eigh(G, check_finite=False)
    keep = w > w[-1] * 1e-12
    correction = q[:, keep] @ (q[:, keep].T @ (S @ z) / w[keep])
    null = z - S.T @ correction
    null *= 10 / 1500**2 / np.sqrt(np.mean(null**2))
    delta = np.zeros(c.grid.shape)
    delta[support] = null
    speed_null = 1 / (1 / 1500 + delta)
    res = A @ delta.ravel()
    report["nullspace_witness"] = {
        "construction": "geometry-only central 25-mm disk, subtract observed row-space projection; no GT",
        "support_pixels": int(support.sum()),
        "rank_at_1e_minus_12_gram": int(keep.sum()),
        "speed_min_mps": float(speed_null.min()),
        "speed_max_mps": float(speed_null.max()),
        "speed_rms_difference_in_support_mps": float(
            np.sqrt(np.mean((speed_null[support] - 1500) ** 2))
        ),
        "all_valid_delay_rms_ns": float(
            np.sqrt(np.mean(res[valid] ** 2)) * 1000000000.0
        ),
        "all_valid_delay_max_abs_ns": float(np.max(np.abs(res[valid])) * 1000000000.0),
        "interpretation": "nonuniqueness of unrestricted straight slowness; regularity priors may exclude the witness, not a universal quality bound",
    }
    np.save(out / "nullspace_speed.npy", speed_null)
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({k: v for k, v in report.items() if k != "singular_values"}),
        flush=True,
    )
    del G, q, S, u, vh, M
    gc.collect()

    def d2(n):
        d = -2 * np.ones(n)
        d[0] = d[-1] = -1
        return sparse.diags(
            [np.ones(n - 1), d, np.ones(n - 1)], [-1, 0, 1], format="csr"
        )

    L = sparse.kron(d2(256), sparse.eye(256), format="csr") + sparse.kron(
        sparse.eye(256), d2(256), format="csr"
    )
    AA = A[train].multiply(np.sqrt(weights[train])[:, None])
    D = sparse.vstack([10 * AA, 0.2 * L], format="csr")
    report["independent_optimization"] = []
    for label, b in [("matched_straight", dat["straight"]), ("kwave", dat["observed"])]:
        start = time.perf_counter()
        rhs = np.r_[
            b.ravel()[train] * np.sqrt(weights[train]) * 1000000.0, np.zeros(A.shape[1])
        ]
        sol = lsmr(
            D,
            rhs,
            atol=1e-10,
            btol=1e-10,
            conlim=1000000000000.0,
            maxiter=args.lsmr_iterations,
        )
        delta = sol[0].reshape(c.grid.shape) * 1e-05
        s = 1 / 1500 + delta
        speed = 1 / s if np.all(s > 0) else None
        item = {
            "label": label,
            "solver": "scipy.sparse.linalg.lsmr augmented fine-grid system",
            "atol": 1e-10,
            "btol": 1e-10,
            "maxiter": args.lsmr_iterations,
            "stop_code": int(sol[1]),
            "iterations": int(sol[2]),
            "elapsed_s": time.perf_counter() - start,
            "residual_normal_norm": float(sol[4]),
            "unconstrained_cost_s2": float(0.5 * sol[3] ** 2 * 1e-12),
            "sound_speed_bounds_satisfied": bool(
                speed is not None and speed.min() >= 1300 and (speed.max() <= 1700)
            ),
            "gt_used_in_optimization": False,
            "checkpoint_selection": "last LSMR iterate; diagnostic only, not original validation policy",
        }
        if speed is not None:
            item["image_metrics"] = compute_regional_image_metrics(speed, gt)
            np.save(out / (label + "_lsmr.npy"), speed)
            pred = A @ delta.ravel()
            item["train_relative"] = float(
                np.linalg.norm(np.sqrt(weights[train]) * (pred - b.ravel())[train])
                / np.linalg.norm(np.sqrt(weights[train]) * b.ravel()[train])
            )
            item["validation_relative"] = float(
                np.linalg.norm(np.sqrt(weights[val]) * (pred - b.ravel())[val])
                / np.linalg.norm(np.sqrt(weights[val]) * b.ravel()[val])
            )
        ref = (
            R
            / ("baseline" if label == "kwave" else "runs/high_d/matched_straight")
            / "straight_cgls/D510022534/result.h5"
        )
        rec = read_result_hdf5(ref).sound_speed_mps
        ds = (1 / rec - 1 / 1500).ravel()
        r = AA @ ds - b.ravel()[train] * np.sqrt(weights[train])
        item["original_cgls_cost_s2"] = float(
            0.5 * (r @ r + 0.02**2 * np.linalg.norm(L @ ds) ** 2)
        )
        item["cost_ratio_lsmr_to_original"] = (
            item["unconstrained_cost_s2"] / item["original_cgls_cost_s2"]
        )
        report["independent_optimization"].append(item)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()
