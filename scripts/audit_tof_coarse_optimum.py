"""Independent fixed-basis quadratic optimum, not an oracle image-quality ceiling."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve
import yaml

from usctbench.benchmark.runner import run_algorithm_case
from usctbench.cli import register_builtin_algorithms
from usctbench.core.io import read_case_hdf5, read_result_hdf5
from usctbench.metrics import compute_regional_image_metrics
from usctbench.operators.straight_ray import StraightRayProjector
from usctbench.operators.model_space import BilinearBasis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--handoff", type=Path, required=True)
    args = p.parse_args()
    root = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    if root == repo or repo in root.parents:
        raise ValueError("private outputs must remain outside Git")
    out = root / "coarse_optimum"
    out.mkdir(parents=True, exist_ok=False)
    case = read_case_hdf5(root / "controls/high_d/kwave.h5")
    c0 = 1500.0
    B = BilinearBasis(case.grid, (32, 32)).matrix
    A = StraightRayProjector.from_case(case).matrix
    dat = np.load(root / "controls/high_d/predictions.npz")
    train, val = dat["train"].ravel(), dat["validation"].ravel()
    w = dat["weights"].ravel()

    def d2(n):
        d = -2 * np.ones(n)
        d[[0, -1]] = -1
        return sparse.diags(
            [np.ones(n - 1), d, np.ones(n - 1)], [-1, 0, 1], format="csr"
        )

    ny, nx = case.grid.shape
    L = sparse.kron(d2(ny), sparse.eye(nx)) + sparse.kron(sparse.eye(ny), d2(nx))
    cfg = yaml.safe_load((args.handoff / "configs/straight_cgls.yaml").read_text())
    cfg["parameters"]["model_grid_shape"] = [32, 32]
    assert cfg["parameters"]["regularization"] == "laplacian"
    lam = cfg["parameters"]["regularization_lambda"]
    lower, upper = cfg["parameters"]["sound_speed_bounds_mps"]
    M = (A[train] @ B).multiply(np.sqrt(w[train])[:, None]) * 10
    R = (L @ B) * lam * 10
    H = (M.T @ M + R.T @ R).toarray()
    H = 0.5 * (H + H.T)
    eig = linalg.eigvalsh(H)
    factor = linalg.cho_factor(H)
    report = {
        "scope": "same 32-square model space, fine-grid regularizer, training data; no GT in solve",
        "hessian_eigenvalue_min": float(eig[0]),
        "hessian_condition": float(eig[-1] / eig[0]),
        "rows": [],
    }
    register_builtin_algorithms()
    for label, key in [("matched_straight", "straight"), ("kwave", "observed")]:
        target = dat[key].ravel()
        rhs = target[train] * np.sqrt(w[train]) * 1e6
        start = time.perf_counter()
        q = linalg.cho_solve(factor, M.T @ rhs)
        solve_s = time.perf_counter() - start
        speed = 1 / (1 / c0 + (B @ q).reshape(case.grid.shape) * 1e-5)
        normal = H @ q - M.T @ rhs
        feasible = bool(
            np.all(q * 1e-5 >= 1 / upper - 1 / c0)
            and np.all(q * 1e-5 <= 1 / lower - 1 / c0)
        )

        def cost(x):
            return (
                0.5
                * (np.linalg.norm(M @ x - rhs) ** 2 + np.linalg.norm(R @ x) ** 2)
                * 1e-12
            )

        cfg_path = out / (label + ".yaml")
        cfg_path.write_text(yaml.safe_dump(cfg))
        dest = run_algorithm_case(
            "straight_cgls",
            root / "controls/high_d" / (label + ".h5"),
            cfg_path,
            out / "cgls" / label,
        )
        meta = yaml.safe_load((dest / "metadata.yaml").read_text())
        if meta["status"] != "success":
            raise RuntimeError(meta["failure_reason"])
        cg = read_result_hdf5(dest / "result.h5").sound_speed_mps
        qc = spsolve(B.T @ B, B.T @ (1 / cg - 1 / c0).ravel()) / 1e-5
        np.testing.assert_allclose(
            (B @ qc) * 1e-5, (1 / cg - 1 / c0).ravel(), atol=1e-18
        )
        np.save(out / (label + "_direct.npy"), speed)
        pred = A @ (B @ q * 1e-5)
        row = {
            "label": label,
            "direct_solve_s_excludes_setup": solve_s,
            "direct_cost_s2": cost(q),
            "cgls_cost_s2": cost(qc),
            "cost_ratio_direct_to_cgls": cost(q) / cost(qc),
            "relative_normal_residual": float(
                np.linalg.norm(normal) / np.linalg.norm(M.T @ rhs)
            ),
            "coefficient_bounds_satisfied": feasible,
            "speed_min_mps": float(speed.min()),
            "speed_max_mps": float(speed.max()),
            "quadratic_stationarity_error_bound_s2": float(
                0.5 * np.linalg.norm(normal) ** 2 / eig[0] * 1e-12
            ),
            "direct_image_metrics": compute_regional_image_metrics(
                speed, case.ground_truth.sound_speed_mps
            ),
            "cgls_metrics": json.loads((dest / "metrics.json").read_text()),
            "direct_train_relative": float(
                np.linalg.norm(np.sqrt(w[train]) * (pred - target)[train])
                / np.linalg.norm(np.sqrt(w[train]) * target[train])
            ),
            "direct_validation_relative": float(
                np.linalg.norm(np.sqrt(w[val]) * (pred - target)[val])
                / np.linalg.norm(np.sqrt(w[val]) * target[val])
            ),
            "selection": "direct stationary point; no early stopping/validation selection, diagnostic NOT fair-budget reconstruction ranking",
        }
        report["rows"].append(row)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        print(
            label,
            row["cost_ratio_direct_to_cgls"],
            row["direct_image_metrics"]["rmse"],
            row["direct_image_metrics"]["ssim"],
            feasible,
            flush=True,
        )


if __name__ == "__main__":
    main()
