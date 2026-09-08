"""Frozen ToF recorded diagnosis. Outputs are development controls, not clinical evidence."""

import argparse
from pathlib import Path
import gc
import json
import time
import numpy as np
from scipy.ndimage import map_coordinates
from usctbench.core.io import read_case_hdf5
from usctbench.core.schema import GridSpec, GeometrySpec
from usctbench.operators.forward.eikonal import EikonalForward, fast_march


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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.out.resolve() == repo or repo in args.out.resolve().parents:
        raise ValueError("private outputs must stay outside the repository")

    R = args.out.resolve()
    c = read_case_hdf5(R / "controls/high_d/kwave.h5")
    ids = np.arange(0, 64, 8)
    geom = GeometrySpec(tx_pos_m=c.geometry.tx_pos_m[ids], rx_pos_m=c.geometry.rx_pos_m)
    out = R / "recorded_field"
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "policy": "re-evaluate documented prepare.py order-1 speed interpolant at recorded simulation grid then twice finer, no new k-Wave simulation",
        "raw_simulation_input_not_provided": True,
        "tx_indices": ids.tolist(),
        "grids": [],
    }
    fields = {}
    for factor in [1, 2]:
        n = 1024 * factor
        dx = c.metadata["spacing_m"] / factor
        grid = GridSpec(
            shape=(n, n), spacing_m=(dx, dx), origin_m=(-n // 2 * dx - dx / 2,) * 2
        )
        ij = np.indices((n, n), dtype=float)
        ij = (ij - n // 2) * dx
        coords = [
            (ij[a] - c.grid.origin_m[a]) / c.grid.spacing_m[a] - 0.5 for a in range(2)
        ]
        speed = map_coordinates(
            c.ground_truth.sound_speed_mps,
            coords,
            order=1,
            mode="constant",
            cval=1500.0,
        )
        del ij, coords
        op = EikonalForward(grid, geom)
        ext = op.extend(1 / speed, background=1 / 1500)
        water = np.full(op.shape, 1 / 1500)
        value = []
        start = time.perf_counter()
        for source in op.source_coordinates:
            tape = fast_march(ext, grid.spacing_m, source)
            a = op.sample_receivers(tape.times)
            del tape
            tape = fast_march(water, grid.spacing_m, source)
            b = op.sample_receivers(tape.times)
            del tape
            value.append(a - b)
        value = np.asarray(value)
        fields[str(n)] = value
        mask = c.measurement.valid_mask[ids]
        r = (value - c.measurement.delta_tof_s[ids])[mask] * 1000000.0
        row = {
            "grid": n,
            "spacing_m": dx,
            "rms_vs_kwave_us": float(np.sqrt(np.mean(r * r))),
            "bias_vs_kwave_us": float(np.mean(r)),
            "elapsed_s": time.perf_counter() - start,
        }
        if factor == 2:
            d = (value - fields["1024"])[mask] * 1000000.0
            row["successive_change_rms_us"] = float(np.sqrt(np.mean(d * d)))
        report["grids"].append(row)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(out / "predictions.npz", **fields)
        print(json.dumps(row), flush=True)
        del op, speed, ext, water
        gc.collect()


if __name__ == "__main__":
    main()
