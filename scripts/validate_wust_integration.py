#!/usr/bin/env python3
"""Opt-in real BenchLab -> WUST gate using a coherent WUST fixture manifest.

The fixture is a validation input, not a production GT-to-observations path.
Run with --help. Artifacts are written outside source control.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from usctbench.algorithms.fwi.algorithm import WUSTFWIAlgorithm
from usctbench.algorithms.fwi.io import array_hash, read_artifact
from usctbench.core.io import write_case_hdf5, write_result_hdf5
from usctbench.core.schema import (
    AlgorithmConfig,
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        type=Path,
        help="Coherent canonical WUST measurements from independent fixture generation",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--matlab", default="matlab")
    parser.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--timeout-s", type=float, default=600)
    parser.add_argument("--pml-m", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    doc, a = read_artifact(args.fixture, "wust.measurements")
    grid = doc["metadata"]["grid"]
    shape = tuple(grid["shape_yx"])
    freq = a["frequencies_hz"][::-1]
    case = USCTCase(
        case_id="wust_cross_repository",
        grid=GridSpec(
            shape=shape, spacing_m=grid["spacing_yx_m"], origin_m=grid["origin_yx_m"]
        ),
        geometry=GeometrySpec(
            type="custom",
            tx_pos_m=a["tx_xy_m"][:, ::-1],
            rx_pos_m=a["rx_xy_m"][:, ::-1],
        ),
        measurement=MeasurementSpec(
            domain="frequency",
            frequencies_hz=freq,
            freq_data=a["Y"].transpose(2, 0, 1)[::-1].conj(),
            valid_mask=a["mask"].transpose(2, 0, 1)[::-1],
        ),
        metadata={
            "reference_sound_speed_mps": 1500.0,
            "measurement_provenance": "independent_coherent_Helmholtz_fixture",
            "pressure_contract": {
                "fourier_sign": 1,
                "real_pressure": True,
                "pressure_type": "total_pressure",
                "data_units": "unknown",
                "spectrum_normalization": "unknown",
            },
        },
        ground_truth=GroundTruthSpec(sound_speed_mps=np.full(shape, 1501.0)),
    )
    write_case_hdf5(case, args.out / "case.h5")
    schedule = [float(min(freq)), float(min(freq)), float(max(freq))]
    config = AlgorithmConfig(
        name="fwi_wust",
        run_controls={"max_elapsed_s": args.timeout_s},
        parameters={
            "runtime_root": args.runtime,
            "matlab_executable": args.matlab,
            "backend": args.backend,
            "sound_speed_bounds_mps": [1300.0, 1800.0],
            "stencil_bounds_mps": [1400.0, 1700.0],
            "pml_m": args.pml_m,
            "frequency_schedule_hz": schedule,
            "source_batch_size": args.batch_size,
            "scratch_root": str(args.out.resolve()),
        },
    )
    (args.out / "config.json").write_text(config.model_dump_json(indent=2))
    results = []
    started = time.monotonic()
    for index, gt in enumerate((1501.0, 1550.0)):
        case.ground_truth.sound_speed_mps[:] = gt
        result = WUSTFWIAlgorithm().run(case, config)
        (args.out / f"result-{index}.json").write_text(
            json.dumps(
                {
                    "status": str(result.status),
                    "failure_reason": result.failure_reason,
                    "metrics": result.metrics,
                    "artifacts": result.artifacts,
                },
                indent=2,
            )
        )
        if result.status != "success":
            raise RuntimeError(result.failure_reason)
        write_result_hdf5(result, args.out / f"result-{index}.h5")
        results.append(result)
    first, second = results
    np.testing.assert_array_equal(first.sound_speed_mps, second.sound_speed_mps)
    for stem, schema in (
        ("frequency", "wust.frequency_input"),
        ("initial", "wust.initial_model"),
    ):
        _, left = read_artifact(
            Path(first.artifacts["wust_directory"]) / (stem + ".json"), schema
        )
        _, right = read_artifact(
            Path(second.artifacts["wust_directory"]) / (stem + ".json"), schema
        )
        assert {k: array_hash(v) for k, v in left.items()} == {
            k: array_hash(v) for k, v in right.items()
        }
    _, mapped = read_artifact(
        Path(first.artifacts["wust_directory"]) / "measurements.json",
        "wust.measurements",
    )
    np.testing.assert_allclose(mapped["Y"], a["Y"], rtol=0, atol=0)
    np.testing.assert_array_equal(mapped["mask"], a["mask"])
    np.testing.assert_array_equal(mapped["tx_xy_m"], a["tx_xy_m"])
    np.testing.assert_array_equal(mapped["rx_xy_m"], a["rx_xy_m"])
    assert (
        first.sound_speed_mps.shape == shape
        and np.isfinite(first.sound_speed_mps).all()
    )
    assert (
        first.metrics["stop_reason"] == "schedule_complete"
        and not first.metrics["converged"]
    )
    report = {
        "passed": True,
        "backend": args.backend,
        "role": "production" if args.backend == "gpu" else "reference",
        "shape": shape,
        "tx": len(a["tx_xy_m"]),
        "rx": len(a["rx_xy_m"]),
        "schedule_hz": schedule,
        "batch_size": args.batch_size,
        "gt_independence_exact": True,
        "axis_sign_mask_exact": True,
        "wall_seconds": time.monotonic() - started,
        "environment": first.metrics["wust"]["environment"],
        "provenance": first.metrics["wust"]["runtime_provenance"],
    }
    (args.out / "gate.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
