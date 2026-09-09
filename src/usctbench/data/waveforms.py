"""Pressure-preserving import for WaveformInversionUST MATLAB v7.3 data.

The upstream MATLAB array is (time, receiver, transmitter); h5py sees the
reversed axes (transmitter, receiver, time). No property-map projector or
arrival-time oracle is called here. Native pressure arrays use (time, tx, rx),
and spectra use (frequency, tx, rx).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from usctbench.core.io import write_case_hdf5
from usctbench.core.provenance import MeasurementProvenance, stamp_measurement_metadata
from usctbench.core.schema import (
    GeometrySpec,
    GridSpec,
    GroundTruthSpec,
    MeasurementSpec,
    USCTCase,
)


def pressure_spectrum(pressure, time_s, frequencies_hz):
    """DTFT quadrature for phasors exp(-i omega t), not nearest FFT bins.

    P(f) = dt sum_n p(t_n) exp(+i 2 pi f t_n). The absolute time origin
    matters. WaveformInversionUST's exp(-i*omega*time) DTFT has the opposite
    sign; for real traces its spectra are the complex conjugate of these.
    Missing samples are not silently replaced or interpolated.
    """
    p = np.asarray(pressure)
    t = np.asarray(time_s, dtype=float)
    f = np.asarray(frequencies_hz, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)):
        raise ValueError("time_s must be a finite vector with at least two samples")
    dt = float(np.mean(np.diff(t)))
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-15):
        raise ValueError("time_s must be uniformly sampled and strictly increasing")
    if p.ndim < 1 or p.shape[0] != t.size or np.iscomplexobj(p):
        raise ValueError("pressure must be real with time as its first axis")
    if (
        f.ndim != 1
        or not f.size
        or not np.all(np.isfinite(f))
        or np.any(f <= 0)
        or np.any(f > 0.5 / dt)
        or np.unique(f).size != f.size
    ):
        raise ValueError(
            "frequencies must be distinct, positive and at or below Nyquist"
        )
    # Frequency-by-frequency evaluation bounds temporary memory independently
    # of the number of requested frequencies. NaNs propagate to missing traces.
    return np.stack(
        [
            np.tensordot(dt * np.exp(2j * np.pi * frequency * t), p, axes=(0, 0))
            for frequency in f
        ]
    )


def _selection(values, size, name):
    if values is None:
        return np.arange(size)
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty integer vector")
    if np.any(raw < 0) or np.any(raw >= size) or np.unique(raw).size != raw.size:
        raise ValueError(f"{name} must contain unique in-range zero-based indices")
    return raw.astype(int)


def _coordinate_grid(handle, output_shape):
    axes = []
    for key in ("yi_orig", "xi_orig"):
        coordinates = np.asarray(handle[key], dtype=float).ravel()
        if coordinates.size < 2 or not np.all(np.isfinite(coordinates)):
            raise ValueError(f"{key} must contain finite cell-center coordinates")
        spacing = float(np.mean(np.diff(coordinates)))
        if spacing <= 0 or not np.allclose(np.diff(coordinates), spacing):
            raise ValueError(f"{key} must be uniformly increasing")
        axes.append(
            (coordinates[0] - spacing / 2, spacing * coordinates.size, coordinates.size)
        )
    shape = (
        tuple(int(axis[2]) for axis in axes)
        if output_shape is None
        else tuple(output_shape)
    )
    return GridSpec(
        shape=shape,
        origin_m=tuple(axis[0] for axis in axes),
        spacing_m=tuple(axis[1] / n for axis, n in zip(axes, shape, strict=True)),
    )


def _read_channels(dataset, axes, tx, rx, n_time, max_output_bytes):
    if tuple(sorted(axes)) != ("rx", "time", "tx") or dataset.ndim != 3:
        raise ValueError("channel_axes must be a permutation of ('tx', 'rx', 'time')")
    if dataset.dtype.kind not in "fiu" or dataset.shape[axes.index("time")] != n_time:
        raise ValueError(
            "channel dataset must contain real pressure with the given time length"
        )
    shape = (n_time, len(tx), len(rx))
    if int(np.prod(shape)) * dataset.dtype.itemsize > max_output_bytes:
        raise ValueError(
            "selected pressure tensor exceeds max_output_bytes; select fewer channels"
        )
    result = np.empty(shape, dtype=dataset.dtype)
    # MATLAB v7.3 commonly compresses one time plane across ALL channels.
    # Trace-by-trace access decompresses that plane n_tx*n_rx times. Read bounded
    # time slabs over the requested channel bounding box, then restore order.
    # A very sparse selection with an oversized bounding box keeps the trace
    # fallback rather than allocating the complete acquisition.
    tx0, rx0 = int(np.min(tx)), int(np.min(rx))
    tx_span, rx_span = int(np.max(tx)) - tx0 + 1, int(np.max(rx)) - rx0 + 1
    plane_bytes = tx_span * rx_span * dataset.dtype.itemsize
    temporary_limit = min(max_output_bytes, 16 * 1024**2)
    if plane_bytes <= temporary_limit:
        block_size = max(1, temporary_limit // plane_bytes)
        permutation = tuple(axes.index(a) for a in ("time", "tx", "rx"))
        for start in range(0, n_time, block_size):
            stop = min(start + block_size, n_time)
            slices = {
                "time": slice(start, stop),
                "tx": slice(tx0, tx0 + tx_span),
                "rx": slice(rx0, rx0 + rx_span),
            }
            block = dataset[tuple(slices[a] for a in axes)].transpose(permutation)
            result[start:stop] = block[:, np.asarray(tx) - tx0][
                :, :, np.asarray(rx) - rx0
            ]
        return result
    for i, transmitter in enumerate(tx):
        for j, receiver in enumerate(rx):
            selector = [slice(None)] * 3
            selector[axes.index("tx")] = int(transmitter)
            selector[axes.index("rx")] = int(receiver)
            result[:, i, j] = dataset[tuple(selector)]
    return result


def convert_kwave_pressure_mat(
    mat_path: str | Path,
    out_path: str | Path,
    *,
    frequencies_hz=None,
    tx_indices=None,
    rx_indices=None,
    channel_axes=("tx", "rx", "time"),
    output_shape=None,
    grid: GridSpec | None = None,
    reference_sound_speed_mps: float = 1500.0,
    water_dataset: str | None = None,
    provenance: str = MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value,
    max_output_bytes: int = 1024**3,
) -> USCTCase:
    """Import actual pressure without generating features from ground truth.

    `C` and `atten` are optional evaluation labels, never input to measurement
    generation, reference-speed selection, geometry or ROI selection. HDF5
    `C` is already [y,x] for upstream MATLAB [x,y] images. An explicit grid is
    required when xi_orig/yi_orig are unavailable. `water_dataset`, when set,
    names a separately acquired reference tensor in the same file and axes.
    The byte limit applies per selected tensor (not total process memory).
    """
    c0 = float(reference_sound_speed_mps)
    if not np.isfinite(c0) or c0 <= 0:
        raise ValueError("reference_sound_speed_mps must be finite and positive")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    source = Path(mat_path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        positions = np.asarray(handle["transducerPositionsXY"], dtype=float)
        if (
            positions.ndim != 2
            or positions.shape[1] != 2
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("HDF5 transducerPositionsXY must be finite (element, xy)")
        tx = _selection(tx_indices, len(positions), "tx_indices")
        rx = _selection(rx_indices, len(positions), "rx_indices")
        time = np.asarray(handle["time"], dtype=float).ravel()
        axes = tuple(channel_axes)
        dataset = handle["full_dataset"]
        if tuple(sorted(axes)) != ("rx", "time", "tx"):
            raise ValueError("invalid channel_axes")
        if dataset.shape[axes.index("tx")] != len(positions) or dataset.shape[
            axes.index("rx")
        ] != len(positions):
            raise ValueError("channel axes do not match transducer count")
        pressure = _read_channels(dataset, axes, tx, rx, len(time), max_output_bytes)
        reference = (
            None
            if water_dataset is None
            else _read_channels(
                handle[water_dataset], axes, tx, rx, len(time), max_output_bytes
            )
        )
        if grid is not None and output_shape is not None:
            raise ValueError("supply grid or output_shape, not both")
        image_grid = (
            grid if grid is not None else _coordinate_grid(handle, output_shape)
        )
        truth = {}
        for key, field in (("C", "sound_speed_mps"), ("atten", "attenuation_np_per_m")):
            if key in handle:
                # Attenuation units in different upstream exports vary. Do not
                # label an unconverted attenuation array as Np/m.
                if key == "atten":
                    continue
                image = np.asarray(handle[key], dtype=float)
                if image.shape != image_grid.shape:
                    from skimage.transform import resize

                    image = resize(
                        image, image_grid.shape, preserve_range=True, anti_aliasing=True
                    )
                truth[field] = image
    # Always validate sampling, including imports that keep only time data.
    validation_frequency = (
        np.array([0.25 / np.mean(np.diff(time))])
        if frequencies_hz is None
        else frequencies_hz
    )
    spectrum = pressure_spectrum(pressure, time, validation_frequency)
    keep_frequency = frequencies_hz is not None
    reference_time = reference
    if reference is not None and keep_frequency:
        reference = pressure_spectrum(reference, time, frequencies_hz)
    geometry = GeometrySpec(tx_pos_m=positions[tx, ::-1], rx_pos_m=positions[rx, ::-1])
    valid = np.all(np.isfinite(pressure), axis=0)
    valid &= (
        np.linalg.norm(geometry.tx_pos_m[:, None] - geometry.rx_pos_m[None], axis=-1)
        > 0
    )
    metadata = stamp_measurement_metadata(
        {
            "source_dataset": "WaveformInversionUST_channel_mat",
            "source_path": str(source),
            "conversion": "pressure_preserving_kwave_v73",
            "frequency_convention": "exp(-i omega t)",
            "spectrum_quadrature": "dt * sum(p(t) * exp(+i*2*pi*f*t))",
            "pressure_axes": ["time", "tx", "rx"],
            "frequency_axes": ["frequency", "tx", "rx"],
            "source_channel_axes": list(axes),
            "selected_tx_indices": tx.tolist(),
            "selected_rx_indices": rx.tolist(),
            "reference_sound_speed_mps": c0,
            "ground_truth_used_for_measurement": False,
            "simulation_backend": "kwave",
            "water_reference_used": reference is not None,
            "measurement_limitations": [
                "Source strength, detector response and pressure units require acquisition-specific calibration.",
                "The pressure import itself applies no window, picker, gain normalization or 3D-to-2D correction; derived feature processing is recorded separately.",
                "Raw attenuation labels are not imported without an explicit unit conversion.",
            ],
        },
        measurement_provenance=provenance,
        benchmark_type="pressure_waveform",
        forward_model="external_pressure_acquisition",
        feature_source="full_dataset pressure samples (not property-map projections)",
        uses_complex_wavefield=keep_frequency,
    )
    case = USCTCase(
        case_id=source.stem,
        grid=image_grid,
        geometry=geometry,
        measurement=MeasurementSpec(
            domain="frequency" if keep_frequency else "time",
            time_data=pressure,
            time_axis_s=time,
            freq_data=spectrum if keep_frequency else None,
            frequencies_hz=np.asarray(frequencies_hz) if keep_frequency else None,
            water_reference=reference,
            water_reference_time=reference_time,
            valid_mask=valid,
        ),
        ground_truth=GroundTruthSpec(**truth),
        metadata=metadata,
    )
    write_case_hdf5(case, out_path)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mat_path")
    parser.add_argument("out_path")
    parser.add_argument("--frequencies-hz", type=float, nargs="+")
    parser.add_argument("--tx-indices", type=int, nargs="+")
    parser.add_argument("--rx-indices", type=int, nargs="+")
    parser.add_argument("--reference-sound-speed-mps", type=float, default=1500.0)
    parser.add_argument("--water-dataset")
    args = vars(parser.parse_args())
    case = convert_kwave_pressure_mat(**args)
    print(f"{case.case_id}: preserved pressure {case.measurement.time_data.shape}")


if __name__ == "__main__":
    main()
