"""Generic JSON/HDF5 interchange only; physical canonicalization belongs to WUST."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_artifact(path, schema, metadata, arrays, axes, units):
    path = Path(path)
    data = path.with_suffix(".h5")
    if path.exists() or data.exists():
        raise FileExistsError(path)
    descriptors = {}
    with h5py.File(data, "x") as handle:
        for key, array in arrays.items():
            value = np.asarray(array)
            spec = {
                "axes": axes[key],
                "units": units[key],
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if np.iscomplexobj(value):
                spec.update(real_dataset=f"/{key}_real", imag_dataset=f"/{key}_imag")
                handle.create_dataset(spec["real_dataset"], data=value.real)
                handle.create_dataset(spec["imag_dataset"], data=value.imag)
            else:
                spec["dataset"] = f"/{key}"
                handle.create_dataset(
                    spec["dataset"],
                    data=value.astype(np.uint8) if value.dtype == bool else value,
                )
            descriptors[key] = spec
    manifest = {
        "schema": schema,
        "schema_version": 1,
        "metadata": metadata,
        "arrays_file": data.name,
        "arrays_sha256": file_hash(data),
        "arrays": descriptors,
    }
    with path.open("x") as stream:
        json.dump(manifest, stream, allow_nan=False, indent=2)
    return manifest


def read_artifact(path, schema):
    path = Path(path)
    doc = json.loads(path.read_text())
    if (
        doc.get("schema") != schema
        or type(doc.get("schema_version")) is not int
        or doc["schema_version"] != 1
    ):
        raise ValueError("Incompatible WUST result schema")
    filename = doc["arrays_file"]
    if Path(filename).name != filename:
        raise ValueError("WUST array artifact must be a sibling file")
    data = (path.parent / filename).resolve(strict=True)
    if data.parent != path.parent.resolve() or file_hash(data) != doc["arrays_sha256"]:
        raise ValueError("WUST artifact hash/location mismatch")
    arrays = {}
    with h5py.File(data, "r") as handle:
        for key, spec in doc["arrays"].items():

            def read(dataset):
                if not isinstance(handle.get(dataset, getlink=True), h5py.HardLink):
                    raise ValueError("External HDF5 links forbidden")
                value = handle[dataset]
                if (
                    value.is_virtual
                    or value.external
                    or Path(value.file.filename).resolve() != data
                ):
                    raise ValueError("External HDF5 storage forbidden")
                return value[...]

            if spec["dtype"].startswith("complex"):
                value = read(spec["real_dataset"]) + 1j * read(spec["imag_dataset"])
            else:
                value = read(spec["dataset"])
                if spec["dtype"] == "bool":
                    if not np.isin(value, [0, 1]).all():
                        raise ValueError("Nonbinary WUST mask")
                    value = value.astype(bool)
            if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
                raise ValueError("WUST array dtype/shape mismatch")
            arrays[key] = value
    return doc, arrays


def frequency_input(case, path):
    measurement = case.measurement
    pressure = measurement.freq_data
    frequencies = measurement.frequencies_hz
    mask = measurement.valid_mask
    if (
        pressure is None
        or not np.iscomplexobj(pressure)
        or frequencies is None
        or mask is None
    ):
        raise ValueError(
            "FWI requires total complex freq_data, frequencies_hz and explicit valid_mask"
        )
    shape = (len(frequencies), len(case.geometry.tx_pos_m), len(case.geometry.rx_pos_m))
    if pressure.shape != shape or mask.shape not in (shape, shape[1:]):
        raise ValueError("Expected pressure [F,TX,RX] and matching explicit mask")
    if (
        not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0)
        or len(np.unique(frequencies)) != len(frequencies)
    ):
        raise ValueError("Frequencies must be finite, positive and unique")
    valid = np.broadcast_to(mask, shape)
    if not np.isfinite(pressure[valid]).all():
        raise ValueError("Nonfinite valid pressure")
    declared = dict(case.metadata.get("pressure_contract", {}))
    if (
        not declared
        and case.metadata.get("conversion") == "pressure_preserving_kwave_v73"
        and case.metadata.get("spectrum_quadrature")
        == "dt * sum(p(t) * exp(+i*2*pi*f*t))"
    ):
        declared = {
            "fourier_sign": 1,
            "real_pressure": True,
            "pressure_type": "total_pressure",
            "data_units": "unknown",
            "spectrum_normalization": "dtft_dt",
        }
    required = {
        "fourier_sign",
        "real_pressure",
        "pressure_type",
        "data_units",
        "spectrum_normalization",
    }
    if set(declared) != required or declared["pressure_type"] != "total_pressure":
        raise ValueError(
            "Explicit total-pressure contract required; ToF/scattered/ratio data are not FWI pressure"
        )
    if type(declared["fourier_sign"]) is not int or declared["fourier_sign"] not in (
        -1,
        1,
    ):
        raise ValueError(
            "Unknown DTFT sign; time-harmonic convention alone is insufficient"
        )
    if declared["fourier_sign"] == 1 and declared["real_pressure"] is not True:
        raise ValueError("Positive-sign conversion requires declared real pressure")
    metadata = {
        **declared,
        "grid": {
            "shape_yx": list(case.grid.shape),
            "spacing_yx_m": list(case.grid.spacing_m),
            "origin_yx_m": list(case.grid.origin_m),
            "origin_kind": "pixel_edge",
        },
        "measurement_provenance": case.metadata.get(
            "measurement_provenance", "unknown"
        ),
    }
    # The sole physical coordinate-column conversion. No pressure conjugation,
    # frequency sorting or MATLAB-index generation occurs in BenchLab.
    arrays = {
        "pressure": pressure,
        "frequencies_hz": frequencies,
        "mask": mask,
        "tx_xy_m": case.geometry.tx_pos_m[:, ::-1],
        "rx_xy_m": case.geometry.rx_pos_m[:, ::-1],
    }
    axes = {
        "pressure": "frequency,tx,rx",
        "frequencies_hz": "frequency",
        "mask": "tx,rx" if mask.ndim == 2 else "frequency,tx,rx",
        "tx_xy_m": "tx,xy",
        "rx_xy_m": "rx,xy",
    }
    units = {
        "pressure": declared["data_units"],
        "frequencies_hz": "Hz",
        "mask": "1",
        "tx_xy_m": "m",
        "rx_xy_m": "m",
    }
    return write_artifact(path, "wust.frequency_input", metadata, arrays, axes, units)
