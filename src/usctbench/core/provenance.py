"""Measurement provenance helpers for benchmark cases and runs."""

from __future__ import annotations

from typing import Any

from usctbench.core.compat import StrEnum


class MeasurementProvenance(StrEnum):
    """Allowed measurement provenance labels stored in case/run metadata."""

    ORACLE_TRAVEL_TIME = "oracle_travel_time"
    SPEEDMAP_TRAVEL_TIME_SURROGATE = "speedmap_travel_time_surrogate"
    SELF_SIMULATED_KWAVE_WAVEFIELD = "self_simulated_kwave_wavefield"
    OPENBREASTUS_PRECOMPUTED_WAVEFIELD = "openbreastus_precomputed_wavefield"
    EXTERNAL_MEASUREMENT = "external_measurement"


TRAVEL_TIME_SURROGATE_PROVENANCES = {
    MeasurementProvenance.ORACLE_TRAVEL_TIME.value,
    MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE.value,
}

KWAVE_WAVEFIELD_PROVENANCES = {
    MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value,
    MeasurementProvenance.OPENBREASTUS_PRECOMPUTED_WAVEFIELD.value,
}


def normalize_measurement_provenance(value: Any) -> str:
    """Return a canonical provenance label or raise for unknown values."""

    text = str(value or "").strip()
    allowed = {item.value for item in MeasurementProvenance}
    if text not in allowed:
        raise ValueError(
            f"measurement_provenance must be one of {sorted(allowed)}, got {text!r}"
        )
    return text


def measurement_provenance_from_metadata(metadata: dict[str, Any]) -> str:
    """Infer legacy metadata into the explicit provenance taxonomy."""

    value = metadata.get("measurement_provenance")
    if value:
        return normalize_measurement_provenance(value)
    text = " ".join(
        str(metadata.get(key, ""))
        for key in (
            "benchmark_type",
            "case_type",
            "conversion",
            "feature_provenance",
            "source_dataset",
        )
    ).lower()
    limitations = " ".join(
        str(item) for item in metadata.get("measurement_limitations", [])
    ).lower()
    combined = f"{text} {limitations}"
    if "oracle" in combined:
        return MeasurementProvenance.ORACLE_TRAVEL_TIME.value
    if (
        "openbreastus" in combined
        and "wavefield" in combined
        and "precomputed" in combined
    ):
        return MeasurementProvenance.OPENBREASTUS_PRECOMPUTED_WAVEFIELD.value
    if (
        "external_measurement" in combined
        or "measured" in combined
        and "rf" in combined
    ):
        return MeasurementProvenance.EXTERNAL_MEASUREMENT.value
    if "kwave" in combined or "k-wave" in combined:
        return MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value
    return MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE.value


def default_inverse_crime_risk(provenance: str, metadata: dict[str, Any]) -> str:
    """Classify how strongly a benchmark uses ground-truth-generated data."""

    if provenance == MeasurementProvenance.ORACLE_TRAVEL_TIME.value:
        return "high_oracle"
    if provenance == MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE.value:
        return "high_speedmap_surrogate"
    if provenance == MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value:
        backend = str(
            metadata.get(
                "simulation_backend",
                metadata.get("simulation_metadata", {}).get("backend", ""),
            )
        ).lower()
        return (
            "medium_self_simulated"
            if "kwave" in backend
            else "medium_self_simulated_smoke"
        )
    if provenance == MeasurementProvenance.OPENBREASTUS_PRECOMPUTED_WAVEFIELD.value:
        return "low_precomputed_independent"
    if provenance == MeasurementProvenance.EXTERNAL_MEASUREMENT.value:
        return "low_external_measurement"
    return "unknown"


def case_measurement_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Derive the required run-level measurement metadata from case metadata."""

    provenance = measurement_provenance_from_metadata(metadata)
    uses_kwave = provenance in KWAVE_WAVEFIELD_PROVENANCES
    uses_openbreastus_precomputed = (
        provenance == MeasurementProvenance.OPENBREASTUS_PRECOMPUTED_WAVEFIELD.value
    )
    uses_gt_generated = provenance in {
        MeasurementProvenance.ORACLE_TRAVEL_TIME.value,
        MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE.value,
        MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value,
    }
    feature_source = (
        metadata.get("feature_source")
        or metadata.get("feature_provenance")
        or metadata.get("measurement_domain", "")
    )
    simulation_qc_passed = metadata.get("simulation_qc_passed")
    simulation_failed_qc = bool(metadata.get("simulation_failed_qc", False))
    if simulation_qc_passed is False:
        simulation_failed_qc = True
    return {
        "measurement_provenance": provenance,
        "forward_model": metadata.get(
            "forward_model", metadata.get("simulation_forward_model", "")
        ),
        "feature_channel": metadata.get("feature_channel", ""),
        "picker_method": metadata.get("picker_method", ""),
        "observable_definition": metadata.get("observable_definition", ""),
        "tof_observation_contract": metadata.get("tof_observation_contract"),
        "water_reference_used": metadata.get("water_reference_used", ""),
        "uses_gt_generated_measurement": bool(uses_gt_generated),
        "uses_kwave_wavefield": bool(uses_kwave),
        "uses_openbreastus_precomputed_wavefield": bool(uses_openbreastus_precomputed),
        "uses_complex_wavefield": bool(metadata.get("uses_complex_wavefield", False)),
        "feature_source": str(feature_source),
        "inverse_crime_risk": metadata.get("inverse_crime_risk")
        or default_inverse_crime_risk(provenance, metadata),
        "simulation_qc_passed": simulation_qc_passed,
        "simulation_failed_qc": simulation_failed_qc,
    }


def stamp_measurement_metadata(
    metadata: dict[str, Any] | None,
    *,
    measurement_provenance: MeasurementProvenance | str,
    benchmark_type: str,
    forward_model: str,
    feature_source: str = "",
    uses_complex_wavefield: bool = False,
    simulation_qc_passed: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return metadata with the mandatory provenance fields set."""

    stamped = dict(metadata or {})
    provenance = normalize_measurement_provenance(str(measurement_provenance))
    stamped.update(
        {
            "benchmark_type": benchmark_type,
            "measurement_provenance": provenance,
            "forward_model": forward_model,
            "uses_gt_generated_measurement": provenance
            in {
                MeasurementProvenance.ORACLE_TRAVEL_TIME.value,
                MeasurementProvenance.SPEEDMAP_TRAVEL_TIME_SURROGATE.value,
                MeasurementProvenance.SELF_SIMULATED_KWAVE_WAVEFIELD.value,
            },
            "uses_kwave_wavefield": provenance in KWAVE_WAVEFIELD_PROVENANCES,
            "uses_openbreastus_precomputed_wavefield": provenance
            == MeasurementProvenance.OPENBREASTUS_PRECOMPUTED_WAVEFIELD.value,
            "uses_complex_wavefield": bool(uses_complex_wavefield),
            "feature_source": feature_source,
        }
    )
    if simulation_qc_passed is not None:
        stamped["simulation_qc_passed"] = bool(simulation_qc_passed)
        stamped["simulation_failed_qc"] = not bool(simulation_qc_passed)
    stamped["inverse_crime_risk"] = stamped.get(
        "inverse_crime_risk"
    ) or default_inverse_crime_risk(provenance, stamped)
    if extra:
        stamped.update(extra)
    return stamped
