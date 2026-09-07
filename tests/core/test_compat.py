from usctbench.core.provenance import MeasurementProvenance
from usctbench.core.schema import GeometryType, MeasurementDomain, ResultStatus


def test_string_enums_serialize_as_values_on_supported_python_versions():
    for enum in (GeometryType, MeasurementDomain, ResultStatus, MeasurementProvenance):
        for member in enum:
            assert isinstance(member, str)
            assert str(member) == member.value
            assert enum(member.value) is member
