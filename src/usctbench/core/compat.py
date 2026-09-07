"""Small standard-library compatibility shims for supported Python versions."""

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum with value serialization, also on Python 3.10."""

    def __str__(self) -> str:
        return self.value
