"""Truth-free acquisition splitting, residual evaluation and stopping support."""

from .data import DataSplit, make_data_split, residual_statistics

__all__ = ["DataSplit", "make_data_split", "residual_statistics"]
