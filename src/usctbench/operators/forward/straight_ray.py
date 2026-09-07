"""Siddon cell-intersection forward operator for straight-ray physics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from usctbench.core.schema import GeometrySpec, GridSpec, USCTCase


@dataclass(frozen=True)
class StraightRayProjector:
    """Line-integral projector over a Cartesian grid.

    Coordinates use the project-wide convention `[y, x]` in meters.
    """

    grid: GridSpec
    tx_pos_m: np.ndarray
    rx_pos_m: np.ndarray
    indices_by_ray: tuple[np.ndarray, ...]
    lengths_by_ray_m: tuple[np.ndarray, ...]

    @classmethod
    def from_case(cls, case: USCTCase) -> "StraightRayProjector":
        return cls.from_grid_geometry(case.grid, case.geometry)

    @classmethod
    def from_grid_geometry(
        cls, grid: GridSpec, geometry: GeometrySpec
    ) -> "StraightRayProjector":
        indices: list[np.ndarray] = []
        lengths: list[np.ndarray] = []
        for tx in geometry.tx_pos_m:
            for rx in geometry.rx_pos_m:
                ray_indices, ray_lengths = _trace_ray(grid, tx, rx)
                indices.append(ray_indices)
                lengths.append(ray_lengths)
        return cls(
            grid=grid,
            tx_pos_m=np.asarray(geometry.tx_pos_m, dtype=float),
            rx_pos_m=np.asarray(geometry.rx_pos_m, dtype=float),
            indices_by_ray=tuple(indices),
            lengths_by_ray_m=tuple(lengths),
        )

    @property
    def n_rays(self) -> int:
        return len(self.indices_by_ray)

    @property
    def n_pixels(self) -> int:
        ny, nx = self.grid.shape
        return ny * nx

    @property
    def ray_shape(self) -> tuple[int, int]:
        return (self.tx_pos_m.shape[0], self.rx_pos_m.shape[0])

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Compute line integrals for all source-receiver pairs."""

        flat = np.asarray(image, dtype=float).reshape(-1)
        if flat.size != self.n_pixels:
            raise ValueError(f"image has {flat.size} pixels, expected {self.n_pixels}")
        out = np.zeros(self.n_rays, dtype=float)
        for ray_id, (indices, lengths) in enumerate(
            zip(self.indices_by_ray, self.lengths_by_ray_m, strict=True)
        ):
            if indices.size:
                out[ray_id] = float(np.dot(lengths, flat[indices]))
        return out

    def adjoint(self, ray_values: np.ndarray) -> np.ndarray:
        """Backproject ray values using the exact transpose of `forward`."""

        from usctbench.operators.adjoint.straight_ray import backproject

        return backproject(self, ray_values)

    def row_norms(self, power: int = 2) -> np.ndarray:
        """Return per-ray sums of path lengths raised to `power`."""

        return np.array(
            [float(np.sum(lengths**power)) for lengths in self.lengths_by_ray_m]
        )

    def col_norms(self, power: int = 2) -> np.ndarray:
        """Return per-pixel sums of path lengths raised to `power`."""

        flat = np.zeros(self.n_pixels, dtype=float)
        for indices, lengths in zip(
            self.indices_by_ray, self.lengths_by_ray_m, strict=True
        ):
            if indices.size:
                np.add.at(flat, indices, lengths**power)
        return flat.reshape(self.grid.shape)


def _trace_ray(
    grid: GridSpec, tx_yx: np.ndarray, rx_yx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Trace one segment through the grid and return flat cell indices plus lengths."""

    y0, x0 = (float(tx_yx[0]), float(tx_yx[1]))
    y1, x1 = (float(rx_yx[0]), float(rx_yx[1]))
    dy_total = y1 - y0
    dx_total = x1 - x0
    segment_length = float(np.hypot(dy_total, dx_total))
    if segment_length == 0.0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    ny, nx = grid.shape
    dy, dx = grid.spacing_m
    origin_y, origin_x = grid.origin_m
    y_edges = origin_y + np.arange(ny + 1) * dy
    x_edges = origin_x + np.arange(nx + 1) * dx

    t_values = [0.0, 1.0]
    if dy_total != 0.0:
        t_values.extend(float((edge - y0) / dy_total) for edge in y_edges)
    if dx_total != 0.0:
        t_values.extend(float((edge - x0) / dx_total) for edge in x_edges)

    t = np.array([value for value in t_values if 0.0 <= value <= 1.0], dtype=float)
    if t.size < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    t = np.unique(np.round(t, decimals=15))

    indices: list[int] = []
    lengths: list[float] = []
    for start, end in zip(t[:-1], t[1:], strict=True):
        if end <= start:
            continue
        mid = 0.5 * (start + end)
        y_mid = y0 + mid * dy_total
        x_mid = x0 + mid * dx_total
        iy = int(np.floor((y_mid - origin_y) / dy))
        ix = int(np.floor((x_mid - origin_x) / dx))
        if 0 <= iy < ny and 0 <= ix < nx:
            indices.append(iy * nx + ix)
            lengths.append(segment_length * (end - start))

    if not indices:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return np.asarray(indices, dtype=int), np.asarray(lengths, dtype=float)
